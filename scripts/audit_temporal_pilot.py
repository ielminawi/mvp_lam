"""Audit completed temporal pilots using saved train/validation artifacts only."""
import argparse
import json
from pathlib import Path

import numpy as np


def usage(counts):
    counts = np.asarray(counts)
    if counts.shape != (16,) or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError('Expected 16 finite nonnegative code counts')
    if not np.equal(counts, np.floor(counts)).all() or counts.sum() == 0:
        raise ValueError('Expected nonempty integer code counts')
    p = counts[counts > 0] / counts.sum()
    entropy = float(-(p * np.log(p)).sum())
    return dict(counts=counts.tolist(), distinct=int(len(p)), perplexity=float(np.exp(entropy)),
                largest_fraction=float(p.max()), entropy_nats=entropy)


def inspect(report):
    if not report['frozen_weights_unchanged']:
        raise ValueError('Frozen weights changed')
    cfg, history = report['config'], report['history']
    if [r['update'] for r in history] != list(range(cfg['updates'] + 1)):
        raise ValueError('Incomplete or unordered update history')
    rows = history[1:]
    for key in ('training_total', 'training_reconstruction', 'training_commitment', 'gradient_norm_before_clip'):
        if not np.isfinite([r[key] for r in rows]).all():
            raise ValueError(f'Nonfinite {key}')
    validations = [r for r in history if 'validation' in r]
    expected = [0] + [i for i in range(1, cfg['updates'] + 1)
                       if i % cfg['validation_every'] == 0 or i == cfg['updates']]
    if [r['update'] for r in validations] != expected:
        raise ValueError('Missing validation checkpoint')
    for row in validations:
        v = row['validation']
        if v['pairs'] != report['validation_pairs'] or sum(v['code_counts']) != 4 * v['pairs']:
            raise ValueError('Validation pair/code count mismatch')
        if not np.isfinite([v['reconstruction_mse'], v['copy_initial_mse']]).all():
            raise ValueError('Nonfinite validation error')
        usage(v['code_counts'])
    first, last = validations[0]['validation'], validations[-1]['validation']
    best = min(validations, key=lambda r: r['validation']['reconstruction_mse'])
    return dict(initial_mse=first['reconstruction_mse'], final_mse=last['reconstruction_mse'],
                reduction_from_initial_percent=100 * (1-last['reconstruction_mse']/first['reconstruction_mse']),
                best_logged_update=best['update'], best_logged_mse=best['validation']['reconstruction_mse'],
                clipping_fraction=float(np.mean([r['gradient_norm_before_clip'] > cfg['gradient_clip'] for r in rows])),
                example_presentations=cfg['updates'] * cfg['micro_batch_size'] * cfg['accumulation_steps'],
                equivalent_training_passes=cfg['updates'] * cfg['micro_batch_size'] * cfg['accumulation_steps']/report['train_pairs'],
                initial_code_usage=usage(first['code_counts']), final_code_usage=usage(last['code_counts']))


def feature_usage(path):
    with np.load(path) as data:
        ids = data['indices']
    if ids.ndim != 2 or ids.shape[1] != 4 or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= 16)):
        raise ValueError(f'Invalid four-slot indices: {path}')
    _, counts = np.unique(ids, axis=0, return_counts=True)
    p = counts / counts.sum()
    return dict(pairs=len(ids), marginal=usage(np.bincount(ids.ravel(), minlength=16)),
                per_slot=[usage(np.bincount(ids[:, i], minlength=16)) for i in range(4)],
                distinct_four_code_tuples=len(counts), tuple_perplexity=float(np.exp(-(p*np.log(p)).sum())))


def plot(reports, output, window):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, report in reports.items():
        history, cfg = report['history'], report['config']
        val = [r for r in history if 'validation' in r]
        axes[0, 0].plot([r['update'] for r in val], [r['validation']['reconstruction_mse'] for r in val], 'o-', label=name)
        rows = history[1:]
        for key, ax in [('training_reconstruction', axes[0, 1]), ('training_commitment', axes[1, 0])]:
            y = np.array([r[key] for r in rows])
            size = min(window, len(y))
            ax.plot([r['update'] for r in rows][size-1:], np.convolve(y, np.ones(size)/size, 'valid'), label=name)
        axes[1, 1].plot([r['update'] for r in rows], [r['gradient_norm_before_clip'] for r in rows], alpha=.7, label=name)
    titles = ['Validation reconstruction MSE (full split)', f'Training reconstruction MSE ({window}-update trailing mean)',
              f'Training commitment MSE ({window}-update trailing mean)', 'Gradient norm before clipping']
    axes[1, 1].axhline(next(iter(reports.values()))['config']['gradient_clip'], color='black', linestyle=':', label='Clip threshold')
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title); ax.set_xlabel('Optimizer update'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Temporal pilot audit — changing training batches; fixed validation split')
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'training_curves.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, (name, report) in zip(axes, reports.items()):
        val = [r['validation'] for r in report['history'] if 'validation' in r]
        for offset, v, label in [(-.2, val[0], 'Initial'), (.2, val[-1], 'Final')]:
            counts = np.asarray(v['code_counts'])
            ax.bar(np.arange(16)+offset, counts/counts.sum(), width=.4, label=label)
        ax.set_title(name); ax.set_xlabel('Code ID'); ax.set_ylabel('Fraction of all slot assignments'); ax.set_xticks(range(16)); ax.legend()
    fig.suptitle('Marginal code usage across four slots (not joint tuple diversity)')
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'code_usage.{ext}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--two-frame-dir', type=Path, required=True)
    parser.add_argument('--four-frame-dir', type=Path, required=True)
    parser.add_argument('--features-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoothing-window', type=int, default=25)
    args = parser.parse_args()
    if args.smoothing_window < 1:
        parser.error('Smoothing window must be positive')
    reports = {name: json.loads((folder/'results.json').read_text()) for name, folder in
               [('two_frame', args.two_frame_dir), ('four_frame', args.four_frame_dir)]}
    a, b = reports.values()
    for key in ('config', 'cache_manifest_sha256', 'train_pairs', 'validation_pairs', 'frozen_weight_sha256'):
        if a[key] != b[key]:
            raise ValueError(f'Unmatched runs: {key}')
    summaries = {}
    for name, report in reports.items():
        if report['variant'] != name:
            raise ValueError('Wrong variant')
        summaries[name] = inspect(report)
    if args.features_dir:
        for name in ('released', 'two_frame', 'four_frame'):
            for split in ('train', 'validation'):
                stats = feature_usage(args.features_dir/name/f'{split}.npz')
                expected = a[f'{split}_pairs']
                if stats['pairs'] != expected:
                    raise ValueError('Feature count mismatch')
                if name in summaries and split == 'validation' and stats['marginal']['counts'] != summaries[name]['final_code_usage']['counts']:
                    raise ValueError('Extracted indices disagree with final validation log')
                summaries.setdefault(name, {}).setdefault('extracted_code_usage', {})[split] = stats
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plot(reports, args.output_dir, args.smoothing_window)
    result = dict(runs=summaries, sources={n: str(p/'results.json') for n,p in
                  [('two_frame', args.two_frame_dir), ('four_frame', args.four_frame_dir)]},
                  interpretation='Single adaptation seed. Small curve changes do not prove convergence. Best logged update is descriptive; final checkpoint selection unchanged. Code usage alone cannot establish useful actions or rule out representation collapse. No test access or model changes.')
    (args.output_dir/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for name in reports:
        s = summaries[name]; codes = s['final_code_usage']
        print(f"{name}: validation {s['initial_mse']:.8f} -> {s['final_mse']:.8f}; reduction {s['reduction_from_initial_percent']:.2f}%")
        print(f"  Training passes: {s['equivalent_training_passes']:.3f}; clipped updates: {100*s['clipping_fraction']:.1f}%")
        print(f"  Final marginal codes: {codes['distinct']}/16; perplexity {codes['perplexity']:.2f}; largest share {100*codes['largest_fraction']:.1f}%")
        if 'extracted_code_usage' in s:
            u = s['extracted_code_usage']['validation']
            print(f"  Validation per-slot perplexities: {[round(x['perplexity'], 2) for x in u['per_slot']]}; distinct tuples: {u['distinct_four_code_tuples']}")
    print(f'Saved report and plots: {args.output_dir}. No fitting, checkpoint changes or test access.')


if __name__ == '__main__':
    main()
