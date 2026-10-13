"""Fixed-endpoint interventions on a frozen four-frame model; validation only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.temporal_pilot import Windows, frozen_hash, load_model, provenance, sha256
from scripts.probe_temporal_pilot import SEEDS, overlay_adaptation

CONDITIONS = {'correct': [0, 1, 2, 3], 'repeat_endpoints': [0, 0, 3, 3],
              'reverse_middle': [0, 2, 1, 3]}


def intervene(patches, condition):
    return patches[:, CONDITIONS[condition]]


def paired_interval(delta, records, draws=2000):
    """Equal-episode mean paired difference and episode-bootstrap interval."""
    means = np.array([delta[records == r].mean() for r in np.unique(records)])
    rng = np.random.default_rng(20261013)
    samples = means[rng.integers(len(means), size=(draws, len(means)))].mean(axis=1)
    return {'episode_mean_difference': float(means.mean()),
            'episode_bootstrap_95_percent_interval': np.quantile(samples, [.025, .975]).tolist()}


def probe_predict(features, preprocessing, weights):
    projected = (features.astype(np.float64)-preprocessing['feature_mean']) @ preprocessing['pca_components'].T
    x = torch.tensor(projected, dtype=torch.float32)
    return np.stack([torch.nn.functional.linear(x, w['weight'], w['bias']).numpy() for w in weights], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir', 'checkpoint', 'dino-dir', 'four-frame-dir', 'probe-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise ValueError('Positive batch size and GPU allocation required')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads((args.cache_dir/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['offsets'] != [0, 3, 6, 9] or manifest['limit_episodes_per_split']:
        raise ValueError('Need complete full temporal cache')
    if any(e['split'] not in ('train', 'validation') for e in manifest['episodes']):
        raise ValueError('Unexpected cache split')
    sources = {s: {e['source_path'] for e in manifest['episodes'] if e['split'] == s}
               for s in ('train', 'validation')}
    if sources['train'] & sources['validation']:
        raise ValueError('Source episode leakage')
    run = json.loads((args.four_frame_dir/'results.json').read_text())
    probe_report = json.loads((args.probe_dir/'results.json').read_text())
    feature_manifest = probe_report['feature_manifest']
    adapted_path = args.four_frame_dir/'latest.pt'
    for r in (run, feature_manifest):
        if r['cache_manifest_sha256'] != sha256(args.cache_dir/'manifest.json'):
            raise ValueError('Cache provenance mismatch')
    if feature_manifest['variants']['four_frame']['adapted_checkpoint_sha256'] != sha256(adapted_path):
        raise ValueError('Probe belongs to another adapted checkpoint')
    if not run['frozen_weights_unchanged'] or run['variant'] != 'four_frame':
        raise ValueError('Wrong/incomplete training report')
    labels, records, frames = [], [], []
    for entry in manifest['episodes']:
        if entry['split'] != 'validation':
            continue
        for field, digest in (('patch_file', 'patch_sha256'), ('window_file', 'window_sha256')):
            if sha256(args.cache_dir/entry[field]) != entry[digest]:
                raise ValueError('Changed validation cache')
        with np.load(args.cache_dir/entry['window_file'], allow_pickle=False) as data:
            labels.append(data['targets_raw']); frames.append(data['original_frame_indices'])
            records.append(np.full(len(data['windows']), entry['record_index']))
    truth, records, frames = np.concatenate(labels), np.concatenate(records), np.concatenate(frames)
    with np.load(args.probe_dir/'four_frame/preprocessing.npz', allow_pickle=False) as data:
        prep = {k: data[k] for k in data.files}
    if any(not np.isfinite(prep[k]).all() for k in ('feature_mean', 'pca_components', 'target_mean', 'target_scale')) or not (prep['target_scale'] > 0).all():
        raise ValueError('Invalid saved preprocessing')
    truth_norm = (truth-prep['target_mean'])/prep['target_scale']
    denominator = float(np.var(truth_norm, axis=0).sum())
    if denominator <= 0:
        raise ValueError('Zero target variance')
    weights = [torch.load(args.probe_dir/f'four_frame/seed_{s}.pt', map_location='cpu', weights_only=True) for s in SEEDS]
    model = load_model(args).configure_pilot()
    saved = torch.load(adapted_path, map_location='cpu', weights_only=True)
    if saved['config'] != run['config']:
        raise ValueError('Checkpoint configuration mismatch')
    overlay_adaptation(model, saved, 'four_frame')
    if frozen_hash(model) != run['frozen_weight_sha256']:
        raise ValueError('Frozen base weights mismatch')
    model.requires_grad_(False).eval()
    before = frozen_hash(model)
    dataset = Windows(args.cache_dir, manifest, 'validation')
    arrays = {c: {k: [] for k in ('continuous', 'indices', 'features', 'reconstruction_error')} for c in CONDITIONS}
    with torch.no_grad():
        for first in range(0, len(dataset), args.batch_size):
            patches = dataset.batch(range(first, min(first+args.batch_size, len(dataset))), 'four_frame')
            for condition in CONDITIONS:
                output = model.reconstruct_window(intervene(patches, condition))
                values = {'continuous': output['emb'], 'indices': output['indices'],
                          'features': output['z_q'].flatten(1),
                          'reconstruction_error': (output['recon']-patches[:, -1:]).square().flatten(1).mean(1)}
                for key, value in values.items():
                    if not torch.isfinite(value).all():
                        raise ValueError('Nonfinite output')
                    arrays[condition][key].append(value.cpu().numpy())
            print(f'Validation: {min(first+args.batch_size, len(dataset))}/{len(dataset)}', flush=True) if first % (args.batch_size*100) == 0 else None
    if frozen_hash(model) != before:
        raise ValueError('Model parameters changed')
    for fields in arrays.values():
        for key in fields:
            fields[key] = np.concatenate(fields[key])
        fields['predictions_normalized'] = probe_predict(fields['features'], prep, weights)
        fields['action_error'] = ((fields['predictions_normalized']-truth_norm[:, None])**2).sum(axis=2).mean(axis=1)
    # Reproduce saved correct-input predictions before interpreting interventions.
    with np.load(args.probe_dir/'four_frame/predictions.npz', allow_pickle=False) as reference:
        np.testing.assert_array_equal(truth, reference['validation_targets_raw'])
        for i, seed in enumerate(SEEDS):
            expected = (reference[f'validation_seed_{seed}_raw']-prep['target_mean'])/prep['target_scale']
            np.testing.assert_allclose(arrays['correct']['predictions_normalized'][:, i], expected, rtol=1e-4, atol=1e-4,
                                       err_msg='Correct-input predictions do not reproduce saved probe; investigate before interpretation')
    summaries = {}
    base = arrays['correct']
    for condition, fields in arrays.items():
        change = fields['continuous']-base['continuous']
        relative = np.linalg.norm(change.reshape(len(truth), -1), axis=1)/np.maximum(np.linalg.norm(base['continuous'].reshape(len(truth), -1), axis=1), 1e-12)
        codes_changed = fields['indices'] != base['indices']
        fields['continuous_relative_l2'] = relative
        summaries[condition] = {
            'continuous_relative_l2_mean': float(relative.mean()),
            'continuous_mse': float(np.mean(change**2)),
            'per_slot_code_change_fraction': codes_changed.mean(axis=0).tolist(),
            'any_code_change_fraction': float(codes_changed.any(axis=1).mean()),
            'reconstruction_mse': float(fields['reconstruction_error'].mean()),
            'action_nmse_mean_over_probe_seeds': float(fields['action_error'].mean()/denominator),
            'paired_reconstruction': paired_interval(fields['reconstruction_error']-base['reconstruction_error'], records),
            'paired_action_nmse': paired_interval((fields['action_error']-base['action_error'])/denominator, records)}
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir/'per_window.npz', record_indices=records, frame_indices=frames, targets_raw=truth,
                        **{f'{c}_{k}': v for c,d in arrays.items() for k,v in d.items()})
    report = {'conditions': summaries, 'pairs': len(truth), 'episodes': len(np.unique(records)),
              'provenance': provenance(args), 'adapted_checkpoint_sha256': sha256(adapted_path),
              'probe_results_sha256': sha256(args.probe_dir/'results.json'),
              'scope': 'Validation only; fixed endpoints; frozen final four-frame checkpoint, PCA and four probes. No fitting.',
              'interpretation': 'Positive paired error difference means intervention is worse. Bootstrap resamples episodes, uses equal-episode means and a fixed full-validation NMSE denominator; not LAM-seed uncertainty. Perturbed sequences may be out of distribution. Sensitivity is not proof of beneficial temporal reasoning.'}
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for condition, summary in summaries.items():
        print(f"{condition}: continuous relative L2={summary['continuous_relative_l2_mean']:.5f}; any code changed={summary['any_code_change_fraction']:.2%}; recon MSE={summary['reconstruction_mse']:.8f}; action NMSE={summary['action_nmse_mean_over_probe_seeds']:.6f}")
        if condition != 'correct':
            for key in ('paired_reconstruction', 'paired_action_nmse'):
                print(f'  {key}: {summary[key]}')
    print(f'Saved {args.output_dir}. No training, model changes or test access.')


if __name__ == '__main__':
    main()
