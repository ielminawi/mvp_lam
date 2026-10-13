"""Exploratory three-segment action probes on existing frozen LAM features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.check_probe_bottleneck import ALPHAS, ridge_fit, train_mlp
from scripts.probe_temporal_pilot import SEEDS, VARIANTS
from scripts.temporal_pilot import sha256
from scripts.diagnose_temporal_intermediates import paired_interval

GROUPS = {'all_21': list(range(21)),
          'translation': [7*s+d for s in range(3) for d in range(3)],
          'orientation': [7*s+d for s in range(3) for d in range(3, 6)],
          'gripper': [6, 13, 20],
          **{name: list(range(7*s, 7*s+7)) for s,name in enumerate(('A_to_B', 'B_to_C', 'C_to_D'))}}


def segment_targets(states, gripper, frames):
    frames = np.asarray(frames)
    if frames.ndim != 2 or frames.shape[1] != 4 or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError('Expected integer original frame indices [N,4]')
    if not np.all(np.diff(frames, axis=1) == 3) or frames.min() < 1 or frames.max() >= len(states):
        raise ValueError('Expected three consecutive H=3 segments within episode')
    targets = np.stack([np.column_stack((states[frames[:, s+1], :6]-states[frames[:, s], :6],
                                        gripper[frames[:, s+1]-1])) for s in range(3)], axis=1)
    if not np.isfinite(targets).all() or not np.all(np.isin(targets[:, :, 6], [0, 1])):
        raise ValueError('Invalid measured-state/gripper targets')
    return targets.reshape(len(frames), 21).astype(np.float64)


def score(predictions, truth, baseline):
    result = {}
    for name, dims in GROUPS.items():
        variance = np.var(truth[:, dims], axis=0).sum()
        errors = [float(np.mean(np.sum((p[:, dims]-truth[:, dims])**2, axis=1))) for p in predictions]
        base_error = float(np.mean(np.sum((baseline[dims]-truth[:, dims])**2, axis=1)))
        values = [e/variance for e in errors] if variance > 1e-12 else None
        result[name] = {'nmse_mean': float(np.mean(values)) if values is not None else None,
                        'nmse_seed_sd': float(np.std(values, ddof=1)) if values is not None and len(values)>1 else None,
                        'baseline_nmse': base_error/variance if variance > 1e-12 else None,
                        'error_reduction_percent': 100*(1-np.mean(errors)/base_error) if base_error > 1e-12 else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sample-dir', 'features-dir', 'probe-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads((args.sample_dir/'manifest.json').read_text())
    feature_manifest = json.loads((args.features_dir/'manifest.json').read_text())
    if sha256(args.sample_dir/'manifest.json') != feature_manifest['sample_manifest_sha256']:
        raise ValueError('Prepared sample provenance mismatch')
    original = json.loads((args.probe_dir/'results.json').read_text())
    if original['feature_manifest'] != feature_manifest:
        raise ValueError('Original probe/features provenance mismatch')
    entry_by_record = {e['record_index']: e for e in manifest['episodes']}
    datasets, targets, episode_hashes = {}, {}, {}
    validation_errors = {}
    for variant in VARIANTS:
        datasets[variant] = {}
        for split in ('train', 'validation'):
            with np.load(args.features_dir/variant/f'{split}.npz', allow_pickle=False) as data:
                datasets[variant][split] = {k: data[k] for k in data.files}
            data = datasets[variant][split]
            if data['features'].shape != (len(data['record_indices']), 512) or not np.isfinite(data['features']).all():
                raise ValueError('Invalid frozen features')
            if variant != 'released':
                for key in ('record_indices', 'frame_indices', 'targets_raw'):
                    np.testing.assert_array_equal(data[key], datasets['released'][split][key])
                continue
            labels = np.empty((len(data['features']), 21))
            for record in np.unique(data['record_indices']):
                entry = entry_by_record[int(record)]
                if entry['split'] != split:
                    raise ValueError('Wrong episode split')
                selected = data['record_indices'] == record
                path = args.sample_dir/entry['file']
                episode_hashes[entry['file']] = sha256(path)
                # Only train/validation entries are opened; RGB arrays are not loaded.
                with np.load(path, allow_pickle=False) as episode:
                    states, gripper = episode['states'], episode['gripper_binarized']
                    frames = data['frame_indices'][selected]
                    labels[selected] = segment_targets(states, gripper, frames)
                    net = data['targets_raw'][selected]
                    np.testing.assert_allclose(labels[selected].reshape(-1, 3, 7)[:, :, :6].sum(1), net[:, :6], atol=1e-6, rtol=1e-5)
                    np.testing.assert_array_equal(labels[selected, 20], net[:, 6])
            targets[split] = labels
    train_records = set(datasets['released']['train']['record_indices'])
    val_records = set(datasets['released']['validation']['record_indices'])
    if train_records & val_records:
        raise ValueError('Record leakage')
    sources = {s: {entry_by_record[int(r)]['source_path'] for r in datasets['released'][s]['record_indices']}
               for s in ('train', 'validation')}
    if sources['train'] & sources['validation']:
        raise ValueError('Source episode leakage')
    mean, std = targets['train'].mean(0), targets['train'].std(0)
    scale = np.where(std > 1e-8, std, 1.)
    y = {s: (targets[s]-mean)/scale for s in targets}
    baseline = y['train'].mean(0)
    args.output_dir.mkdir(parents=True)
    np.savez(args.output_dir/'targets.npz', target_mean=mean, target_scale=scale,
             **{f'{s}_{k}': v for s in targets for k,v in {
                 'raw': targets[s], 'record_indices': datasets['released'][s]['record_indices'],
                 'frame_indices': datasets['released'][s]['frame_indices']}.items()})
    report = {'scope': 'Frozen-feature path-sensitive probing; no LAM training or test access.',
              'target': '21 outputs: A->B, B->C, C->D; H=3 each. Six stored-state differences plus binarized gripper command at segment end-1. No angle wrapping, same convention as existing targets.',
              'normalization': 'Empirical mean/std per output fitted on training windows only; constant dimensions use scale1. Different from prior horizon-aware endpoint normalization; absolute NMSE across tasks is not directly comparable.',
              'protocol': {'ridge': 'PCA128, sum-SSE ridge; unpenalized intercept; alpha grid selected by validation MSE', 'alphas': ALPHAS,
                           'mlp': 'PCA128->64 ReLU->21; Adam .001 WD .0001 batch512; max200epochs patience20; best validation MSE', 'seeds': SEEDS,
                           'shuffle': 'training 21D labels shuffled as complete rows, seed20261013; correct validation targets'},
              'limitations': 'Exploratory validation-tuned results, not independent test performance. Four-frame encoder observes intermediate endpoints; any gain is observed-path information, not forecasting. Probe seeds do not cover LAM uncertainty. Segment actions describe coarse H=3 net movement, not all one-step actions.',
              'feature_manifest': feature_manifest, 'sample_episode_sha256': episode_hashes, 'variants': {}}
    permutation = np.random.default_rng(20261013).permutation(len(y['train']))
    for variant in VARIANTS:
        dest = args.output_dir/variant; dest.mkdir()
        with np.load(args.probe_dir/variant/'preprocessing.npz', allow_pickle=False) as data:
            center, components = data['feature_mean'], data['pca_components']
        x = {s: (datasets[variant][s]['features'].astype(np.float64)-center) @ components.T for s in targets}
        trials = []
        for alpha in ALPHAS:
            weight, bias = ridge_fit(x['train'], y['train'], alpha)
            trials.append((float(np.mean((x['validation']@weight+bias-y['validation'])**2)), alpha, weight, bias))
        _, alpha, weight, bias = min(trials, key=lambda row: row[0])
        result = {'ridge': {'alpha': alpha, 'grid': [{'alpha': t[1], 'validation_mse': t[0]} for t in trials]}}
        validation_errors[variant] = {}
        predictions = {}
        for split in targets:
            prediction = x[split]@weight+bias
            predictions[f'{split}_ridge'] = prediction
            result['ridge'][split] = score([prediction], y[split], baseline)
        np.savez(dest/'ridge.npz', weight=weight, bias=bias)
        for method, shuffled in (('mlp', False), ('shuffled_mlp', True)):
            collected = {s: [] for s in targets}; selections = []
            for seed in SEEDS:
                state, pred, selection = train_mlp(x['train'], y['train'][permutation] if shuffled else y['train'],
                                                  x['validation'], y['validation'], seed)
                torch.save(state, dest/f'{method}_seed_{seed}.pt')
                selections.append({'seed': seed, **selection})
                for split in targets:
                    collected[split].append(pred[split])
                    predictions[f'{split}_{method}_seed_{seed}'] = pred[split]
                print(f"{variant}/{method}/seed{seed}: validation NMSE={score([pred['validation']], y['validation'], baseline)['all_21']['nmse_mean']:.6f}", flush=True)
            result[method] = {'selection': selections, **{s: score(collected[s], y[s], baseline) for s in targets}}
            if method == 'mlp':
                for group, dims in GROUPS.items():
                    variance = float(np.var(y['validation'][:, dims], axis=0).sum())
                    if variance > 1e-12:
                        validation_errors[variant][group] = np.mean([np.sum((p[:, dims]-y['validation'][:, dims])**2, axis=1) for p in collected['validation']], axis=0)/variance
        np.savez_compressed(dest/'predictions.npz', **predictions)
        np.savez(dest/'preprocessing.npz', feature_mean=center, pca_components=components, target_mean=mean, target_scale=scale)
        report['variants'][variant] = result
    report['paired_mlp_four_minus_two'] = {g: paired_interval(validation_errors['four_frame'][g]-validation_errors['two_frame'][g], datasets['released']['validation']['record_indices']) for g in validation_errors['four_frame']}
    report['paired_interval_definition'] = 'Positive means four-frame worse; equal-episode mean normalized squared-error differences, fixed full-validation denominator, episode bootstrap2000 draws seed20261013. Validation tuning is not accounted for; exploratory, not confirmatory.'
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('\nVALIDATION: three-segment path prediction (NMSE; lower better)')
    for variant, result in report['variants'].items():
        print(variant)
        for method in ('ridge', 'mlp', 'shuffled_mlp'):
            print(' ', method, ':', ', '.join(f"{g}={v['nmse_mean']:.6f}" if v['nmse_mean'] is not None else f'{g}=undefined' for g,v in result[method]['validation'].items()))
        print('  Mean-action baseline:', result['ridge']['validation']['all_21']['baseline_nmse'])
    print('Paired MLP four-minus-two (all21):', report['paired_mlp_four_minus_two'].get('all_21'))
    print('Saved', args.output_dir, '; no LAM fitting or test access.')


if __name__ == '__main__':
    main()
