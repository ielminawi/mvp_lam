"""Analyze saved intervention action predictions; no inference or fitting."""
import argparse
import json
from pathlib import Path
import numpy as np


def changes(base, prediction, truth, records):
    delta = prediction-base
    error_delta = ((prediction-truth[:, None])**2-(base-truth[:, None])**2).sum(2).mean(1)
    episode_deltas = np.array([error_delta[records == r].mean() for r in np.unique(records)])
    return {'normalized_prediction_rmse_change': float(np.sqrt(np.mean(delta**2))),
            'normalized_mean_absolute_prediction_change': float(np.mean(np.abs(delta))),
            'normalized_action_squared_error_difference': float(error_delta.mean()),
            'episodes_improved': int((episode_deltas < -1e-10).sum()),
            'episodes_worsened': int((episode_deltas > 1e-10).sum()),
            'episodes_unchanged': int((np.abs(episode_deltas) <= 1e-10).sum()),
            'per_episode_error_differences': episode_deltas.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic-dir', 'probe-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    from scripts.temporal_pilot import sha256
    diagnostic = json.loads((args.diagnostic_dir/'results.json').read_text())
    if diagnostic['probe_results_sha256'] != sha256(args.probe_dir/'results.json'):
        raise ValueError('Probe provenance mismatch')
    with np.load(args.probe_dir/'four_frame/preprocessing.npz', allow_pickle=False) as data:
        mean, scale = data['target_mean'], data['target_scale']
    with np.load(args.diagnostic_dir/'per_window.npz', allow_pickle=False) as data:
        truth = (data['targets_raw']-mean)/scale
        records = data['record_indices']
        base = data['correct_predictions_normalized']
        results = {}
        groups = {'all_7': list(range(7)), 'translation': [0, 1, 2], 'orientation': [3, 4, 5], 'gripper': [6]}
        for condition in ('repeat_endpoints', 'reverse_middle'):
            prediction = data[f'{condition}_predictions_normalized']
            results[condition] = {g: changes(base[:, :, dims], prediction[:, :, dims], truth[:, dims], records) for g,dims in groups.items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'conditions': results, 'episode_record_order': np.unique(records).tolist(),
              'interpretation': 'Prediction shifts are measured in training-normalized action units, not raw physical units. Errors average four fixed probes; positive error difference means worse. Episode counts do not establish statistical significance. Decoder output differences are not available in the saved diagnostic and are not inferred from its MSE.'}
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for condition, groups in results.items():
        print(condition)
        for group, stats in groups.items():
            print(f"  {group}: prediction change RMSE={stats['normalized_prediction_rmse_change']:.6f}; episodes improved/worsened={stats['episodes_improved']}/{stats['episodes_worsened']}")
    print('Saved', args.output_dir/'results.json')


if __name__ == '__main__':
    main()
