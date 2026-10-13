"""Exploratory probe-capacity comparisons on frozen temporal features; no test access."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.probe_temporal_pilot import GROUPS, SEEDS, VARIANTS, group_scores

ALPHAS = [.01, .1, 1., 10., 100., 1000.]
GROUP_NAMES = ('all_7', 'translation', 'orientation', 'gripper')


def ridge_fit(x, y, alpha):
    """Sum-squared error + alpha*||W||², with an unpenalized intercept."""
    center, target = x.mean(0), y.mean(0)
    xc = x-center
    weights = np.linalg.solve(xc.T @ xc+alpha*np.eye(x.shape[1]), xc.T @ (y-target))
    return weights, target-center@weights


def train_mlp(x, y, xv, yv, seed, epochs=200, patience=20):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, y.shape[1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    gen = torch.Generator().manual_seed(seed+100)
    x, y, xv = [torch.tensor(a, dtype=torch.float32) for a in (x, y, xv)]
    best, best_epoch, state, stale = float('inf'), 0, None, 0
    for epoch in range(1, epochs+1):
        model.train()
        for batch in torch.randperm(len(x), generator=gen).split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x[batch]), y[batch])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite MLP training loss')
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(xv).numpy().astype(np.float64)
        score = float(np.mean((prediction-yv)**2))
        if score < best:
            best, best_epoch = score, epoch
            state = {k: v.detach().clone() for k,v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    with torch.no_grad():
        predictions = {'train': model(x).numpy().astype(np.float64), 'validation': model(xv).numpy().astype(np.float64)}
    return state, predictions, {'selected_epoch': best_epoch, 'epochs_run': epoch}


def summarize(predictions, truth, baseline):
    rows = [group_scores(p, truth, baseline) for p in predictions]
    return {g: {'nmse_mean': float(np.mean([r[g]['nmse'] for r in rows])),
                'nmse_seed_sd': float(np.std([r[g]['nmse'] for r in rows], ddof=1)) if len(rows)>1 else 0.,
                'baseline_nmse': rows[0][g]['baseline_nmse']} for g in GROUP_NAMES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features-dir', 'probe-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    original = json.loads((args.probe_dir/'results.json').read_text())
    manifest = json.loads((args.features_dir/'manifest.json').read_text())
    if original['feature_manifest'] != manifest:
        raise ValueError('Probe/features provenance mismatch')
    datasets, preprocessing = {}, {}
    for variant in VARIANTS:
        datasets[variant] = {}
        for split in ('train', 'validation'):
            with np.load(args.features_dir/variant/f'{split}.npz', allow_pickle=False) as data:
                datasets[variant][split] = {k: data[k] for k in data.files}
            for key in ('features', 'targets_raw'):
                if not np.isfinite(datasets[variant][split][key]).all():
                    raise ValueError('Nonfinite input')
            for key in ('targets_raw', 'record_indices', 'frame_indices'):
                np.testing.assert_array_equal(datasets[variant][split][key], datasets['released'][split][key])
        if set(datasets[variant]['train']['record_indices']) & set(datasets[variant]['validation']['record_indices']):
            raise ValueError('Record leakage')
        with np.load(args.probe_dir/variant/'preprocessing.npz', allow_pickle=False) as data:
            preprocessing[variant] = {k: data[k] for k in data.files}
        if any(not np.isfinite(preprocessing[variant][k]).all() for k in ('feature_mean', 'pca_components', 'target_mean', 'target_scale')) or not (preprocessing[variant]['target_scale'] > 0).all():
            raise ValueError('Invalid preprocessing')
        for key in ('target_mean', 'target_scale'):
            np.testing.assert_array_equal(preprocessing[variant][key], preprocessing['released'][key])
    args.output_dir.mkdir(parents=True)
    report = {'scope': 'Exploratory frozen-feature train/validation diagnostic. No test access or LAM changes.',
              'protocol': {'ridge_alphas': ALPHAS, 'ridge_selection': 'minimum validation MSE; sum-SSE penalty, unpenalized intercept',
                           'mlp': 'input->64 ReLU->7; Adam .001; weight_decay .0001; batch512; maximum200epochs; patience20; best validation MSE',
                           'seeds': SEEDS, 'preprocessing': 'reuse original training-only feature mean/PCA; no whitening or feature rescaling',
                           'shuffled': 'full512 MLP; fixed training-label permutation seed20261013; validation labels correct'},
              'limitations': 'Validation is reused for tuning and reporting. Scores are exploratory and selection-biased; not an independent benchmark or exact paper reproduction. Seed SD covers probes only. MLP versus original linear also changes optimization budget; ridge references help separate linear fitting from capacity.',
              'feature_manifest': manifest, 'variants': {}}
    permutation = np.random.default_rng(20261013).permutation(len(datasets['released']['train']['features']))
    for variant in VARIANTS:
        dest = args.output_dir/variant; dest.mkdir()
        prep, data = preprocessing[variant], datasets[variant]
        y = {s: (data[s]['targets_raw'].astype(np.float64)-prep['target_mean'])/prep['target_scale'] for s in data}
        baseline = y['train'].mean(0)
        full = {s: data[s]['features'].astype(np.float64)-prep['feature_mean'] for s in data}
        pca = {s: full[s] @ prep['pca_components'].T for s in data}
        result = {'original_linear_30_epochs': original['variants'][variant]['splits']['validation']['summary']}
        saved_predictions = {f'{s}_targets_normalized': y[s] for s in data}
        for space, x in (('pca128', pca), ('full512', full)):
            trials = []
            for alpha in ALPHAS:
                weight, bias = ridge_fit(x['train'], y['train'], alpha)
                prediction = x['validation']@weight+bias
                trials.append((float(np.mean((prediction-y['validation'])**2)), alpha, weight, bias))
            _, alpha, weight, bias = min(trials, key=lambda row: row[0])
            predictions = {s: x[s]@weight+bias for s in data}
            result[f'ridge_{space}'] = {'alpha': alpha, 'selection_grid': [{'alpha': t[1], 'validation_mse': t[0]} for t in trials],
                                       **{s: summarize([predictions[s]], y[s], baseline) for s in data}}
            np.savez(dest/f'ridge_{space}.npz', weight=weight, bias=bias)
            for s in data:
                saved_predictions[f'{s}_ridge_{space}'] = predictions[s]
        for name, x, shuffled in (('mlp_pca128', pca, False), ('mlp_full512', full, False), ('shuffled_mlp_full512', full, True)):
            collected = {s: [] for s in data}; selections = []
            for seed in SEEDS:
                state, predictions, selection = train_mlp(x['train'], y['train'][permutation] if shuffled else y['train'],
                                                          x['validation'], y['validation'], seed)
                torch.save(state, dest/f'{name}_seed_{seed}.pt')
                selections.append({'seed': seed, **selection})
                for s in data:
                    collected[s].append(predictions[s])
                    saved_predictions[f'{s}_{name}_seed_{seed}'] = predictions[s]
                print(f"{variant}/{name}/seed{seed}: validation NMSE={group_scores(predictions['validation'], y['validation'], baseline)['all_7']['nmse']:.6f}; epoch={selection['selected_epoch']}", flush=True)
            result[name] = {'selection': selections, **{s: summarize(collected[s], y[s], baseline) for s in data}}
        np.savez_compressed(dest/'predictions.npz', **saved_predictions)
        np.savez(dest/'preprocessing.npz', **prep)
        report['variants'][variant] = result
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('\nVALIDATION: exploratory probe comparison (lower NMSE better)')
    for variant, result in report['variants'].items():
        print(variant, 'original_linear=', result['original_linear_30_epochs']['all_7']['nmse_mean'])
        for method in ('ridge_pca128', 'ridge_full512', 'mlp_pca128', 'mlp_full512', 'shuffled_mlp_full512'):
            print(f"  {method}: {result[method]['validation']['all_7']['nmse_mean']:.6f}")
    print(f'Saved {args.output_dir}; no LAM training or test access.')


if __name__ == '__main__':
    main()
