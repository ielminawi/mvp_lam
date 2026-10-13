"""Exploratory ridge mapping between camera latents; calibration-only model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def fit(entries, strength):
    # Equal total weight per episode, irrespective of its number of transitions.
    source = np.concatenate([e['source'] for e in entries]).astype(np.float64)
    target = np.concatenate([e['target'] for e in entries]).astype(np.float64)
    weights = np.concatenate([np.full(len(e['source']), 1/(len(entries)*len(e['source']))) for e in entries])
    x_mean = weights @ source
    y_mean = weights @ target
    x = (source-x_mean)*np.sqrt(weights[:,None])
    y = (target-y_mean)*np.sqrt(weights[:,None])
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    alpha = strength * singular[0]**2 if singular[0] > 1e-12 else strength
    mapping = (vt.T*(singular/(singular**2+alpha))) @ (u.T @ y)
    return mapping, x_mean, y_mean, alpha


def mse(entry, prediction):
    return float(np.mean((prediction-entry['target'])**2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output_dir}')
    report_path = args.diagnostics_dir/'results.json'
    reference = json.loads(report_path.read_text())
    cal_ids, test_ids = reference['calibration_record_indices'], reference['test_record_indices']
    if len(cal_ids) != 10 or len(test_ids) != 10 or set(cal_ids) & set(test_ids):
        raise ValueError('Unexpected reference split')
    grid = [1e-4, 1e-3, 1e-2, .1, 1., 10., 100.]
    report = {'scope': __doc__, 'status': 'Follow-up motivated by previously inspected test results; exploratory, not independent confirmation.',
        'reference_results_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'calibration_record_indices': cal_ids, 'test_record_indices': test_ids,
        'method': 'Affine ridge regression, unpenalized intercept; equal episode weights; no PCA or coordinate normalization.',
        'regularization': 'alpha = relative strength times largest squared singular value of episode-weighted centered calibration features.',
        'selection': 'Minimum mean leave-one-calibration-episode-out MSE over the predefined strength grid; test never selects strength.',
        'strength_grid': grid, 'results': {}}
    saved_models, saved_predictions = {}, {}
    for representation in ('quantized','continuous'):
        def load(ids):
            result=[]
            for index in ids:
                with np.load(args.diagnostics_dir/f'episode_{index:05d}.npz',allow_pickle=False) as data:
                    suffix = '' if representation=='quantized' else '_continuous'
                    e={'record_index':index,'source':data['source'+suffix].astype(np.float64),
                       'target':data['target'+suffix].astype(np.float64)}
                if e['source'].shape != e['target'].shape or e['source'].shape[1] != 512:
                    raise ValueError('Unexpected paired latent shape')
                if not all(np.isfinite(e[k]).all() for k in ('source','target')):
                    raise ValueError('Nonfinite latents')
                result.append(e)
            return result
        calibration = load(cal_ids)
        cv = []
        for strength in grid:
            folds=[]
            for held in range(len(calibration)):
                mapping,x_mean,y_mean,alpha=fit([e for i,e in enumerate(calibration) if i!=held],strength)
                e=calibration[held]
                folds.append(mse(e,(e['source']-x_mean) @ mapping+y_mean))
            cv.append({'strength':strength,'episode_mse':folds,'mean_mse':float(np.mean(folds))})
        selected=min(cv,key=lambda row:row['mean_mse'])
        mapping,x_mean,y_mean,alpha=fit(calibration,selected['strength'])
        offset=np.mean([(e['target']-e['source']).mean(0) for e in calibration],axis=0)
        target_mean=np.mean([e['target'].mean(0) for e in calibration],axis=0)
        saved_models.update({f'{representation}_mapping':mapping,f'{representation}_source_mean':x_mean,
            f'{representation}_target_mean':y_mean,f'{representation}_alpha':np.array(alpha)})
        test=load(test_ids)
        rows=[]
        for e in test:
            predictions={'no_correction':e['source'],'offset':e['source']+offset,
                'target_mean':np.broadcast_to(target_mean,e['target'].shape),
                'ridge':(e['source']-x_mean) @ mapping+y_mean}
            scores={key:mse(e,value) for key,value in predictions.items()}
            rows.append({'record_index':e['record_index'],'pairs':len(e['source']),'mse':scores})
            for name,value in predictions.items():
                saved_predictions[f"{representation}_episode_{e['record_index']:05d}_{name}"]=value
        summary={key:float(np.mean([row['mse'][key] for row in rows])) for key in rows[0]['mse']}
        old=reference['diagnostics'][representation]['paired']['episode_mean_mse']
        for key in old:
            np.testing.assert_allclose(summary[key],old[key],rtol=1e-8,atol=1e-10)
        report['results'][representation]={'cross_validation':cv,'selected_strength':selected['strength'],
            'selected_cv_mse':selected['mean_mse'],'fitted_alpha':alpha,'test_episode_mean_mse':summary,'test_episodes':rows}
        print(f'\n{representation.upper()}')
        print(f"Selected relative strength: {selected['strength']}; calibration CV MSE: {selected['mean_mse']:.8f}")
        for name,value in summary.items():
            print(f'  test {name}: MSE={value:.8f}')
        for baseline in ('no_correction','offset','target_mean'):
            improvement=100*(1-summary['ridge']/summary[baseline]) if summary[baseline]>1e-12 else None
            print(f'  ridge error reduction vs {baseline} (%): {improvement}')
    args.output_dir.mkdir(parents=True)
    np.savez(args.output_dir/'models.npz',**saved_models)
    np.savez(args.output_dir/'predictions.npz',**saved_predictions)
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2))
    print('\nSaved:',args.output_dir/'results.json')
    print('No LAM training or new GPU extraction. Test scores are exploratory.')


if __name__=='__main__':
    main()
