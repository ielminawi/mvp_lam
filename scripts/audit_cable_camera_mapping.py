"""Per-episode ridge audit and ten shuffled-calibration refits; exploratory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def decomposition(entries):
    x=np.concatenate([e['source'] for e in entries])
    y=np.concatenate([e['target'] for e in entries])
    weights=np.concatenate([np.full(len(e['source']),1/(len(entries)*len(e['source']))) for e in entries])
    mx,my=weights@x,weights@y
    u,s,vt=np.linalg.svd((x-mx)*np.sqrt(weights[:,None]),full_matrices=False)
    return mx,my,s,vt,u.T@((y-my)*np.sqrt(weights[:,None]))


def predict(parts, strength, source):
    mx,my,s,vt,uy=parts
    alpha=strength*s[0]**2 if s[0]>1e-12 else strength
    # Equivalent ridge prediction without constructing a full 512x512 matrix.
    return (((source-mx)@vt.T)*(s/(s*s+alpha)))@uy+my


def fit_select(calibration, test, grid):
    scores=np.zeros(len(grid))
    for held,e in enumerate(calibration):
        parts=decomposition([v for i,v in enumerate(calibration) if i!=held])
        for j,strength in enumerate(grid):
            scores[j]+=np.mean((predict(parts,strength,e['source'])-e['target'])**2)/len(calibration)
    chosen=int(np.argmin(scores))
    parts=decomposition(calibration)
    predictions=[predict(parts,grid[chosen],e['source']) for e in test]
    errors=[float(np.mean((p-e['target'])**2)) for p,e in zip(predictions,test)]
    return {'selected_strength':grid[chosen],'cv_mse_by_strength':scores.tolist(),
            'test_episode_mse':errors,'test_mean_mse':float(np.mean(errors))},predictions


def shuffle_calibration(entries, seed):
    rng=np.random.default_rng(seed)
    shuffled=[]
    for e in entries:
        n=len(e['source'])
        if n<2:
            raise ValueError('Need two pairs for within-episode mismatch')
        while True:
            order=rng.permutation(n)
            if np.all(order!=np.arange(n)):
                break
        shuffled.append({**e,'target':e['target'][order]})
    return shuffled


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostics-dir','mapping-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Choose new directory: {args.output_dir}')
    mapping_path=args.mapping_dir/'results.json'
    mapping=json.loads(mapping_path.read_text())
    reference_path=args.diagnostics_dir/'results.json'
    if hashlib.sha256(reference_path.read_bytes()).hexdigest()!=mapping['reference_results_sha256']:
        raise ValueError('Wrong cached feature experiment')
    grid=mapping['strength_grid']
    report={'scope':__doc__,'mapping_results_sha256':hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        'control':'Ten fixed seeds 20261013..20261022. Shuffle targets within each calibration episode with no fixed indices; rerun the same leave-one-episode-out selection. Test pairings remain correct.',
        'limitations':'Within-episode shuffling preserves scene and episode-level associations; temporally overlapping windows remain correlated. This control tests finer temporal pairing, not all dependence. Repeats are not independent dataset samples; no significance claim. Existing test has already guided development.',
        'results':{}}
    predictions={}
    args.output_dir.mkdir(parents=True)
    for representation in ('quantized','continuous'):
        def load(ids):
            result=[]
            suffix='' if representation=='quantized' else '_continuous'
            for index in ids:
                with np.load(args.diagnostics_dir/f'episode_{index:05d}.npz',allow_pickle=False) as saved:
                    result.append({'record_index':index,'source':saved['source'+suffix].astype(np.float64),
                                   'target':saved['target'+suffix].astype(np.float64)})
            return result
        cal=load(mapping['calibration_record_indices']);test=load(mapping['test_record_indices'])
        real,real_pred=fit_select(cal,test,grid)
        old=mapping['results'][representation]
        np.testing.assert_allclose(real['test_mean_mse'],old['test_episode_mean_mse']['ridge'],atol=1e-9)
        if real['selected_strength']!=old['selected_strength']:
            raise ValueError('Selected strength did not reproduce')
        print(f'\n{representation.upper()}: correctly paired mapping',flush=True)
        wins=0;rows=[]
        for e,score,row,pred in zip(test,real['test_episode_mse'],old['test_episodes'],real_pred):
            if e['record_index']!=row['record_index']:
                raise ValueError('Episode order mismatch')
            base=row['mse']['target_mean']
            reduction=100*(1-score/base) if base>1e-12 else None
            wins+=score<base
            rows.append({'record_index':e['record_index'],'ridge_mse':score,'target_mean_mse':base,
                         'error_reduction_vs_mean_percent':reduction})
            predictions[f'{representation}_real_{e["record_index"]}']=pred
            print(f"  record {e['record_index']}: ridge={score:.6f}; mean={base:.6f}; reduction={reduction:.2f}%",flush=True)
        print(f'  Beats mean baseline: {wins}/{len(test)} episodes',flush=True)
        controls=[]
        for seed in range(20261013,20261023):
            control,preds=fit_select(shuffle_calibration(cal,seed),test,grid)
            control['seed']=seed
            controls.append(control)
            for e,pred in zip(test,preds):
                predictions[f'{representation}_shuffle_{seed}_{e["record_index"]}']=pred
            print(f"  shuffled seed {seed}: test MSE={control['test_mean_mse']:.8f}; strength={control['selected_strength']}",flush=True)
        values=[c['test_mean_mse'] for c in controls]
        mean=float(np.mean(values))
        report['results'][representation]={'paired':real,'per_episode':rows,'episodes_beating_mean':wins,
            'shuffled_controls':controls,'shuffled_test_mse_mean':mean,'shuffled_test_mse_std':float(np.std(values,ddof=1)),
            'paired_error_reduction_vs_shuffled_mean_percent':100*(1-real['test_mean_mse']/mean)}
        print(f"  Paired MSE={real['test_mean_mse']:.8f}; shuffled mean={mean:.8f} +/- {np.std(values,ddof=1):.8f}",flush=True)
        print(f"  Paired reduction vs shuffled mean: {100*(1-real['test_mean_mse']/mean):.2f}%",flush=True)
    np.savez(args.output_dir/'predictions.npz',**predictions)
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2))
    print('\nSaved:',args.output_dir/'results.json')
    print('No LAM training, GPU extraction or test-based regularization selection.')


if __name__=='__main__':
    main()
