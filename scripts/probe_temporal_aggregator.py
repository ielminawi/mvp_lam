"""Continuous/quantized endpoint probes for frozen aggregators and zero-gate controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from latent_action_model.genie.modules.temporal_aggregator import AggregatedLAM
from scripts.temporal_pilot import RELEASE_HASH, Windows, frozen_hash, load_model, provenance, sha256
from scripts.compare_temporal_continuous import identities_match
from scripts.diagnose_temporal_readout import ridge_grid
from scripts.check_probe_bottleneck import summarize
from scripts.diagnose_temporal_intermediates import paired_interval

VARIANTS = ('two_zero', 'two_aggregated', 'four_zero', 'four_aggregated')


def readouts(model, patches):
    batch, times = patches.shape[:2]
    slots = model.backbone.action_latent.expand(batch, times, -1, -1)
    contextualized = model.backbone.encoder(torch.cat((slots, patches), dim=2))[:, :, :4]
    summary = model.aggregator(contextualized)
    correction = summary-contextualized[:, -1]
    relative = correction.flatten(1).norm(dim=1)/contextualized[:, -1].flatten(1).norm(dim=1).clamp_min(1e-12)
    outputs = {}
    for name, value in (('zero', contextualized[:, -1]), ('aggregated', summary)):
        continuous = model.backbone.to_codebook(value)
        quantized, _, _, ids = model.backbone.vq(continuous)
        outputs[name] = {'continuous':continuous.flatten(1), 'quantized':quantized.flatten(1), 'indices':ids}
    return outputs, relative


def load_aggregator_state(model, saved, report, variant, cache_hash):
    if saved['update'] != report['config']['updates'] or saved['update'] != report['history'][-1]['update']:
        raise ValueError('Not final completed aggregator checkpoint')
    for key in ('variant','config','cache_manifest_sha256','frozen_weight_sha256','base_checkpoint_sha256'):
        if saved['run'][key] != report[key]:
            raise ValueError(f'Checkpoint/report mismatch: {key}')
    if report['variant'] != variant or report['cache_manifest_sha256'] != cache_hash or report['base_checkpoint_sha256'] != RELEASE_HASH or not report['frozen_weights_unchanged']:
        raise ValueError('Wrong completed aggregator run')
    if frozen_hash(model) != report['frozen_weight_sha256']:
        raise ValueError('Frozen released backbone mismatch')
    if any(not torch.isfinite(v).all() for v in saved['aggregator'].values()):
        raise ValueError('Nonfinite aggregator weights')
    model.aggregator.load_state_dict(saved['aggregator'], strict=True)


def extract(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available() or args.batch_size < 1:
        raise ValueError('GPU allocation and positive batch size required')
    cache = json.loads((args.cache_dir/'manifest.json').read_text())
    original = json.loads((args.features_dir/'manifest.json').read_text())
    cache_hash = sha256(args.cache_dir/'manifest.json')
    if cache['status'] != 'complete' or cache['offsets'] != [0,3,6,9] or cache['limit_episodes_per_split'] or original['cache_manifest_sha256'] != cache_hash:
        raise ValueError('Need matching full cache')
    if any(e['split'] not in ('train','validation') for e in cache['episodes']):
        raise ValueError('Unexpected split')
    sources={s:{e['source_path'] for e in cache['episodes'] if e['split']==s} for s in ('train','validation')}
    if sources['train'] & sources['validation']:
        raise ValueError('Source leakage')
    for entry in cache['episodes']:
        for key,digest in (('patch_file','patch_sha256'),('window_file','window_sha256')):
            if sha256(args.cache_dir/entry[key]) != entry[digest]:
                raise ValueError('Changed cache input')
    directories={'two_frame':args.two_frame_dir,'four_frame':args.four_frame_dir}
    reports={v:json.loads((d/'results.json').read_text()) for v,d in directories.items()}
    if reports['two_frame']['config'] != reports['four_frame']['config']:
        raise ValueError('Unmatched training configuration')
    args.output_dir.mkdir(parents=True)
    manifest={'scope':'Frozen aggregator/zero-gate readout features, train/validation only; no LAM fitting/test access',
              'source_feature_manifest':original,'cache_manifest_sha256':cache_hash,'provenance':provenance(args),
              'runs':reports,'checkpoint_sha256':{},'files':{},'correction':{}}
    for variant in directories:
        prefix='two' if variant=='two_frame' else 'four'
        checkpoint=directories[variant]/'latest.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=AggregatedLAM(load_model(args)).cuda()
        load_aggregator_state(model,saved,reports[variant],variant,cache_hash)
        del saved
        model.requires_grad_(False).eval(); before=frozen_hash(model)
        manifest['checkpoint_sha256'][variant]=sha256(checkpoint)
        for name in ('zero','aggregated'):
            (args.output_dir/f'{prefix}_{name}').mkdir()
        for split in ('train','validation'):
            dataset=Windows(args.cache_dir,cache,split)
            with np.load(args.features_dir/'released'/f'{split}.npz',allow_pickle=False) as data:
                reference={k:data[k] for k in data.files}
            identity={k:[] for k in ('targets_raw','record_indices','frame_indices')}
            for entry,_ in dataset.entries:
                with np.load(args.cache_dir/entry['window_file'],allow_pickle=False) as data:
                    identity['targets_raw'].append(data['targets_raw']); identity['frame_indices'].append(data['original_frame_indices'])
                    identity['record_indices'].append(np.full(len(data['windows']),entry['record_index']))
            identity={k:np.concatenate(v) for k,v in identity.items()}; identities_match(identity,reference)
            collected={name:{k:[] for k in ('continuous','quantized','indices')} for name in ('zero','aggregated')}; relative=[]
            with torch.no_grad():
                for first in range(0,len(dataset),args.batch_size):
                    last=min(first+args.batch_size,len(dataset))
                    outputs,change=readouts(model,dataset.batch(range(first,last),variant))
                    for name,fields in outputs.items():
                        for key,value in fields.items():
                            if not torch.isfinite(value).all():
                                raise ValueError('Nonfinite readout')
                            collected[name][key].append(value.cpu().numpy())
                    if variant=='two_frame':
                        np.testing.assert_array_equal(outputs['zero']['indices'].cpu().numpy(),reference['indices'][first:last])
                        np.testing.assert_allclose(outputs['zero']['quantized'].cpu().numpy(),reference['features'][first:last],rtol=1e-5,atol=1e-5)
                    relative.append(change.cpu().numpy())
                    if first % (args.batch_size*100)==0:
                        print(f'{variant}/{split}: {last}/{len(dataset)}',flush=True)
            for name,fields in collected.items():
                values={k:np.concatenate(v) for k,v in fields.items()}
                if split=='validation':
                    counts=np.bincount(values['indices'].ravel(),minlength=16).tolist()
                    expected=reports[variant]['history'][-1 if name=='aggregated' else 0]['validation']['code_counts']
                    if counts != expected:
                        raise ValueError('Training baseline/final code counts not reproduced')
                path=args.output_dir/f'{prefix}_{name}'/f'{split}.npz'
                np.savez_compressed(path,**identity,**values)
                manifest['files'][f'{prefix}_{name}/{split}.npz']=sha256(path)
            manifest['correction'][f'{prefix}/{split}']={'relative_summary_l2_mean':float(np.concatenate(relative).mean()),
                                                       'tanh_gate':float(torch.tanh(model.aggregator.gate))}
        if frozen_hash(model)!=before:
            raise ValueError('Frozen weights changed')
        del model; torch.cuda.empty_cache()
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print('Saved aggregator features and correction diagnostics:',args.output_dir)


def fit(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest=json.loads((args.features_dir/'manifest.json').read_text())
    original=json.loads((args.probe_dir/'results.json').read_text())
    if original['feature_manifest']!=manifest['source_feature_manifest']:
        raise ValueError('Normalization/probe provenance mismatch')
    with np.load(args.probe_dir/'four_frame/preprocessing.npz',allow_pickle=False) as data:
        mean,scale=data['target_mean'],data['target_scale']
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale>0).all():
        raise ValueError('Invalid target normalization')
    datasets={}
    for variant in VARIANTS:
        datasets[variant]={}
        for split in ('train','validation'):
            path=args.features_dir/variant/f'{split}.npz'
            if sha256(path)!=manifest['files'][f'{variant}/{split}.npz']:
                raise ValueError('Changed extracted features')
            with np.load(path,allow_pickle=False) as data:
                datasets[variant][split]={k:data[k] for k in data.files}
            identities_match(datasets[variant][split],datasets['two_zero'][split])
        if set(datasets[variant]['train']['record_indices']) & set(datasets[variant]['validation']['record_indices']):
            raise ValueError('Record leakage')
    y={s:(datasets['two_zero'][s]['targets_raw'].astype(np.float64)-mean)/scale for s in ('train','validation')}
    baseline=y['train'].mean(0); denominator=np.var(y['validation'],axis=0).sum()
    if denominator<=0:
        raise ValueError('Zero action variance')
    args.output_dir.mkdir(parents=True); results={}; errors={}
    for variant in VARIANTS:
        dest=args.output_dir/variant; dest.mkdir(); results[variant]={}; errors[variant]={}; predictions={}
        for kind in ('quantized','continuous'):
            x={s:datasets[variant][s][kind].astype(np.float64) for s in y}
            if any(v.shape!=(len(y[s]),512) or not np.isfinite(v).all() for s,v in x.items()):
                raise ValueError('Invalid features')
            best,grid=ridge_grid(x['train'],y['train'],x['validation'],y['validation'])
            _,alpha,weight,bias=best; result={'alpha':alpha,'grid':grid}
            for split in y:
                prediction=x[split]@weight+bias; predictions[f'{split}_{kind}']=prediction
                result[split]=summarize([prediction],y[split],baseline)
            errors[variant][kind]=((predictions[f'validation_{kind}']-y['validation'])**2).sum(1)/denominator
            results[variant][kind]=result
            np.savez(dest/f'{kind}_ridge.npz',weight=weight,bias=bias,target_mean=mean,target_scale=scale)
            print(variant,kind,{g:v['nmse_mean'] for g,v in result['validation'].items()},flush=True)
        np.savez_compressed(dest/'predictions.npz',**predictions,validation_targets_normalized=y['validation'],
                            validation_record_indices=datasets[variant]['validation']['record_indices'])
    pairs={'four_aggregated_minus_two_aggregated':('four_aggregated','two_aggregated'),
           'two_aggregated_minus_two_zero':('two_aggregated','two_zero'),
           'four_aggregated_minus_four_zero':('four_aggregated','four_zero')}
    paired={label:{kind:paired_interval(errors[a][kind]-errors[b][kind],datasets['two_zero']['validation']['record_indices'])
                   for kind in ('quantized','continuous')} for label,(a,b) in pairs.items()}
    report={'variants':results,'paired':paired,'extraction_manifest':manifest,
            'protocol':'Full512 ridge, no PCA/whitening; alpha .01/.1/1/10/100/1000 selected on validation MSE; sum-SSE penalty, unpenalized intercept. Shared endpoint7 normalization.',
            'limitations':'Exploratory validation selection/report reuse; no independent test or LAM-seed uncertainty. Four/Two zero use the same frozen released encoder, different input lengths. Negative trained-minus-zero error supports improvement; paired CIs episode-weighted fixed denominator, not selection-adjusted. A continuous gain is not a deployable discrete gain.'}
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Paired comparisons:',paired)
    print('Saved',args.output_dir,'; no LAM fitting or test access.')


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    extraction=sub.add_parser('extract')
    for name in ('cache-dir','features-dir','two-frame-dir','four-frame-dir','checkpoint','dino-dir','output-dir'):
        extraction.add_argument('--'+name,type=Path,required=True)
    extraction.add_argument('--batch-size',type=int,default=4)
    fitting=sub.add_parser('fit')
    for name in ('features-dir','probe-dir','output-dir'):
        fitting.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(1)
    (extract if args.command=='extract' else fit)(args)


if __name__=='__main__':
    main()
