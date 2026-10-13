"""Matched final-slot continuous/quantized ridge probes; frozen LAMs, no test access."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from scripts.temporal_pilot import Windows, frozen_hash, load_model, provenance, sha256
from scripts.probe_temporal_pilot import VARIANTS, overlay_adaptation
from scripts.diagnose_temporal_readout import ridge_grid
from scripts.check_probe_bottleneck import summarize
from scripts.diagnose_temporal_intermediates import paired_interval


def identities_match(data, reference):
    for key in ('record_indices', 'frame_indices', 'targets_raw'):
        np.testing.assert_array_equal(data[key], reference[key], err_msg=f'Misaligned {key}')


def extract(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available() or args.batch_size < 1:
        raise ValueError('GPU allocation and positive batch size required')
    cache = json.loads((args.cache_dir/'manifest.json').read_text())
    features = json.loads((args.features_dir/'manifest.json').read_text())
    readout = json.loads((args.four_readout_dir/'manifest.json').read_text())
    if readout['feature_manifest'] != features:
        raise ValueError('Four-frame readout provenance mismatch')
    cache_hash = sha256(args.cache_dir/'manifest.json')
    if features['cache_manifest_sha256'] != cache_hash or readout['cache_manifest_sha256'] != cache_hash:
        raise ValueError('Cache provenance mismatch')
    if cache['status'] != 'complete' or cache['offsets'] != [0, 3, 6, 9] or cache['limit_episodes_per_split']:
        raise ValueError('Need full temporal cache')
    if any(e['split'] not in ('train', 'validation') for e in cache['episodes']):
        raise ValueError('Unexpected cache split')
    sources = {s: {e['source_path'] for e in cache['episodes'] if e['split']==s} for s in ('train', 'validation')}
    if sources['train'] & sources['validation']:
        raise ValueError('Source leakage')
    for entry in cache['episodes']:
        for file_key, hash_key in (('patch_file', 'patch_sha256'), ('window_file', 'window_sha256')):
            if sha256(args.cache_dir/entry[file_key]) != entry[hash_key]:
                raise ValueError('Changed cache file')
    two_report = json.loads((args.two_frame_dir/'results.json').read_text())
    four_report = features['variants']['four_frame']['training_report']
    if two_report['config'] != four_report['config'] or two_report['cache_manifest_sha256'] != cache_hash or not two_report['frozen_weights_unchanged']:
        raise ValueError('Unmatched adaptation runs')
    adapted = args.two_frame_dir/'latest.pt'
    if sha256(adapted) != features['variants']['two_frame']['adapted_checkpoint_sha256']:
        raise ValueError('Two-frame checkpoint mismatch')
    if readout['adapted_checkpoint_sha256'] != features['variants']['four_frame']['adapted_checkpoint_sha256']:
        raise ValueError('Four-frame checkpoint mismatch')
    args.output_dir.mkdir(parents=True)
    report = {'feature_manifest': features, 'four_readout_manifest_sha256': sha256(args.four_readout_dir/'manifest.json'),
              'provenance': provenance(args), 'scope': 'Final four slots only, continuous512 versus quantized512; train/validation, no LAM fitting or test access.', 'files': {}}
    for variant in VARIANTS:
        destination = args.output_dir/variant; destination.mkdir()
        model = None
        if variant != 'four_frame':
            model = load_model(args).configure_pilot()
            if variant == 'two_frame':
                saved = torch.load(adapted, map_location='cpu', weights_only=True)
                if saved['config'] != two_report['config']:
                    raise ValueError('Checkpoint/report configuration mismatch')
                overlay_adaptation(model, saved, variant)
                if frozen_hash(model) != two_report['frozen_weight_sha256']:
                    raise ValueError('Frozen base mismatch')
                del saved
            model.requires_grad_(False).eval()
            before = frozen_hash(model)
        for split in ('train', 'validation'):
            with np.load(args.features_dir/variant/f'{split}.npz', allow_pickle=False) as saved:
                reference = {k: saved[k] for k in saved.files}
            if variant == 'four_frame':
                with np.load(args.four_readout_dir/f'{split}.npz', allow_pickle=False) as saved:
                    values = {k: saved[k] for k in ('D_continuous', 'D_quantized', 'record_indices', 'frame_indices', 'targets_raw')}
                identities_match(values, reference)
                np.testing.assert_allclose(values['D_quantized'], reference['features'], rtol=1e-5, atol=1e-5)
                continuous, quantized = values['D_continuous'], values['D_quantized']
            else:
                dataset = Windows(args.cache_dir, cache, split)
                identity = {k: [] for k in ('record_indices', 'frame_indices', 'targets_raw')}
                for entry, _ in dataset.entries:
                    with np.load(args.cache_dir/entry['window_file'], allow_pickle=False) as saved:
                        identity['record_indices'].append(np.full(len(saved['windows']), entry['record_index']))
                        identity['frame_indices'].append(saved['original_frame_indices']); identity['targets_raw'].append(saved['targets_raw'])
                identities_match({k:np.concatenate(v) for k,v in identity.items()}, reference)
                continuous, quantized = [], []
                with torch.no_grad():
                    for first in range(0, len(dataset), args.batch_size):
                        last = min(first+args.batch_size, len(dataset))
                        encoded = model.encode_window(dataset.batch(range(first, last), 'two_frame'))
                        np.testing.assert_array_equal(encoded['indices'].cpu().numpy(), reference['indices'][first:last])
                        continuous.append(encoded['emb'].flatten(1).cpu().numpy())
                        quantized.append(encoded['z_q'].flatten(1).cpu().numpy())
                        if first % (args.batch_size*100) == 0:
                            print(f'{variant}/{split}: {last}/{len(dataset)}', flush=True)
                continuous, quantized = np.concatenate(continuous), np.concatenate(quantized)
                np.testing.assert_allclose(quantized, reference['features'], rtol=1e-5, atol=1e-5)
            for matrix in (continuous, quantized):
                if matrix.shape != (len(reference['targets_raw']), 512) or not np.isfinite(matrix).all():
                    raise ValueError('Invalid final-slot features')
            file = destination/f'{split}.npz'
            np.savez_compressed(file, continuous=continuous, quantized=quantized,
                                **{k:reference[k] for k in ('record_indices', 'frame_indices', 'targets_raw')})
            report['files'][f'{variant}/{split}.npz'] = sha256(file)
        if model is not None:
            if frozen_hash(model) != before:
                raise ValueError('Model parameters changed')
            del model
            gc.collect(); torch.cuda.empty_cache()
    (args.output_dir/'manifest.json').write_text(json.dumps(report, indent=2))
    print('Saved matched features:', args.output_dir, '; no LAM changes or test access.')


def fit(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads((args.continuous_dir/'manifest.json').read_text())
    original = json.loads((args.probe_dir/'results.json').read_text())
    if original['feature_manifest'] != manifest['feature_manifest']:
        raise ValueError('Probe provenance mismatch')
    with np.load(args.probe_dir/'four_frame/preprocessing.npz', allow_pickle=False) as saved:
        mean, scale = saved['target_mean'], saved['target_scale']
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Invalid target normalization')
    datasets = {}
    for variant in VARIANTS:
        datasets[variant] = {}
        for split in ('train', 'validation'):
            path = args.continuous_dir/variant/f'{split}.npz'
            if sha256(path) != manifest['files'][f'{variant}/{split}.npz']:
                raise ValueError('Changed extracted file')
            with np.load(path, allow_pickle=False) as saved:
                datasets[variant][split] = {k:saved[k] for k in saved.files}
            identities_match(datasets[variant][split], datasets['released'][split])
        if set(datasets[variant]['train']['record_indices']) & set(datasets[variant]['validation']['record_indices']):
            raise ValueError('Record leakage')
    y = {s:(datasets['released'][s]['targets_raw'].astype(np.float64)-mean)/scale for s in ('train', 'validation')}
    baseline = y['train'].mean(0)
    args.output_dir.mkdir(parents=True)
    results, errors = {}, {}
    for variant in VARIANTS:
        destination = args.output_dir/variant; destination.mkdir()
        results[variant], errors[variant] = {}, {}
        predictions = {f'{s}_targets_normalized':y[s] for s in y}
        for kind in ('quantized', 'continuous'):
            x = {s:datasets[variant][s][kind].astype(np.float64) for s in y}
            if any(v.shape != (len(y[s]), 512) or not np.isfinite(v).all() for s,v in x.items()):
                raise ValueError('Invalid features')
            best, grid = ridge_grid(x['train'], y['train'], x['validation'], y['validation'])
            _, alpha, weight, bias = best
            result = {'alpha': alpha, 'grid': grid}
            for split in y:
                prediction = x[split]@weight+bias
                predictions[f'{split}_{kind}'] = prediction
                result[split] = summarize([prediction], y[split], baseline)
            errors[variant][kind] = ((predictions[f'validation_{kind}']-y['validation'])**2).sum(1)/np.var(y['validation'],axis=0).sum()
            np.savez(destination/f'{kind}_ridge.npz', weight=weight, bias=bias, target_mean=mean, target_scale=scale)
            results[variant][kind] = result
            print(variant, kind, {g:v['nmse_mean'] for g,v in result['validation'].items()}, flush=True)
        np.savez_compressed(destination/'predictions.npz', **predictions,
                            validation_record_indices=datasets[variant]['validation']['record_indices'])
    records = datasets['released']['validation']['record_indices']
    paired = {kind: paired_interval(errors['four_frame'][kind]-errors['two_frame'][kind], records) for kind in ('continuous', 'quantized')}
    report = {'variants': results, 'paired_four_minus_two': paired, 'extraction_manifest': manifest,
              'protocol': 'Same final512D slots, full features no PCA; ridge alpha .01/.1/1/10/100/1000 selected on validation MSE; sum-SSE penalty, unpenalized intercept. Shared endpoint7 targets and training-derived normalization.',
              'limitations': 'Exploratory validation-tuned, not independent test. Equal feature dimension does not mean equal information capacity: continuous is not four discrete IDs. One LAM adaptation seed. Paired intervals equal-episode weighted, fixed scored-split variance, not selection-adjusted or LAM uncertainty. Positive paired difference means four-frame worse.'}
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Paired four-minus-two:', paired)
    print('Saved', args.output_dir, '; no LAM changes or test access.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    extraction = sub.add_parser('extract')
    for name in ('cache-dir', 'features-dir', 'four-readout-dir', 'two-frame-dir', 'checkpoint', 'dino-dir', 'output-dir'):
        extraction.add_argument('--'+name, type=Path, required=True)
    extraction.add_argument('--batch-size', type=int, default=4)
    fitting = sub.add_parser('fit')
    for name in ('continuous-dir', 'probe-dir', 'output-dir'):
        fitting.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    (extract if args.command == 'extract' else fit)(args)


if __name__ == '__main__':
    main()
