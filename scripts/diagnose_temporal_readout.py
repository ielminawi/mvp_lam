"""Frozen four-frame readout diagnostics: all slots and per-frame code decoding."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.temporal_pilot import Windows, frozen_hash, load_model, provenance, sha256
from scripts.probe_temporal_pilot import overlay_adaptation
from scripts.check_probe_bottleneck import ALPHAS, summarize
from scripts.diagnose_temporal_intermediates import paired_interval


def encode_all(model, patches):
    batch, times = patches.shape[:2]
    slots = model.action_latent.expand(batch, times, -1, -1)
    encoded = model.encoder(torch.cat((slots, patches), dim=2))
    continuous = model.to_codebook(encoded[:, :, :model.num_codes])
    quantized, _, _, indices = model.vq(continuous.reshape(batch*times, model.num_codes, -1))
    return continuous, quantized.reshape(batch, times, model.num_codes, -1), indices.reshape(batch, times, model.num_codes)


def decode_codes(model, initial, codes):
    tokens = torch.cat((model.action_up(codes), model.patch_up(initial)), dim=2)
    return model.decoder(tokens)[:, :, -initial.shape[2]:]


def ridge_grid(x, y, validation, truth):
    """One spectral decomposition for the full-feature ridge grid; no PCA truncation."""
    center, mean = x.mean(0), y.mean(0)
    xc = x-center
    eigenvalues, vectors = np.linalg.eigh(xc.T@xc)
    eigenvalues = np.maximum(eigenvalues, 0)
    rhs = vectors.T @ (xc.T@(y-mean))
    trials = []
    for alpha in ALPHAS:
        weight = vectors@(rhs/(eigenvalues[:, None]+alpha))
        bias = mean-center@weight
        prediction = validation@weight+bias
        trials.append((float(np.mean((prediction-truth)**2)), alpha, weight, bias))
    best = min(trials, key=lambda row: row[0])
    return best, [{'alpha': r[1], 'validation_mse': r[0]} for r in trials]


def extract(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise ValueError('Positive batch size and GPU allocation required')
    manifest = json.loads((args.cache_dir/'manifest.json').read_text())
    feature_manifest = json.loads((args.features_dir/'manifest.json').read_text())
    run = json.loads((args.four_frame_dir/'results.json').read_text())
    if manifest['status'] != 'complete' or manifest['offsets'] != [0, 3, 6, 9] or manifest['limit_episodes_per_split']:
        raise ValueError('Need full completed four-frame cache')
    if any(e['split'] not in ('train', 'validation') for e in manifest['episodes']):
        raise ValueError('Unexpected split')
    sources = {s: {e['source_path'] for e in manifest['episodes'] if e['split']==s} for s in ('train', 'validation')}
    if sources['train'] & sources['validation']:
        raise ValueError('Source leakage')
    cache_hash = sha256(args.cache_dir/'manifest.json')
    if run['cache_manifest_sha256'] != cache_hash or feature_manifest['cache_manifest_sha256'] != cache_hash:
        raise ValueError('Cache provenance mismatch')
    adapted = args.four_frame_dir/'latest.pt'
    if feature_manifest['variants']['four_frame']['adapted_checkpoint_sha256'] != sha256(adapted):
        raise ValueError('Feature checkpoint mismatch')
    if not run['frozen_weights_unchanged'] or run['variant'] != 'four_frame':
        raise ValueError('Invalid completed run')
    for entry in manifest['episodes']:
        for key, digest in (('patch_file', 'patch_sha256'), ('window_file', 'window_sha256')):
            if sha256(args.cache_dir/entry[key]) != entry[digest]:
                raise ValueError('Changed cache file')
    model = load_model(args).configure_pilot()
    saved = torch.load(adapted, map_location='cpu', weights_only=True)
    if saved['config'] != run['config']:
        raise ValueError('Checkpoint configuration mismatch')
    overlay_adaptation(model, saved, 'four_frame')
    if frozen_hash(model) != run['frozen_weight_sha256']:
        raise ValueError('Frozen base mismatch')
    model.requires_grad_(False).eval()
    before = frozen_hash(model)
    args.output_dir.mkdir(parents=True)
    validation_errors = []
    for split in ('train', 'validation'):
        dataset = Windows(args.cache_dir, manifest, split)
        with np.load(args.features_dir/'four_frame'/f'{split}.npz', allow_pickle=False) as data:
            reference = {k: data[k] for k in data.files}
        records, frames, labels = [], [], []
        for entry, _ in dataset.entries:
            with np.load(args.cache_dir/entry['window_file'], allow_pickle=False) as data:
                records.append(np.full(len(data['windows']), entry['record_index']))
                frames.append(data['original_frame_indices']); labels.append(data['targets_raw'])
        identities = dict(record_indices=np.concatenate(records), frame_indices=np.concatenate(frames), targets_raw=np.concatenate(labels))
        for key in identities:
            np.testing.assert_array_equal(identities[key], reference[key])
        features = {name: [] for name in ('D_quantized', 'D_continuous', 'all_continuous')}
        with torch.no_grad():
            for first in range(0, len(dataset), args.batch_size):
                last = min(first+args.batch_size, len(dataset))
                patches = dataset.batch(range(first, last), 'four_frame')
                continuous, quantized, indices = encode_all(model, patches)
                # Exact code IDs and close vectors validate the existing readout baseline.
                np.testing.assert_array_equal(indices[:, -1].cpu().numpy(), reference['indices'][first:last])
                np.testing.assert_allclose(quantized[:, -1].flatten(1).cpu().numpy(), reference['features'][first:last], rtol=1e-5, atol=1e-5)
                values = {'D_quantized': quantized[:, -1].flatten(1), 'D_continuous': continuous[:, -1].flatten(1),
                          'all_continuous': continuous.flatten(1)}
                for name,value in values.items():
                    if not torch.isfinite(value).all():
                        raise ValueError('Nonfinite representation')
                    features[name].append(value.cpu().numpy())
                if split == 'validation':
                    errors = []
                    for t in range(4):
                        reconstruction = decode_codes(model, patches[:, :1], quantized[:, t:t+1])
                        errors.append((reconstruction-patches[:, -1:]).square().flatten(1).mean(1).cpu().numpy())
                    validation_errors.append(np.stack(errors, axis=1))
                if first % (args.batch_size*100) == 0:
                    print(f'{split}: {last}/{len(dataset)}', flush=True)
        np.savez_compressed(args.output_dir/f'{split}.npz', **identities, **{k: np.concatenate(v) for k,v in features.items()})
    if frozen_hash(model) != before:
        raise ValueError('Frozen parameters changed')
    errors = np.concatenate(validation_errors)
    np.savez(args.output_dir/'decoder_errors.npz', errors=errors, record_indices=identities['record_indices'], frame_indices=identities['frame_indices'])
    decoder = {name: {'reconstruction_mse': float(errors[:, t].mean()),
                       'paired_minus_D': paired_interval(errors[:, t]-errors[:, 3], identities['record_indices'])}
               for t,name in enumerate(('A_codes', 'B_codes', 'C_codes', 'D_codes'))}
    np.testing.assert_allclose(errors[:, 3].mean(), run['history'][-1]['validation']['reconstruction_mse'], rtol=1e-4, atol=1e-6)
    report = {'decoder': decoder, 'provenance': provenance(args), 'cache_manifest_sha256': cache_hash,
              'adapted_checkpoint_sha256': sha256(adapted), 'feature_manifest': feature_manifest,
              'scope': 'Frozen four-frame model; train/validation representations, decoder test validation only. No test data or fitting.',
              'decoder_interpretation': 'Every condition decodes D from A with four codes from one encoder frame. Earlier causal slots see less of the window and were not directly optimized for this decoding task. Poor earlier-slot reconstruction does not prove no useful information. Positive paired difference is worse than D; equal-episode bootstrap, not LAM-seed uncertainty.'}
    (args.output_dir/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for name, result in decoder.items():
        print(name, result)
    print('Extraction/decoder comparison saved:', args.output_dir)


def fit(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads((args.readout_dir/'manifest.json').read_text())
    probe_report = json.loads((args.probe_dir/'results.json').read_text())
    if manifest['feature_manifest'] != probe_report['feature_manifest']:
        raise ValueError('Probe provenance mismatch')
    with np.load(args.probe_dir/'four_frame/preprocessing.npz', allow_pickle=False) as data:
        mean, scale = data['target_mean'], data['target_scale']
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Invalid target normalization')
    data = {}
    for split in ('train', 'validation'):
        with np.load(args.readout_dir/f'{split}.npz', allow_pickle=False) as saved:
            data[split] = {k: saved[k] for k in saved.files}
    if set(data['train']['record_indices']) & set(data['validation']['record_indices']):
        raise ValueError('Record leakage')
    y = {s: (data[s]['targets_raw'].astype(np.float64)-mean)/scale for s in data}
    baseline = y['train'].mean(0)
    args.output_dir.mkdir(parents=True)
    results, predictions = {}, {}
    for name, dim in (('D_quantized', 512), ('D_continuous', 512), ('all_continuous', 2048)):
        x = {s: data[s][name].astype(np.float64) for s in data}
        if any(v.shape != (len(y[s]), dim) or not np.isfinite(v).all() for s,v in x.items()):
            raise ValueError('Invalid readout features')
        best, grid = ridge_grid(x['train'], y['train'], x['validation'], y['validation'])
        _, alpha, weight, bias = best
        result = {'alpha': alpha, 'grid': grid}
        for split in data:
            prediction = x[split]@weight+bias
            predictions[f'{split}_{name}'] = prediction
            result[split] = summarize([prediction], y[split], baseline)
        np.savez(args.output_dir/f'{name}_ridge.npz', weight=weight, bias=bias, target_mean=mean, target_scale=scale)
        results[name] = result
        print(name, {g:r['nmse_mean'] for g,r in result['validation'].items()}, flush=True)
    denominator = np.var(y['validation'], axis=0).sum()
    errors = {name: ((predictions[f'validation_{name}']-y['validation'])**2).sum(1)/denominator for name in results}
    paired = {name: paired_interval(errors[name]-errors['D_continuous'], data['validation']['record_indices']) for name in ('D_quantized', 'all_continuous')}
    np.savez_compressed(args.output_dir/'predictions.npz', **predictions,
                        validation_targets_normalized=y['validation'], validation_record_indices=data['validation']['record_indices'])
    report = {'readouts': results, 'paired_minus_D_continuous': paired, 'decoder': manifest['decoder'],
              'readout_manifest': manifest,
              'protocol': 'Full features, no PCA; ridge sum-SSE, unpenalized intercept; alpha .01/.1/1/10/100/1000 selected on validation MSE; same endpoint7 labels and normalization.',
              'limitations': 'Exploratory validation tuning, not independent test. All-slots representation is4xlarger; a gain diagnoses distributed accessible information, not an equal-capacity four-code improvement. Continuous gains do not establish deployable quantized gains. Paired CIs are episode-weighted, fixed validation denominator, not selection-adjusted or LAM-seed uncertainty.'}
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Paired relative to D continuous:', paired)
    print('Saved', args.output_dir, '; no LAM updates or test access.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    extraction = sub.add_parser('extract')
    for name in ('cache-dir', 'checkpoint', 'dino-dir', 'four-frame-dir', 'features-dir', 'output-dir'):
        extraction.add_argument('--'+name, type=Path, required=True)
    extraction.add_argument('--batch-size', type=int, default=4)
    fitting = sub.add_parser('fit')
    for name in ('readout-dir', 'probe-dir', 'output-dir'):
        fitting.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    (extract if args.command == 'extract' else fit)(args)


if __name__ == '__main__':
    main()
