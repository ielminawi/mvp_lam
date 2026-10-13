"""Fresh-shard evaluation of frozen camera mappings; no fitting or selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import subprocess

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
# TensorFlow only decodes data; reserve GPU memory for PyTorch.
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
import tensorflow_datasets as tfds
import numpy as np
import torch
from torchvision.transforms.functional import resize
from latent_action_model.genie.modules.lam import DINOLatentActionModel


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def evaluate(episodes, offset, target_mean):
    per_episode = []
    totals = {key: 0. for key in ('no_correction', 'offset', 'target_mean')}
    count = 0
    for entry in episodes:
        source, target = entry['source'].astype(np.float64), entry['target'].astype(np.float64)
        scores = {
            'no_correction': float(np.mean((source-target)**2)),
            'offset': float(np.mean((source+offset-target)**2)),
            'target_mean': float(np.mean((target_mean-target)**2)),
        }
        for key in scores:
            totals[key] += len(source)*scores[key]
        count += len(source)
        per_episode.append({'record_index': entry['record_index'], 'pairs': len(source), 'mse': scores,
                            'offset_error_reduction_percent': 100*(1-scores['offset']/scores['no_correction']) if scores['no_correction'] > 1e-12 else None})
    macro = {key: float(np.mean([row['mse'][key] for row in per_episode])) for key in totals}
    return {'episodes': per_episode, 'episode_mean_mse': macro,
            'pair_mean_mse': {key: value/count for key, value in totals.items()},
            'episode_mean_error_reduction_percent': 100*(1-macro['offset']/macro['no_correction']) if macro['no_correction'] > 1e-12 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'checkpoint', 'dino-dir', 'output-dir', 'reference-dir', 'mapping-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--shard-name', default='berkeley_cable_routing-train.tfrecord-00001-of-00064')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output_dir}')
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise RuntimeError('Positive batch size and Slurm GPU allocation required')
    reference = json.loads((args.reference_dir/'results.json').read_text())
    torch.set_num_threads(1)
    if Path(args.shard_name).name != args.shard_name:
        raise ValueError('Shard must be a filename')
    shard = args.data_dir/args.shard_name
    checkpoint_hash = sha(args.checkpoint)
    if checkpoint_hash != 'd388a242d603807eb97f6d80063d6d7daf000a55cfdf59498666eee87a2fe3b6':
        raise ValueError('Checkpoint differs from verified release')
    if checkpoint_hash != reference['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from original experiment')
    mapping_report = json.loads((args.mapping_dir/'results.json').read_text())
    if sha(args.reference_dir/'results.json') != mapping_report['reference_results_sha256']:
        raise ValueError('Mapping reference does not match supplied diagnostics')
    if subprocess.check_output(['git','-C',str(args.dino_dir),'rev-parse','HEAD'],text=True).strip() != reference['dino_commit']:
        raise ValueError('DINO version changed')
    with np.load(args.mapping_dir/'models.npz',allow_pickle=False) as saved:
        fitted = {key:saved[key] for key in saved.files}
    builder = tfds.builder_from_directory(str(args.data_dir))
    original_load = torch.hub.load
    def local_load(repo, name, *positional, **kwargs):
        if (repo, name) != ('facebookresearch/dinov2', 'dinov2_vitb14_reg'):
            raise ValueError('Unexpected hub request')
        return original_load(str(args.dino_dir.resolve()), name, source='local', pretrained=False)
    with patch('torch.hub.load', side_effect=local_load):
        model = DINOLatentActionModel(in_dim=3, model_dim=768, latent_dim=128,
            num_latents=16, patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint)
    if not all(key.startswith('lam.') for key in state):
        raise ValueError('Unexpected checkpoint prefix')
    model.load_state_dict({key.removeprefix('lam.'): value for key, value in state.items()}, strict=True)
    del checkpoint, state
    model = model.cuda().eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    episodes, fingerprints = [], set()
    for record_index, record in enumerate(tf.data.TFRecordDataset(str(shard))):
        steps = builder.info.features.deserialize_example(record)['steps']
        if isinstance(steps, dict):
            steps = tf.data.Dataset.from_tensor_slices(steps)
        images = {name: [] for name in ('image', 'top_image')}
        for step in steps:
            for name in images:
                images[name].append(step['observation'][name].numpy())
        if len(images['image']) <= 9:
            raise ValueError(f'Episode {record_index} too short; revise protocol explicitly')
        images = {name: np.stack(values) for name, values in images.items()}
        fingerprint = hashlib.sha256(b''.join(images[name].tobytes() for name in images)).hexdigest()
        if fingerprint in set(reference['episode_fingerprints'].values()):
            raise ValueError('Fresh shard contains an exact paired-episode duplicate of original data')
        if fingerprint in fingerprints:
            raise ValueError('Exact duplicate paired episode; revise split before testing')
        fingerprints.add(fingerprint)
        starts = np.arange(len(images['image'])-9, dtype=np.int32)
        ends = starts+9
        arrays = {'start_indices': starts, 'end_indices': ends}
        for name, label in (('image', 'source'), ('top_image', 'target')):
            values, codes, continuous = [], [], []
            for first in range(0, len(starts), args.batch_size):
                a, b = starts[first:first+args.batch_size], ends[first:first+args.batch_size]
                pixels = np.stack((images[name][a], images[name][b]), axis=1)
                if np.any(pixels.reshape(len(a)*2, -1).std(axis=1) == 0):
                    raise ValueError('Constant image encountered')
                videos = torch.from_numpy(pixels).permute(0,1,4,2,3).float()/255.
                videos = resize(videos.flatten(0,1), [224,224], antialias=True).reshape(len(a),2,3,224,224).cuda()
                with torch.inference_mode():
                    output = model.vq_encode(videos)
                    feature = output['z_q'].flatten(1).cpu().numpy()
                    if feature.shape != (len(a),512) or not np.isfinite(feature).all():
                        raise ValueError('Invalid latents')
                    values.append(feature)
                    codes.append(output['indices'].cpu().numpy())
                    emb = output['emb'].flatten(1).cpu().numpy()
                    if emb.shape != (len(a),512) or not np.isfinite(emb).all():
                        raise ValueError('Invalid continuous embeddings')
                    continuous.append(emb)
            arrays[label] = np.concatenate(values)
            arrays[label+'_codes'] = np.concatenate(codes)
            arrays[label+'_continuous'] = np.concatenate(continuous)
        np.savez(args.output_dir/f'episode_{record_index:05d}.npz', **arrays)
        episodes.append({'record_index': record_index, 'fingerprint': fingerprint, **arrays})
        print(f'Episode {record_index}: {len(starts)} paired transitions extracted', flush=True)
    if not episodes:
        raise ValueError('No fresh episodes')
    report = {
        'scope': __doc__, 'shard': str(shard), 'shard_sha256': sha(shard),
        'checkpoint_sha256': checkpoint_hash, 'horizon_steps': 9,
        'mapping_results_sha256': sha(args.mapping_dir/'results.json'),
        'reference_results_sha256': sha(args.reference_dir/'results.json'),
        'model_file_sha256': sha(args.mapping_dir/'models.npz'),
        'episode_count': len(episodes),
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'episode_fingerprints': {str(e['record_index']):e['fingerprint'] for e in episodes},
        'protocol': 'All t,t+9 pairs; same preprocessing; fixed source=image target=top_image. Equal episode weighting. Saved ridge, offset and mean baselines; no refitting or hyperparameter selection.',
        'limitations': 'Camera stability/arrangement assessed visually, precise synchronization unverified. Original pretraining overlap unknown. Exact paired-image duplicates checked; near-duplicate/source overlap cannot be ruled out.',
        'results': {}}
    predictions = {}
    for representation in ('quantized','continuous'):
        suffix = '' if representation == 'quantized' else '_continuous'
        held = [{'record_index':e['record_index'],'source':e['source'+suffix].astype(np.float64),
                 'target':e['target'+suffix].astype(np.float64)} for e in episodes]
        with np.load(args.reference_dir/f'{representation}_calibration.npz',allow_pickle=False) as saved:
            offset,average = saved['offset'],saved['target_mean']
        result = evaluate(held,offset,average)
        w = fitted[representation+'_mapping']
        mx,my = fitted[representation+'_source_mean'],fitted[representation+'_target_mean']
        rows=[]
        for e,baseline_row in zip(held,result['episodes']):
            pred = (e['source']-mx)@w+my
            error=float(np.mean((pred-e['target'])**2))
            base=baseline_row['mse']['target_mean']
            reduction=100*(1-error/base) if base>1e-12 else None
            rows.append({'record_index':e['record_index'],'pairs':len(pred),'ridge_mse':error,
                         'target_mean_mse':base,'ridge_error_reduction_vs_mean_percent':reduction})
            predictions[f"{representation}_episode_{e['record_index']:05d}_ridge"] = pred
        result['episode_mean_mse']['ridge']=float(np.mean([r['ridge_mse'] for r in rows]))
        result['ridge_episode_results']=rows
        result['episodes_beating_mean']=sum(r['ridge_mse']<r['target_mean_mse'] for r in rows)
        result['selected_strength_from_original_calibration']=mapping_report['results'][representation]['selected_strength']
        report['results'][representation]=result
        print(f'\n{representation.upper()}: FRESH SHARD, FROZEN MODELS',flush=True)
        for name,value in result['episode_mean_mse'].items():
            print(f'  {name}: MSE={value:.8f}',flush=True)
        ridge=result['episode_mean_mse']['ridge']
        for name in ('no_correction','offset','target_mean'):
            baseline=result['episode_mean_mse'][name]
            print(f'  ridge error reduction vs {name} (%): {100*(1-ridge/baseline):.3f}',flush=True)
        print(f"  Beats mean baseline: {result['episodes_beating_mean']}/{len(rows)} episodes",flush=True)
        for row in rows:
            print(f"  record {row['record_index']}: ridge={row['ridge_mse']:.6f}; mean={row['target_mean_mse']:.6f}; reduction={row['ridge_error_reduction_vs_mean_percent']:.2f}%",flush=True)
    np.savez(args.output_dir/'predictions.npz',**predictions)
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2))
    print('\nSaved:',args.output_dir/'results.json')
    print('No mapping refitting, calibration, hyperparameter selection, or LAM training performed.')


if __name__ == '__main__':
    main()
