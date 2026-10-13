"""Frozen paired-view LAM pilot: episode-calibrated additive camera offset."""
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
    for name in ('data-dir', 'checkpoint', 'dino-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output_dir}')
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise RuntimeError('Positive batch size and Slurm GPU allocation required')
    torch.set_num_threads(1)
    shard = args.data_dir/'berkeley_cable_routing-train.tfrecord-00000-of-00064'
    checkpoint_hash = sha(args.checkpoint)
    if checkpoint_hash != 'd388a242d603807eb97f6d80063d6d7daf000a55cfdf59498666eee87a2fe3b6':
        raise ValueError('Checkpoint differs from verified release')
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
        if fingerprint in fingerprints:
            raise ValueError('Exact duplicate paired episode; revise split before testing')
        fingerprints.add(fingerprint)
        starts = np.arange(len(images['image'])-9, dtype=np.int32)
        ends = starts+9
        arrays = {'start_indices': starts, 'end_indices': ends}
        for name, label in (('image', 'source'), ('top_image', 'target')):
            values, codes = [], []
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
            arrays[label] = np.concatenate(values)
            arrays[label+'_codes'] = np.concatenate(codes)
        np.savez(args.output_dir/f'episode_{record_index:05d}.npz', **arrays)
        episodes.append({'record_index': record_index, 'fingerprint': fingerprint, **arrays})
        print(f'Episode {record_index}: {len(starts)} paired transitions extracted', flush=True)
    if len(episodes) != 20:
        raise ValueError('Expected 20 episodes for the predefined 10/10 split')
    order = np.random.default_rng(42).permutation(20)
    calibration = [episodes[i] for i in order[:10]]
    test = [episodes[i] for i in order[10:]]
    offset = np.mean([(e['target'].astype(np.float64)-e['source']).mean(axis=0) for e in calibration], axis=0)
    target_mean = np.mean([e['target'].astype(np.float64).mean(axis=0) for e in calibration], axis=0)
    np.savez(args.output_dir/'calibration.npz', offset=offset, target_mean=target_mean)
    report = {'scope': __doc__, 'horizon_steps': 9, 'seed': 42,
        'pair_policy': 'All valid endpoint pairs t,t+9, including first and final observations; Cable-specific pilot, no Bridge trimming.',
        'representation': 'z_q flattened to 512; raw latent-coordinate MSE, no PCA or action normalization',
        'calibration': 'Average within each episode, then average across 10 calibration episodes',
        'calibration_record_indices': [e['record_index'] for e in calibration],
        'test_record_indices': [e['record_index'] for e in test],
        'checkpoint_sha256': checkpoint_hash, 'shard_sha256': sha(shard),
        'episode_fingerprints': {str(e['record_index']): e['fingerprint'] for e in episodes},
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'dino_commit': subprocess.check_output(['git','-C',str(args.dino_dir),'rev-parse','HEAD'], text=True).strip(),
        'torch_version': str(torch.__version__), 'tensorflow_version': tf.__version__,
        'preprocessing': 'uint8/255; bilinear antialiased 224x224 resize; internal DINO normalization',
        'limitations': 'Small exploratory sample. Visual camera stability only; exact synchronization and original source deduplication unverified. Test episodes may overlap original LAM training. Offset need not lie on codebook manifold.',
        'test': evaluate(test, offset, target_mean)}
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2))
    print('\nTEST: equal episode weighting')
    for name, value in report['test']['episode_mean_mse'].items():
        print(f'  {name}: MSE={value:.8f}')
    print('Offset error reduction (%):', report['test']['episode_mean_error_reduction_percent'])
    print('Per-episode offset error reduction (%):')
    for row in report['test']['episodes']:
        print(f"  record {row['record_index']}: {row['offset_error_reduction_percent']}")
    print('Saved:', args.output_dir/'results.json')
    print('No LAM training, action probing, or decoder evaluation performed.')


if __name__ == '__main__':
    main()
