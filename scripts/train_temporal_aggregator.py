"""Train only a gated readout over frozen released encoder slots; resumable pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from latent_action_model.genie.modules.temporal_aggregator import AggregatedLAM
from scripts.temporal_pilot import RELEASE_HASH, Windows, frozen_hash, load_model, provenance, read_config, sha256, validate


def atomic_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def sample_stream(generator, order, cursor, count):
    selected = []
    while len(selected) < count:
        take = min(count-len(selected), len(order)-cursor)
        selected.extend(order[cursor:cursor+take].tolist()); cursor += take
        if cursor == len(order):
            order, cursor = generator.permutation(len(order)), 0
    return selected, order, cursor


def train(args, config, manifest):
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    backbone = load_model(args)
    model = AggregatedLAM(backbone).cuda()
    training, validation = [Windows(args.cache_dir, manifest, split) for split in ('train', 'validation')]
    parameters = list(model.aggregator.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = np.random.default_rng(config['seed'])
    before = frozen_hash(model)
    cache_hash = sha256(args.cache_dir/'manifest.json')
    run = {'variant': args.variant, 'config': config, 'cache_manifest_sha256': cache_hash,
           'frozen_weight_sha256': before, 'train_pairs': len(training), 'validation_pairs': len(validation),
           'backbone': 'released checkpoint, not previously adapted; all backbone parameters frozen',
           'base_checkpoint_sha256': RELEASE_HASH, 'provenance': json.loads(json.dumps(provenance(args)))}
    order, cursor, history, start = generator.permutation(len(training)), 0, [], 0
    if args.resume:
        if not (args.output_dir/'latest.pt').exists():
            raise FileNotFoundError('No complete resumable checkpoint')
        saved = torch.load(args.output_dir/'latest.pt', map_location='cpu', weights_only=True)
        for key in ('variant', 'config', 'cache_manifest_sha256', 'frozen_weight_sha256', 'train_pairs', 'validation_pairs', 'base_checkpoint_sha256'):
            if saved['run'][key] != run[key]:
                raise ValueError(f'Resume mismatch: {key}')
        if saved['update'] >= config['updates']:
            raise ValueError('Run already complete')
        model.aggregator.load_state_dict(saved['aggregator'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        generator.bit_generator.state = saved['numpy_generator_state']
        order, cursor = saved['order'].numpy(), saved['cursor']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history, start = saved['history'], saved['update']
        run['resume_provenance'] = run['provenance']; run['provenance'] = saved['run']['provenance']
        print('Resuming after update', start, flush=True)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        initial = validate(model, validation, args.variant, config['micro_batch_size'])
        history = [{'update': 0, 'validation': initial, 'gate': 0.}]
        print('Initial validation:', initial, flush=True)
    (args.output_dir/'run.json').write_text(json.dumps(run, indent=2))
    def save(update):
        atomic_checkpoint(args.output_dir/'latest.pt', {
            'run': run, 'update': update, 'aggregator': model.aggregator.state_dict(), 'optimizer': optimizer.state_dict(),
            'numpy_generator_state': generator.bit_generator.state, 'order': torch.from_numpy(order.copy()), 'cursor': cursor,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'history': history})
        # Rewrite from checkpoint history on resume, removing any unsaved partial tail.
        (args.output_dir/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    save(start)
    batch = config['micro_batch_size']*config['accumulation_steps']
    for update in range(start+1, config['updates']+1):
        selected, order, cursor = sample_stream(generator, order, cursor, batch)
        optimizer.zero_grad(set_to_none=True); totals = np.zeros(3)
        model.train()
        for first in range(0, batch, config['micro_batch_size']):
            patches = training.batch(selected[first:first+config['micro_batch_size']], args.variant)
            losses = model.pilot_loss(model.reconstruct_window(patches), config['commitment_beta'])
            if not all(torch.isfinite(v).item() for v in losses):
                raise ValueError('Nonfinite loss')
            (losses[0]/config['accumulation_steps']).backward()
            totals += np.array([v.detach().item() for v in losses])/config['accumulation_steps']
        norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        row = {'update': update, 'training_total': float(totals[0]), 'training_reconstruction': float(totals[1]),
               'training_commitment': float(totals[2]), 'gradient_norm_before_clip': float(norm),
               'gate': float(torch.tanh(model.aggregator.gate).detach())}
        if update % config['validation_every'] == 0 or update == config['updates']:
            row['validation'] = validate(model, validation, args.variant, config['micro_batch_size'])
        history.append(row)
        if 'validation' in row:
            if frozen_hash(model) != before:
                raise ValueError('Frozen backbone weights changed')
            save(update)
            print(f"Update {update}: train MSE={totals[1]:.6f}; validation={row['validation']['reconstruction_mse']:.6f}; gate={row['gate']:.6f}", flush=True)
    report = {**run, 'history': history, 'frozen_weights_unchanged': frozen_hash(model)==before,
              'trainable_parameters': sum(p.numel() for p in parameters),
              'checkpoint_selection': 'predeclared final update, no test access',
              'training': 'Only query attention, normalization and gate; reconstruction + commitment, no numerical labels'}
    (args.output_dir/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Aggregator pilot completed:', args.output_dir/'results.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir', 'checkpoint', 'dino-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('experiments/temporal_aggregator/config.json'))
    parser.add_argument('--variant', choices=('two_frame', 'four_frame'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        parser.error('Use a GPU allocation')
    config = read_config(args.config)
    manifest = json.loads((args.cache_dir/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['offsets'] != [0,3,6,9] or manifest['provenance']['checkpoint_sha256'] != RELEASE_HASH:
        raise ValueError('Invalid cache')
    if any(e['split'] not in ('train', 'validation') for e in manifest['episodes']):
        raise ValueError('Unexpected split')
    sources = {s:{e['source_path'] for e in manifest['episodes'] if e['split']==s} for s in ('train','validation')}
    if sources['train'] & sources['validation']:
        raise ValueError('Source leakage')
    if manifest['limit_episodes_per_split'] and config['updates'] > 2:
        raise ValueError('Use full cache for scientific pilot')
    for entry in manifest['episodes']:
        for key,digest in (('patch_file','patch_sha256'),('window_file','window_sha256')):
            if sha256(args.cache_dir/entry[key]) != entry[digest]:
                raise ValueError('Changed cache input')
    train(args, config, manifest)


if __name__ == '__main__':
    main()
