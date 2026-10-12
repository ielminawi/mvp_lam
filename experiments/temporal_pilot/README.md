# Fixed-horizon temporal pilot

Question: do intermediate frames improve a four-code representation of the **same** A→D transition?

This is single-view adaptation of a multiview-pretrained checkpoint. It does not verify multiview superiority or reproduce original MVP-LAM training. `config.json` is a proposed 500-update pilot, not a tuned or validated schedule. `smoke.json` only checks two optimization steps; its scores are not scientific results.

## Matched comparison

| Variant | Original-frame offsets | Finetuning |
| --- | --- | --- |
| Released reference | 0, 9 | None |
| two_frame | 0, 9 | Encoder, action slots, projection |
| four_frame | 0, 3, 6, 9 | Same |

Freeze DINO, codebook, patch/action decoder projections and decoder. Disable code restarts. Keep causal attention and the existing spatial positional encoding; do not add elapsed-time conditioning. Both variants return only the final four slots. Encoder adaptation uses reconstruction MSE plus 0.25 times commitment MSE. The decoder is frozen **but remains differentiable**. Robot labels are not used in training.

Adam at 1e-5, zero weight decay, fixed LR, float32, clipping at norm1; microbatch1 × accumulation8 = effective batch8; seed42, 500 updates, validation every100. Both variants consume the identical seeded shuffled window stream. Equal updates/examples are not equal wall-clock compute. These settings differ from linear-probe training.

## Code boundary

`WholeWindowLAM` subclasses the released architecture without altering its `vq_encode`/`forward`. The experimental path accepts cached DINO features, selects final-frame slots and reconstructs only the final frame from the first. The original checkpoint loads strictly without new parameters. Spatial positional encoding is now a nonpersistent buffer transferred to the input device/dtype, preserving checkpoint keys. An unavailable legacy-module import is removed; the unused rotary dependency is lazy-loaded.

## Git workflow

Local branch: `TMVP-LAM`. Commit/push only source, configurations and documentation. Euler pulls the same branch. Manifests record the Git commit/diff, model hash, DINO revision, Torch version and GPU. Features, prepared data and model results belong in scratch/home, not Git.

Euler may have the earlier one-line `modules/__init__.py` fix uncommitted. Before pulling, inspect `git diff -- latent_action_model/genie/modules/__init__.py`. If that is its only tracked change, preserve it with `git stash push -m "pre-pilot import fix" -- latent_action_model/genie/modules/__init__.py`, then pull. The committed version includes this fix; do not reapply the stash blindly. Never reset unrelated work. Use `git pull --ff-only origin TMVP-LAM` on the correct branch.

## First run: smoke only

Run all commands from the repository root with `mvplam` active inside a GPU allocation. No new dependencies are required relative to the working inference environment. Allocate enough time for caching and validation; inspect memory/runtime before the full pilot.

```bash
python -m scripts.temporal_pilot cache \
  --sample-dir /cluster/scratch/ielminawi/mvp_lam/data/probe_sample_v3 \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_smoke_cache \
  --limit-episodes 2

python -m scripts.temporal_pilot check \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_smoke_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2

python -m scripts.temporal_pilot train \
  --variant two_frame --config experiments/temporal_pilot/smoke.json \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_smoke_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_smoke_two

python -m scripts.temporal_pilot train \
  --variant four_frame --config experiments/temporal_pilot/smoke.json \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_smoke_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_smoke_four
```

Cache only train/validation; never load test episode files. Four-frame validity filtering is shared by both variants. Cached feature matrices are float32 `.npy` files on scratch with checksum-protected metadata. They may require tens of GB for the full sample; cache once and reuse. `--limit-episodes 2` selects two episodes **per split** for an implementation smoke check; omit it only for the full cache.

`check` verifies exact two-frame path equivalence on cached patches, encoder gradients through the frozen decoder, and frozen-parameter invariance after 2/4-frame optimizer steps. CPU unit tests use fake DINO features, not released-checkpoint GPU execution:

```bash
python -m unittest discover -s tests -p 'test_temporal_pilot.py'
```

## Full pilot after smoke checks

Create a separate full cache without `--limit-episodes`; use `config.json` and separate two_frame/four_frame output directories. Inspect the cache manifest for dropped intermediate-frame windows. Keep the same cache/config for both models. `latest.pt` saves adapted encoding parameters and metadata at each validation boundary, not a whole replacement LAM checkpoint. Load the verified released base and overlay these parameters to evaluate. Optimizer resume is not implemented: interrupted runs must restart into a new directory. No best-validation or test-based selection; use the predeclared final update.

First compare frozen-reference reconstruction against the two-frame run's initial score and matched final reconstruction/code usage. A four-frame score at update0 is an untrained longer-input diagnostic, not another released baseline. Lower reconstruction error alone does not establish better actions. A later stage will extract features from each adapted model and fit matched new PCA/probes; that evaluator is not part of this first training smoke test. Reserve fresh test episodes before final extension claims; V3 test has already been inspected.
