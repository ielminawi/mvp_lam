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

## Matched action probing after both full pilots

`scripts.probe_temporal_pilot` compares released endpoints, adapted endpoints, and adapted four-frame windows. It verifies identical cache provenance/training configurations, final-update checkpoints, exact adapted-parameter keys and frozen base weights. Common raw action labels and frame/episode identities come from the cached window metadata; the V3 horizon-aware normalization is reused. Extraction never loads test files.

Run extraction in a GPU allocation after pulling this revision on the login node:

```bash
python -m scripts.probe_temporal_pilot extract \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_pilot_cache \
  --sample-dir /cluster/scratch/ielminawi/mvp_lam/data/probe_sample_v3 \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --two-frame-dir notes/temporal_pilot_two_v1 \
  --four-frame-dir notes/temporal_pilot_four_v1 \
  --output-dir notes/temporal_probe_features_v1 \
  --batch-size 4
```

Then fit probes (CPU; may run in the same allocation):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.probe_temporal_pilot fit \
  --features-dir notes/temporal_probe_features_v1 \
  --output-dir notes/temporal_probe_results_v1
```

Each representation gets separately fitted training-only centered PCA128 (no whitening), then four Linear128→7 probes with seeds42–45, Adam .001, batch512, 30epochs, cosine per epoch to0, no clipping/weight decay. Batch orders and initial probe weights match across representations. PCA coordinates can differ; least squares provides a supplementary training-only reference less sensitive to finite Adam fitting. Models, preprocessing, raw predictions and overall/component train/validation NMSE are saved. All use the final epoch; there is no checkpoint selection. Group error reduction uses the training-mean action baseline. Zero-variance group NMSE is undefined. Gripper threshold accuracy is supplementary.

Probe-seed variation does not estimate variation across LAM training seeds: each adapted LAM currently has one seed. These are development validation results, not final test claims. Historical V3 scores should be compared cautiously if cache filtering changed the windows; the newly extracted released reference is the matched baseline.

## Audit completed pilot curves and code usage

This reads saved training reports and optionally extracted code indices. It does not load models, train, select a new checkpoint or access test data. Run on CPU (an existing GPU allocation also works):

```bash
python -m scripts.audit_temporal_pilot \
  --two-frame-dir notes/temporal_pilot_two_v1 \
  --four-frame-dir notes/temporal_pilot_four_v1 \
  --features-dir notes/temporal_probe_features_v1 \
  --output-dir notes/temporal_pilot_audit_v1
```

Outputs: `results.json`, `training_curves.png/.pdf`, `code_usage.png/.pdf`. Requires NumPy and Matplotlib. The audit checks matched configurations/cache/frozen weights, complete finite histories and validation code counts. Extracted validation indices must match final logged code counts. Plots distinguish full-split validation from trailing means of changing training batches. The JSON also reports clipping frequency, example presentations, per-slot code perplexity and observed four-code tuple diversity. Marginal perplexity ranges from 1 to 16; high marginal usage alone can hide fixed assignments per slot. Neither diversity nor a flattening curve proves action quality or convergence. Best logged validation update is descriptive; the predeclared final models remain selected. Uncertainty from one LAM adaptation seed remains unresolved.

```bash
python -m unittest discover -s tests -p 'test_temporal_audit.py'
```

## Longer matched run: 2,500 updates

`long.json` differs from `config.json` only in `updates` (500 → 2500). Both variants restart from the released checkpoint with seed42; this is a 2,500-update run in total, not 2,500 additional updates. Restarting recreates Adam state and the seeded example stream from update1; loading the old encoder weights alone would not be an exact continuation. Existing 500-update outputs stay intact. With effective batch8, each variant sees 20,000 example presentations (2.251 passes for 8,886 windows). Validation remains every100 updates; final checkpoint selection remains predeclared. This experiment extends optimization, with one adaptation seed, and does not establish general superiority.

Reuse the full cache; no recaching is needed. Run each command in a GPU allocation with sufficient time. Runtime is roughly five times its earlier 500-update run, plus initialization; run variants in separate allocations if necessary. There is no optimizer resume, so a time-limited interruption cannot be continued exactly from `latest.pt`.

```bash
python -m scripts.temporal_pilot train \
  --variant two_frame \
  --config experiments/temporal_pilot/long.json \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_pilot_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_pilot_two_2500_v1

python -m scripts.temporal_pilot train \
  --variant four_frame \
  --config experiments/temporal_pilot/long.json \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_pilot_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_pilot_four_2500_v1
```

After both complete, audit these new run directories into `notes/temporal_pilot_audit_2500_v1`. Omit `--features-dir` until features have been extracted from the new checkpoints: the previous feature directory belongs to the 500-update models. Then repeat matched extraction/probing with the new model directories and separate feature/result outputs. Do not use V3 test results to select duration or checkpoints.

## Fixed-endpoint intermediate-frame diagnostic

Use the completed 2,500-update four-frame model and its existing probes. No training or PCA/probe refitting. In a GPU allocation:

```bash
python -m scripts.diagnose_temporal_intermediates \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_pilot_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --four-frame-dir notes/temporal_pilot_four_2500_v1 \
  --probe-dir notes/temporal_probe_results_2500_v1 \
  --output-dir notes/temporal_intermediate_diagnostic_v1 \
  --batch-size 4
```

Conditions: correct A,B,C,D; repeated endpoints A,A,D,D; reversed middle A,C,B,D. Endpoints, reconstruction targets and action targets remain identical. Read only cached validation episodes. Measure final four continuous vectors before quantization (`emb`), per-slot and any code changes, reconstruction MSE, and action NMSE averaged over four frozen probes. Correct-input predictions must reproduce saved reference predictions within numerical tolerance; failures require investigation, not relaxed thresholds by default. Cache/checkpoint/probe provenance is checked.

`results.json` reports metrics and paired error differences. `per_window.npz` retains representations, indices, predictions, errors and episode/frame identities. Positive intervention-minus-correct error means the intervention is worse. Global scores weight windows equally; paired confidence intervals use equal-episode mean differences and resample entire episodes (2,000 draws, seed20261013). NMSE intervals use a fixed full-validation denominator, so these intervals describe the episode-averaged normalized error difference, not refitted bootstrap NMSE. They do not measure LAM-seed uncertainty. Altered sequences may be out of distribution; sensitivity alone does not prove beneficial temporal reasoning. No attention-map interpretation is needed.

```bash
python -m unittest discover -s tests -p 'test_temporal_interventions.py'
```

## Prediction changes and probe-capacity diagnostic

CPU commands; may also run inside an existing GPU allocation. No LAM fitting, decoder changes, raw data access or test access:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.analyze_temporal_prediction_changes \
  --diagnostic-dir notes/temporal_intermediate_diagnostic_v1 \
  --probe-dir notes/temporal_probe_results_2500_v1 \
  --output-dir notes/temporal_prediction_changes_v1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.check_probe_bottleneck \
  --features-dir notes/temporal_probe_features_2500_v1 \
  --probe-dir notes/temporal_probe_results_2500_v1 \
  --output-dir notes/temporal_probe_capacity_v1
```

The first command measures saved action-prediction changes in normalized units, separately for translation/orientation/gripper, and counts episodes improving/worsening. Equal errors do not imply identical predictions. It cannot measure full decoder-output differences because these were not saved.

The second compares all three frozen representations using ridge regression on PCA128/full512 and a one-hidden-layer MLP on PCA128/full512. Ridge alpha grid: .01,.1,1,10,100,1000 (sum-SSE penalty, unpenalized bias), selected by validation MSE. MLP: 64 hidden units, ReLU, Adam .001, weight decay .0001, batch512, maximum200epochs, patience20, best validation MSE; seeds42–45. Same existing training-only mean/PCA and target normalization; no whitening. A full512 MLP with fixed shuffled training labels is a negative control, evaluated against correct labels. Store models, preprocessing and normalized train/validation predictions plus results.

This is exploratory: validation is reused for selection and reporting, so scores are selection-biased and cannot be treated as independent benchmark improvements. Nonlinear versus original affine comparisons also change optimization duration; ridge references help assess linear optimization. Compare PCA/full ridge to isolate truncation, PCA/full MLP to inspect truncation under the same nonlinear protocol, and ridge/MLP to assess accessible nonlinear information. Training feature rank was below128, so large PCA losses are not expected a priori. The affine original probe and least-squares reference remain available. Four probe seeds do not measure LAM-seed uncertainty. A nonlinear gain for all models is not evidence that intermediate frames help; compare four-frame against matched two-frame and released results. Reserve a fresh final test before any confirmatory claim.

```bash
python -m unittest discover -s tests -p 'test_probe_bottleneck.py'
```
