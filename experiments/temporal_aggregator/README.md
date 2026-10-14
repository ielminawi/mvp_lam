# Gated four-query aggregation pilot

Question: can attention over all contextualized encoder slots improve a compact four-code readout, compared with selecting D's slots directly? Both two-frame and four-frame use the same **released** frozen backbone, not the previously separately adapted encoders. This isolates readout learning under shared encoder weights. Endpoints remain offsets0,9; intermediate frames3,6; decoder still reconstructs D from A. No cross-view objective, numerical action supervision or test access.

## Model and objective

Freeze DINO, original action slots, spatial-temporal encoder, to_codebook projection,16-entry codebook, action/patch decoder input projections and decoder. Disable code restarts; backbone always eval. Train only four768D summary queries, LayerNorm and multi-head-attention weights (12heads,dropout0), and one scalar gate. Encoder outputs are computed without gradients; projection/quantizer/decoder stay differentiable.

Readout: `D_slots + tanh(gate) * MHA(queries, norm(all_slots), norm(all_slots))`. Eight slots for endpoints input; sixteen for four frames. Gate starts0, so initial output exactly matches the original final-slot readout. At first step the gate learns; attention/query gradients begin once it opens. No new temporal embeddings are added. Existing causal context is carried in encoder outputs, while aggregation attends across the observed window. Output remains four768D slots→frozenprojectionfour128D→fourcodes. A continuous512D readout is not equivalent in information capacity to four discrete IDs.

Loss: reconstruction MSE +.25commitment. Adam1e−4, noWD, clipping1, float32, seed42, effectivebatch8 (micro1×accum8),500updates, validationevery100. Learning rate is for newly initialized aggregation parameters and deliberately differs from earlier encoder adaptation1e−5. `smoke.json`:2updates,effectivebatch1,valevery1; not scientific. Final update predeclared. Both variants use same seeded shuffled sample stream. Reconstruction improvement alone does not establish better action representation; later matched probes are required.

## Euler smoke run first

Pull TMVP-LAM on the login node. Activate mvplam inside a GPU allocation. Reuse the existing smoke DINO cache:

```bash
python -m unittest discover -s tests -p 'test_temporal_aggregator.py'

python -m scripts.train_temporal_aggregator \
  --variant two_frame --config experiments/temporal_aggregator/smoke.json \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_smoke_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_aggregator_smoke_two_v1

python -m scripts.train_temporal_aggregator \
  --variant four_frame --config experiments/temporal_aggregator/smoke.json \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_smoke_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_aggregator_smoke_four_v1
```

CPU tests use a one-block backbone and deterministic synthetic input, not the released GPU checkpoint. They verify exact zero-gate compatibility, gradients after gate opening, frozen-weight invariance, and restoration of optimizer/sample-stream states.

## Full matched500-update pilot after smoke

```bash
python -m scripts.train_temporal_aggregator \
  --variant two_frame --config experiments/temporal_aggregator/config.json \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_pilot_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_aggregator_two_v1

python -m scripts.train_temporal_aggregator \
  --variant four_frame --config experiments/temporal_aggregator/config.json \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_pilot_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_aggregator_four_v1
```

No recaching needed. Frozen encoder forward is still computed for each selected batch; frozen slots are not separately cached in this version. Do not compare this aggregator checkpoint format with earlier encoder-overlay formats. `latest.pt` contains aggregator weights, Adam state, numpy generator state, permutation/cursor, torch/CUDA RNG states and checkpoint history. Writes use temporary file+rename. Checkpoints at update0 and each validation boundary. `results.json` appears only after final completion; `run.json`/`history.jsonl` record metadata/losses/gate.

## Resume an interrupted aggregator pilot

Repeat the same command, same directory/config/cache/variant, with `--resume`. Example:

```bash
python -m scripts.train_temporal_aggregator \
  --variant four_frame --config experiments/temporal_aggregator/config.json \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_pilot_cache \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_aggregator_four_v1 --resume
```

Resume restores the last fully saved validation boundary, discarding unsaved work afterward. It validates checkpoint/config/cache/frozenweights before continuation. RNG and optimizer state are preserved; bitwise equivalence across GPU hardware/software is not guaranteed because deterministic CUDA execution is not enforced. Do not increase the training budget silently on resume or delete incomplete directories. Completed runs refuse resume. An interrupted GPU save leaves the previous complete checkpoint; temporary files are not selected for resume.

## Interpretation

Initial two-frame and four-frame scores differ because the released encoder sees different input lengths; zero-gate aggregator preserves each respective baseline. Compare matched final models and changes from their own baseline. Both helping equally supports a better readout, not an intermediate-frame benefit. Only continuous improvement points toward remaining quantization limits. One adaptation seed/small budget is a pilot; reserve fresh data and independent seeds for confirmation. Keep D-only, endpoint-action and path-action references for later probing. Do not select training duration by already inspected test results.

## Matched endpoint probes after both runs complete

Extract trained two/four aggregator representations and their direct-D zero-gate controls in one frozen pass per variant; no numerical label supervision or LAM updates. Validate run completion, matched configuration/cache, released frozen weights, initial/final code usage, shared labels/identities, and released two-frame reference IDs/vectors. Record relative768D summary correction magnitude (not just gate). GPU:

```bash
python -m scripts.probe_temporal_aggregator extract \
  --cache-dir /cluster/scratch/ielminawi/mvp_lam/data/temporal_pilot_cache \
  --features-dir notes/temporal_probe_features_2500_v1 \
  --two-frame-dir notes/temporal_aggregator_two_v1 \
  --four-frame-dir notes/temporal_aggregator_four_v1 \
  --checkpoint checkpoints/mvp_lam.bin \
  --dino-dir /cluster/home/ielminawi/projects/dinov2 \
  --output-dir notes/temporal_aggregator_features_v1 \
  --batch-size 4

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.probe_temporal_aggregator fit \
  --features-dir notes/temporal_aggregator_features_v1 \
  --probe-dir notes/temporal_probe_results_2500_v1 \
  --output-dir notes/temporal_aggregator_probe_v1
```

CPU fitting (same allocation works): full512D regularized affine ridge probes, noPCA/whitening, shared endpoint7 target normalization from previous training split. Sixalpha grid .01,.1,1,10,100,1000 selected by validationMSE, sum-SSE penalty with unpenalizedintercept. Report continuous and quantized all7/translation/orientation/gripper plus mean-action reference. Save feature hashes, modelweights, normalized predictions and results. Compare trainedminuszero within each inputlength, and fouraggregatedminustwoaggregated; positive paired differences mean worse. Episode-bootstrap intervals use equal-episode means, fixed globalvariance; not adjustedforvalidationselection orLAM-seeduncertainty. Zero four-frame is the released encoder with four inputs, not its originaltwoframe behavior. No nonlinear or path probes here; diagnostic startswith matched ridge. Validationreuse remainsexploratory.

```bash
python -m unittest discover -s tests -p 'test_aggregator_probe.py'
```
