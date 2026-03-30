## 2026-03-27 19:00:00 +0000

Summary of user intent: get up to speed with the ShitSpotter project and its
current research state, then extend the small-data-tuning benchmark to support
OpenGroundingDINO (DINOv2) hyperparameter tuning — specifically batch size and
learning rate sweeps — so we can test whether DINOv2 is as sensitive to these
knobs as DEIMv2 (DINOv3) has been.

### Context and motivation

The small-data benchmark has been the primary comparison harness for DINOv2 vs
DINOv3 detectors. DEIMv2 (DINOv3 backbone) has been extensively tuned through
multiple config-tag sweeps, going from batch-24 down to batch-4, with various
LR scale combinations. The finding is that batch size is a major lever for
DEIMv2, but even at batch-4, DEIMv2 still trails OpenGroundingDINO (DINOv2) by
about 0.10 AP on test.

The critical gap in the experimental design: OpenGroundingDINO has only ever
been run with its default config (batch_size=4, lr=0.0001, lr_backbone=1e-5,
15 epochs). We have no evidence about whether DINOv2 is robust or sensitive to
the same hyperparameter changes. If DINOv2 degrades with larger batches
(like DINOv3 does), that's interesting. If DINOv2 is stable, it strengthens the
conclusion that the DINOv3 gap is architectural/recipe-level rather than purely
a tuning artifact. Either way, it makes the comparison more honest.

### What I did

Extended `run_opengroundingdino_dino_detector_benchmark.sh` to accept three
optional fields in `GDINO_CONFIG_SPECS`: `batch_size`, `lr_scale`, and
`backbone_lr_scale`. The design mirrors the DEIMv2 approach:

- LR scales linearly with batch size from known base values
- Scale factors are additional multipliers on top of that
- Config overrides are appended to the copied Python config template
  (OpenGroundingDINO configs are Python files where later definitions override
  earlier ones)
- The run manifest now records all resolved hyperparameters
- Existing 5-field specs work unchanged (backward compatible)

Also updated the README and created a CHANGELOG.md in the experiment directory
so other agents (including Codex) know what changed and why.

### Design decisions and tradeoffs

1. **Appending overrides to the Python config** rather than templating a new
   config from scratch. This is the simplest approach and matches how the
   existing script already adds `use_coco_eval = False` and `label_list`.
   Risk: if OpenGroundingDINO's config loading changes to not support
   re-definition, this would break. But that's unlikely for a Python config.

2. **Linear LR scaling with batch size by default**, matching the DEIMv2
   convention. The user can set lr_scale=0.5 to keep the absolute LR constant
   when doubling batch size, which is a useful control experiment.

3. **Keeping the default GDINO_CONFIG_SPECS as the single baseline** rather
   than adding new configs to the default. This ensures the existing codex
   agent's workflow is not disrupted. New configs are opt-in via env var.

4. **No analyzer changes needed** — the existing code already reads
   `train_batch_size` from manifests and has fallback parsing of the Python
   config.

### What I verified

- Read the actual deployed OpenGroundingDINO config from a previous baseline
  run on disk at `/data/joncrall/dvc-repos/shitspotter_expt_dvc/...` to
  confirm the base hyperparameter values.
- Confirmed the analyzer handles both old (4-level) and new (5-level) path
  layouts and reads batch size from manifests.
- Verified OpenGroundingDINO has a BERT text encoder (language grounding) while
  DEIMv2 does not — this is a meaningful architectural difference that may
  partly explain the performance gap, especially on small data where language
  priors help.

### What's uncertain

- Whether the OpenGroundingDINO repo is available on the user's training
  machine. The script changes are ready but can't be tested without it.
- Whether OpenGroundingDINO's training loop handles batch_size changes
  gracefully (e.g., does it need gradient accumulation for large batches, or
  will it OOM?).
- The right set of configs to actually run. Suggested starting point:
  baseline (batch 4), batch 8, batch 16, and a batch-8-no-lr-scale control.

### Reusable takeaways

- When one arm of a comparison has been heavily tuned and the other hasn't,
  the gap measurement is confounded. Tuning the baseline is as important as
  tuning the challenger.
- Config-tag systems pay for themselves quickly once you need more than one
  recipe per model family.
- Backward compatibility in config spec formats (optional trailing fields with
  defaults) avoids breaking existing workflows while enabling new experiments.


## 2026-03-27 — Overnight DINOv2 tuning results and timing infrastructure

Summary of user intent: analyze completed overnight DINOv2 hyperparameter sweep
results (8 configs × 3 train sizes = 24 runs), add training duration estimation
to the analysis pipeline, and integrate `kwutil.ProcessContext` for richer
reproducibility metadata in future training runs.

### Key findings from DINOv2 hyperparameter sweep

Ran 8 OpenGroundingDINO configs across train sizes 128/256/512:
- **baseline** (batch4, lr=0.0001) — the original untuned config
- **batch8** (batch8, lr=0.0002) — linear LR scaling
- **batch8_no_lr_scale** (batch8, lr=0.0001) — batch increase without LR change
- **batch4_lr2x** (batch4, lr=0.0002) — LR increase without batch change
- **batch8_lr0p75/1p25/1p5/2x** — various LR scales at batch 8

Results confirm DINOv2 benefits from tuning:
1. **Higher LR consistently helps.** At train128, batch8_lr1p25 (vali=0.582)
   beats baseline (vali=0.544) by +0.038 AP. The effect holds across sizes.
2. **Batch size helps independently of LR.** batch8 (lr=0.0002) vs
   batch4_lr2x (same lr=0.0002) shows ~+0.01 vali / +0.02 test at train128,
   confirming batch size has an effect beyond its LR scaling.
3. **DINOv2 is robust to batch changes** — unlike DEIMv2 which degrades with
   larger batches, DINOv2 benefits from batch8. No sign of degradation.
4. **Best LR varies by train size.** At train128, lr_scale=1.25 wins; at
   train512, lr_scale=0.75 wins on vali. The optimal LR shifts down as data
   increases — expected behavior.
5. **DINOv2–DINOv3 gap widens after tuning.** Best tuned DINOv2 at train128
   is 0.582 vali / 0.662 test vs best DEIMv2 0.453 / 0.502 — gap is ~0.13-0.16 AP.

### Timing infrastructure

Added two pieces:

1. **`_estimate_training_duration()` in `analyze_dino_detector_benchmark.py`:**
   Checks for `kwutil.ProcessContext` telemetry files first (direct measurement),
   falls back to checkpoint file modification timestamp deltas. The `timing_method`
   column in the output records which approach was used so downstream analysis
   can account for accuracy differences. Methods: `process_context` (direct),
   `checkpoint_mtime_delta` (estimated, excludes data prep overhead).

2. **ProcessContext wrapper in `run_opengroundingdino_dino_detector_benchmark.sh`:**
   Training is now wrapped in a Python shim that creates a `kwutil.ProcessContext`,
   records the run manifest as config, writes `initial_telemetry.json` before
   training and `final_telemetry.json` after. Falls back gracefully to plain
   `bash train_dist.sh` if kwutil is unavailable. This matches the pattern
   already used in `shitspotter/detectron2/fit.py`.

### Design decisions

- **Checkpoint mtime fallback**: For existing runs without ProcessContext data,
  the delta between first and last checkpoint file timestamps is a reasonable
  proxy. It underestimates total wall time (excludes data prep) but captures
  the training loop duration, which is the expensive part.
- **Graceful degradation**: The ProcessContext wrapper uses `|| true` and a
  fallback `bash train_dist.sh` to ensure training proceeds even if kwutil
  is missing or the wrapper has issues.
- **Both columns in TSV**: `train_duration_minutes` and `timing_method` are
  added as new columns, not replacing anything. This preserves backward
  compatibility for any downstream consumers of the TSV.


## 2026-03-28 — batch12 OOM, epoch sweep added

### What happened

Proposed and ran batch12 configs (batch12_lr1p25, batch12_lr1p5) as the next
experiment, but both OOM'd during the backward pass:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.11 GiB.
GPU 0 has 23.66 GiB total, 905.94 MiB free.
Including non-PyTorch memory, process has 18.60 GiB in use.
```

Key memory facts:
- batch8 peak: ~14.1 GiB (safe on either GPU)
- batch12 peak: >18.6 GiB + attempted 2.11 GiB = ~20.7 GiB needed
- GPU 1 (24 GiB) had ~5 GiB occupied by another process, leaving ~19 GiB
- batch12 needs more than that → OOM during backward

**Conclusion: batch8 is the practical ceiling** for the 3090/4090 setup with
this model. No gradient accumulation support in OpenGroundingDINO's training
loop, so effective batch size can't be increased without code changes.

There was also a `FileNotFoundError` in the first run attempt (different log):
the pretrained weights file at `models/groundingdino_swint_ogc.pth` was
missing. The user re-downloaded it between attempts.

### Pivot to epoch sweep

With batch size maxed at 8, the natural next dimension is training duration.
The overnight results showed best checkpoints at ckpt0002-0007 out of 15,
suggesting the model converges early at high LR — but that doesn't mean more
epochs can't help at moderate LR. Proposed: run the two best LR recipes
(lr_scale=1.25, lr_scale=1.5) with 25 epochs instead of 15.

### What I implemented

Added `epochs` as a 9th optional field in `GDINO_CONFIG_SPECS`:

```
config_tag|gpu_num|pretrain|text_encoder|cfg_template[|batch_size|lr_scale|backbone_lr_scale|epochs]
```

Changes to `run_opengroundingdino_dino_detector_benchmark.sh`:
1. Parse `config_epochs` from field 9; append `epochs = N` to Python config
   override block when set
2. Dynamically compute expected final checkpoint:
   `checkpoint{epochs-1:04d}.pth` (was hardcoded to `checkpoint0014.pth`)
3. Added epoch display in the per-run print block

The hardcoded `checkpoint0014.pth` was a latent bug — any epoch sweep would
have silently re-triggered training on every run because the completion check
always looked for the 15-epoch checkpoint.

### Command for next run

```bash
PTH=/data/joncrall/dvc-repos/shitspotter_expt_dvc/models/groundingdino_swint_ogc.pth
GDINO_TRAIN_SIZES="128" \
GDINO_CONFIG_SPECS="batch8_lr1p25_ep25|1|$PTH|bert-base-uncased|config/cfg_odvg.py|8|1.25|1.0|25 batch8_lr1p5_ep25|1|$PTH|bert-base-uncased|config/cfg_odvg.py|8|1.5|1.0|25" \
bash experiments/small-data-tuning/run_opengroundingdino_dino_detector_benchmark.sh
```

Expected: ~27 min per config at train128 (scaling from 16 min × 25/15).

### Reusable takeaways

- **batch8 is the VRAM ceiling** for OpenGroundingDINO on these GPUs. Any
  future experiment with this model should not exceed batch8 unless gradient
  accumulation is added to the training loop.
- **Hardcoded final-checkpoint filenames are fragile** when epochs vary. Always
  derive the expected final checkpoint from the epoch count.
- **Check GPU occupancy before scheduling runs.** The OOM was partly caused by
  another process holding ~5 GiB on the 24 GiB GPU. Running `nvidia-smi` first
  would catch this.


## 2026-03-28 — ep25 results: more epochs hurt; DINOv2 tuning complete

### 25-epoch experiment results

The ep25 configs (batch8, lr_scale=1.25 and 1.5, 25 epochs) both underperformed
their 15-epoch counterparts at train128:

| config            | epochs | vali_ap | test_ap | best_ckpt |
|-------------------|-------:|--------:|--------:|-----------|
| batch8_lr1p25     |     15 |  0.5824 |  0.6336 | ckpt0002  |
| batch8_lr1p25_ep25|     25 |  0.5560 |  0.6108 | ckpt0005  |
| batch8_lr1p5      |     15 |  0.5749 |  0.6377 | ckpt0005  |
| batch8_lr1p5_ep25 |     25 |  0.5481 |  0.5838 | ckpt0002  |

**More epochs hurt.** -0.026 vali on lr1p25, -0.027 vali on lr1p5. Both ep25
best checkpoints are still very early (ckpt0002–0005), confirming the model
finds its optimum quickly and then overfits. The LR schedule is cosine decay
over N epochs, so a 25-epoch run spends more time at low LR past the good
checkpoint, apparently drifting.

### DINOv2 hyperparameter sweep — summary of findings

Explored: batch size (4, 8), LR scale (0.75, 1.0, 1.25, 1.5, 2.0 × linear
batch scaling), epochs (15, 25). All evaluated at train128.

**Best config: batch8, lr_scale=1.25, 15 epochs** (vali=0.582, test=0.634)

Tuning axes exhausted:
- Larger batch (12+): OOM — batch8 is the hardware ceiling
- Higher LR (2x): Better test but worse vali — noisy, not clearly better
- More epochs (25): Overfits — worse on both metrics
- Fewer epochs: Not tested; best ckpt always at ep2–5, suggesting 10 epochs
  might suffice, but diminishing returns for the paper

### DINOv2 vs DEIMv2 gap — final assessment

After full tuning on both sides:

| model   | train_size | vali_ap | test_ap |
|---------|-----------|--------:|--------:|
| DINOv2  | 128       |   0.582 |   0.634 |
| DEIMv2  | 128       |   0.453 |   0.502 |
| gap     |           |  ~0.129 |  ~0.132 |

The ~0.13 AP gap is robust: it persists across both vali and test, and was
measured with both sides tuned. The gap is not a tuning artifact.

### Full-data DINOv2 result (train5747)

**batch8_lr1p25 at train5747: vali=0.7118, test=0.7224**

This is the headline result. Scaling from 512 to 5747 images (+0.026 vali /
+0.037 test) continues the scaling trend cleanly:

| train_size | vali_ap | test_ap | delta_vali | delta_test |
|----------:|--------:|--------:|-----------:|-----------:|
|        128 |  0.5824 |  0.6336 |            |            |
|        256 |  0.5970 |  0.6199 |     +0.015 |     -0.014 |
|        512 |  0.6454 |  0.6848 |     +0.048 |     +0.065 |
|       5747 |  0.7118 |  0.7224 |     +0.066 |     +0.038 |

The model clearly benefits from more data — the scaling curve is still going
up with no sign of saturation.

**vs. current best published model (v5):** v5 was a combined pipeline
(DINOv2 detector + SAM2) with combined test AP=0.550 and detector vali=0.572.
The new DINOv2-only detector scores test=0.722 — +0.172 AP above v5's
detector component, and well above v5's combined score. This is a strong
improvement even before considering SAM2 integration.

**vs. DEIMv2 best at train512:** 0.647 vali / 0.671 test. DINOv2 at full
data is +0.065 vali / +0.051 test above DEIMv2's best small-data result.
DEIMv2 was not run at full data, so direct apples-to-apples isn't available,
but the gap at train512 was already ~0.10 AP.

**Best checkpoint: checkpoint0006** (out of 14), consistent with earlier
finding that the model converges early. The 5747-image dataset provides enough
signal that the model reaches its optimum in ~6/15 epochs.

### Next step: scale best DINOv2 to full training set

The small-data benchmark was a proxy to iterate quickly. Now that the best
config is identified (batch8_lr1p25, 15 epochs), the next experiment is to
train on the **full training dataset** to see whether DINOv2 can beat the
current best model.

Current best model: **v5** (foundation-detseg pipeline = DINOv2 detector +
SAM2 segmentation, full training data)
- combined vali AP: 0.600
- combined test AP: 0.550

Note: the v5 combined AP includes the SAM2 segmentation stage. The DINOv2
detector alone at train512 (512 images) already hits test AP ~0.685 on the
detection-only metric. These aren't directly comparable (detection AP vs
combined segmentation AP), but the detector component of v5 scored vali=0.572,
test=0.526 — so DINOv2 at train512 already beats that on raw detection. Full
training data should push further still.

The full train set is `train_imgs5747_1e73d54f` (5747 images). This is ~11×
more data than train512. The prepare script supports `TRAIN_SIZES` as a bash
array, so a full-data run requires adding a `5747` (or `all`) entry or pointing
directly at the full dataset.

Command to run (after adding full-data support to prepare, or running directly):

```bash
cd /home/joncrall/code/shitspotter
PTH=/data/joncrall/dvc-repos/shitspotter_expt_dvc/models/groundingdino_swint_ogc.pth

# Option A: extend the benchmark to include the full training size
TRAIN_SIZES="128 256 512 5747" \
bash experiments/small-data-tuning/prepare_dino_detector_benchmark.sh

# Then train only the full-data config with best recipe
GDINO_TRAIN_SIZES="5747" \
GDINO_CONFIG_SPECS="batch8_lr1p25|1|$PTH|bert-base-uncased|config/cfg_odvg.py|8|1.25|1.0" \
bash experiments/small-data-tuning/run_opengroundingdino_dino_detector_benchmark.sh
```

Expected training time: ~16 min × (5747/512) × (1/1) ≈ ~3 hours at batch8
(linear scaling with data size). This fits in an overnight run.


## 2026-03-28 — OpenGroundingDINO inference adapter in shitspotter package

### Summary

Added an `OpenGroundingDINOPredictor` to `shitspotter/algo_foundation_v3/` so
the trained DINOv2 detector can be combined with SAM2 segmentation in the
production inference pipeline.  This extends the existing `deimv2_sam2` and
`maskdino` backends with a new `opengroundingdino_sam2` backend.

### Files changed

- **`shitspotter/algo_foundation_v3/detector_opengroundingdino.py`** (new):
  `OpenGroundingDINOPredictor` class that adapts the tpl/Open-GroundingDino
  repo to the standard `predict_image_records(image)` interface returning
  `[{'label': str, 'bbox_ltrb': [x1,y1,x2,y2], 'score': float}]`.

  Key implementation details:
  - `_build_inference_args()`: loads the SLConfig config from disk and merges
    it into a `types.SimpleNamespace`.  Sets `use_coco_eval=False`,
    `label_list=['poop']`, `distributed=False`, `amp=False`.  The PostProcess
    class in the tpl repo reads `args.label_list` at `__init__` time to build
    the positive token-to-label map.
  - `_lazy_init()`: adds the tpl repo to sys.path, calls `build_model_main(args)`
    to get (model, criterion, postprocessors), loads checkpoint via
    `clean_state_dict(ckpt['model'])`.
  - Inference: PIL image → resize (shorter side 800, max 1333, matching val
    transforms) → ToTensor → Normalize(imagenet) → create NestedTensor with
    zero mask → `model(nested, captions=[caption])` → `postprocessors['bbox']`
    → filter by score_thresh → return records.
  - Caption format: `' . '.join(label_list) + ' .'` — this is the exact
    format used in the tpl training and evaluation code.

- **`shitspotter/algo_foundation_v3/model_registry.py`**:
  - Added `'opengroundingdino'` to `REPO_ENV_VARS`
    (`SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH`)
  - Added `'opengroundingdino_shitspotter'` preset to `DETECTOR_PRESETS`
    (note: `config_fpath` and `checkpoint_fpath` must be set per-package since
    they vary across training runs)
  - Added `'opengroundingdino_sam2'` backend to `BACKEND_DEFAULTS`

- **`shitspotter/algo_foundation_v3/packaging.py`**:
  - Added validation clause for `opengroundingdino_sam2` backend (requires
    both `detector` and `segmenter` sections)

- **`shitspotter/algo_foundation_v3/cli_predict.py`**:
  - Imported `OpenGroundingDINOPredictor`
  - Added `elif resolved_package['backend'] == 'opengroundingdino_sam2':` branch
    (same pattern as `deimv2_sam2`)
  - Added device override handling for the new backend

### To use this pipeline

Create a package YAML file (e.g., `gdino_sam2_v1.yaml`) with:

```yaml
backend: opengroundingdino_sam2
detector:
  preset: opengroundingdino_shitspotter
  repo_dpath: /path/to/tpl/Open-GroundingDino
  config_fpath: /path/to/training/output/config_cfg.py
  checkpoint_fpath: /path/to/training/output/checkpoint0006.pth
  label_list: [poop]
segmenter:
  preset: sam2.1_hiera_base_plus
  repo_dpath: /path/to/sam2
  checkpoint_fpath: /path/to/sam2_checkpoint.pth
```

Then run:

```bash
python -m shitspotter.algo_foundation_v3.cli_predict \
    --src /path/to/test.kwcoco.zip \
    --dst /path/to/pred.kwcoco.zip \
    --model gdino_sam2_v1.yaml
```

### Design decisions

1. **No separate config templating needed.** The training run already writes
   a `config_cfg.py` to the output directory.  That file is already merged
   with all the training overrides (including our `use_coco_eval = False` and
   `label_list`), so it's the cleanest source of truth for inference.  If that
   file is used as `config_fpath`, the PostProcess will be built with the
   correct label list.

2. **label_list override in detector_cfg.** The `label_list` field in the
   detector section of the package overrides the one in the config file.  This
   makes it easy to adapt a checkpoint to a different label vocabulary without
   rewriting the config file.

3. **Zero-mask NestedTensor.** For single-image inference without batching /
   padding, the mask is all zeros (no masked pixels).  This matches what the
   model expects for a fully-padded image.

4. **score_thresh default 0.0.** Let `postprocess.DEFAULT_POSTPROCESS` handle
   filtering.  The predictor applies a lower bound only to avoid passing
   thousands of near-zero boxes to SAM2.

### What's not yet done

- No eval integration in the benchmark runner (uses a separate eval path).
- No training adapter in `detector_opengroundingdino.py` (training uses the
  existing `run_opengroundingdino_dino_detector_benchmark.sh` bash pipeline).
- The `opengroundingdino_shitspotter` preset intentionally omits `config_fpath`
  and `checkpoint_fpath` — these must be set in the package YAML.  We could
  add a default checkpoint path once we have a stable trained model.


## 2026-03-30 — Segmentation eval sweep: gdino + SAM2 candidates

### Summary

Wrote infrastructure to evaluate OpenGroundingDINO + SAM2 combos end-to-end
(detector → SAM2 segmentation → kwcoco eval → pixel AP).  Also extended
`build_package` / `cli_package build` to accept `detector_config_fpath`.

### Files changed

- **`shitspotter/algo_foundation_v3/packaging.py`** — added `detector_config_fpath`
  parameter to `build_package()`.  OpenGroundingDINO needs both a checkpoint
  and a SLConfig file; the deimv2/maskdino backends ignore this field.

- **`shitspotter/algo_foundation_v3/cli_package.py`** — exposed
  `--detector_config_fpath` in `cli_package build`.

- **`experiments/foundation_detseg_v3/common.sh`** — added
  `SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH` export (defaults to
  `tpl/Open-GroundingDino`).

- **`experiments/foundation_detseg_v3/packages/opengroundingdino_sam2_default.yaml`**
  (new) — template package with `REPLACE_ME_*` placeholders for all three
  paths (`GDINO_CONFIG`, `GDINO_CHECKPOINT`, `SAM2_CHECKPOINT`).

- **`experiments/foundation_detseg_v3/run_gdino_sam2_sweep.sh`** (new) — the
  main experiment script.  Reads `GDINO_CKPT` and `GDINO_CFG` from the
  environment, builds packages, runs `cli_predict` + `kwcoco eval` on vali and
  test for three candidates:
  - `gdino_zeroshot` — best DINOv2 (train5747) + pretrained SAM2 (no tuning)
  - `gdino_tuned`    — best DINOv2 (train5747) + v5's fine-tuned SAM2
  - `v5_baseline`    — v5 DEIMv2 detector + v5's fine-tuned SAM2

  Results are cached (skips re-running if `detect_metrics.json` exists) and
  printed in a summary table at the end.

### How to run

```bash
# 1. Find the best gdino checkpoint (batch8_lr1p25, train5747, ckpt0006)
EXPT=$DVC_EXPT_DPATH
GDINO_RUN=$(find $EXPT/small_data_tuning/dino_detector_benchmark \
    -name 'checkpoint0006.pth' -path '*/batch8_lr1p25/*' -path '*/train5747*' \
    | head -1 | xargs dirname)

GDINO_CKPT="$GDINO_RUN/checkpoint0006.pth"
GDINO_CFG="$GDINO_RUN/config_cfg.py"

# 2. Run the sweep
GDINO_CKPT="$GDINO_CKPT" \
GDINO_CFG="$GDINO_CFG" \
bash experiments/foundation_detseg_v3/run_gdino_sam2_sweep.sh
```

### What the results will tell us

- **gdino_zeroshot vs gdino_tuned**: how much does SAM2 fine-tuning help with
  DINOv2's detection boxes (which are more accurate than DEIMv2's)?

- **gdino_tuned vs v5_baseline**: how much does switching the detector from
  DEIMv2 to DINOv2 improve the combined segmentation AP?  The detector
  component already scores +0.172 AP on detection — but SAM2's segmentation
  quality depends heavily on box accuracy, so the gain may compound.

- **gdino_zeroshot vs v5_baseline**: whether zero-shot SAM2 + better detector
  beats tuned SAM2 + weaker detector — i.e., is detector quality or SAM2
  fine-tuning the bigger lever.

### Expected outcome

Based on the prior finding that SAM2 achieves ~1.0 AP with oracle boxes, the
DINOv2 detector's better box quality should translate directly to better
combined AP.  My expectation:
  gdino_tuned > gdino_zeroshot > v5_baseline

But the zero-shot vs tuned SAM2 gap is uncertain — the tuned SAM2 was trained
on boxes from the weaker DEIMv2 detector, so it may not transfer perfectly to
DINOv2's box distribution.  Re-tuning SAM2 on DINOv2 detections would be the
natural follow-up if tuned underperforms zero-shot.
