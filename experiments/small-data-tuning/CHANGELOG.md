# Changelog — small-data-tuning

Records changes to scripts and infrastructure in this experiment directory.
Each entry identifies the agent that made the change and what it does.

## 2026-03-27 — claude-code

### Extended OpenGroundingDINO benchmark runner for hyperparameter tuning

**What changed:**

- `run_opengroundingdino_dino_detector_benchmark.sh`: Extended the
  `GDINO_CONFIG_SPECS` format with three optional trailing fields:
  `batch_size`, `lr_scale`, and `backbone_lr_scale`. When `batch_size` is
  provided, learning rates are scaled linearly from their base values
  (lr=0.0001, lr_backbone=1e-5, both at batch_size=4) then multiplied by the
  respective scale factors. The overrides are appended to the copied Python
  config template. The run manifest now records the resolved hyperparameters.
  Existing 5-field specs are fully backward compatible.

- `README.md`: Added "Extending OpenGroundingDINO tuning" section documenting
  the new config spec format and example usage.

**Why:** The DINOv2/OpenGroundingDINO baseline has only been run with its
default hyperparameters (batch_size=4, lr=0.0001). To fairly compare against
DEIMv2 — which has been extensively tuned — we need to test whether DINOv2 is
similarly sensitive to batch size and learning rate. This also provides a
control: if DINOv2 is robust to batch size changes, it strengthens the
conclusion that the DINOv3 gap is not purely a tuning artifact.

**Backward compatibility:** The default `GDINO_CONFIG_SPECS` remains a single
5-field baseline spec. The analyzer already reads `train_batch_size` from run
manifests and falls back to parsing the Python config, so no analyzer changes
were needed.

## 2026-03-27 (later) — claude-code

### Added timing estimation and ProcessContext telemetry

**What changed:**

- `analyze_dino_detector_benchmark.py`: Added `_estimate_training_duration()`
  function that reads timing from `kwutil.ProcessContext` telemetry files
  (direct measurement) or falls back to checkpoint file modification timestamp
  deltas (estimated). New columns `train_duration_minutes` and `timing_method`
  in the benchmark summary TSV and results table image.

- `run_opengroundingdino_dino_detector_benchmark.sh`: Training is now wrapped
  in a `kwutil.ProcessContext` via a Python shim. Writes
  `initial_telemetry.json` (machine info, config, start time) and
  `final_telemetry.json` (adds duration, stop time) to the train output
  directory. Falls back to plain `bash train_dist.sh` if kwutil is unavailable.

**Why:** Training duration is useful for cost estimation and comparing configs.
ProcessContext also captures machine info (hostname, GPU, CPU, memory) which
aids reproducibility. The pattern matches what `shitspotter/detectron2/fit.py`
already does.

**Backward compatibility:** New TSV columns are additive. The ProcessContext
wrapper degrades gracefully. Existing runs without telemetry files still get
timing estimated from checkpoint timestamps.

## 2026-03-28 (later) — claude-code

### Added epochs field to config spec; fixed hardcoded final-checkpoint check

**What changed:**

- `run_opengroundingdino_dino_detector_benchmark.sh`: Added `epochs` as a 9th
  optional field in `GDINO_CONFIG_SPECS`. When set, appends `epochs = N` to
  the Python config override block. The training-completion check now computes
  the expected final checkpoint dynamically (`checkpoint{epochs-1:04d}.pth`)
  instead of the previous hardcoded `checkpoint0014.pth`. Added epoch display
  in per-run stdout. Run manifest records `epochs` when explicitly set.

**Why:** batch12 experiments OOM'd (needed ~20.7 GiB, GPU had ~19 GiB free).
batch8 is the practical ceiling. Pivoting to epoch sweep: testing whether the
two best LR recipes (lr_scale=1.25 and lr_scale=1.5) improve further with 25
epochs vs the default 15. The hardcoded checkpoint0014.pth was also a latent
bug — any non-default epoch count would have caused silent re-training.

**Backward compatibility:** `config_epochs` defaults to empty (uses config
template default of 15 epochs). The dynamic checkpoint check produces
`checkpoint0014.pth` when no override is set, preserving existing behavior.

## 2026-03-29 — claude-code

### Fixed corrupt annotation crash in prepare script for full-data run

**What changed:**

- `prepare_dino_detector_benchmark.py`: `_poop_only_subset()` now detects and
  drops annotations whose `category_id` is absent from the category table
  (orphaned/corrupt annotations). This was exposed when preparing the full
  5747-image train split: annotation `id=4268, image_id=6455` had
  `category_id=2` which does not exist in the source dataset. The smaller
  subsets (128/256/512) never sampled that image, so the bug was invisible
  until now. A `warnings.warn` is emitted so the issue is visible in logs.

**Why:** `remove_categories` only removes categories that are registered; it
cannot clean up annotations pointing to non-existent IDs. `simplify_kwcoco`
then crashes when it tries to look up the category name.

### Log-scale x-axis for train-size plots; full-data support in peek script

**What changed:**

- `analyze_dino_detector_benchmark.py`: Plot x-axis now uses log scale
  automatically when the ratio of max to min train size exceeds 10×. This
  keeps 128/256/512 readable alongside 5747. Tick marks are set to the actual
  train sizes present in the data. X-axis label updates to note "(log scale)".

- `peek_dino_detector_benchmark_progress.sh`: `FOCUS_TRAIN_SIZES` default
  extended from `"128 256"` to `"128 256 512 5747"` so the full-data run
  appears in the filtered summary table. The Python filter also now shows all
  rows when `focus_sizes` is empty (pass `FOCUS_TRAIN_SIZES=""` to see
  everything).

**Why:** Adding a train5747 split compresses 128/256/512 into the far left on
a linear axis, making the small-data sweep unreadable. Log scale gives each
order of magnitude equal visual weight.
