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
