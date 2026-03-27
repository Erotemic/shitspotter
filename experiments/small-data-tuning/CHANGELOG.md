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
