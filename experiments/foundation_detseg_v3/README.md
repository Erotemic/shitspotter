# Foundation Det+Seg V3

This experiment family adds a new detector-plus-segmenter bootstrap path for
ShitSpotter polygon annotation work.

Primary backend:

- `deimv2_sam2` = DEIMv2-M detector + SAM2.1 box-prompted refinement

Baseline backend:

- `maskdino` = MaskDINO-R50 direct instance segmentation

All scripts in this directory are cwd-independent. They resolve the repo root
relative to the script location, so you can run them from anywhere. They also
fall back to `$HOME/code/shitspotter/experiments/foundation_detseg_v3` when
their contents are pasted directly into an interactive bash shell and
`BASH_SOURCE` does not point at a real file.

## End-to-end quick start

These commands follow the same path conventions used in the older ShitSpotter
experiment scripts:

- data bundles come from `$(geowatch_dvc --tags="shitspotter_data")`
- experiment outputs go under `$(geowatch_dvc --tags="shitspotter_expt")`
- training runs land under `.../training/$HOSTNAME/$USER/ShitSpotter/runs/...`
- reusable downloaded model assets live under `$DVC_DATA_DPATH/models/...`
- the default comparison splits are the hashed files
  `train_imgs5747_1e73d54f.kwcoco.zip`,
  `vali_imgs691_99b22ad0.kwcoco.zip`, and
  `test_imgs121_6cb3b6ff.kwcoco.zip`

### 1. Set up your shell environment

These defaults match the usual local dev layout with the repo checked out at
`$HOME/code/shitspotter` and the extra model repos checked out under `tpl/`.

```bash
export SHITSPOTTER_DPATH="${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}"
export DVC_DATA_DPATH="${DVC_DATA_DPATH:-$(geowatch_dvc --tags="shitspotter_data")}"
export DVC_EXPT_DPATH="${DVC_EXPT_DPATH:-$(geowatch_dvc --tags="shitspotter_expt")}"

export SHITSPOTTER_DEIMV2_REPO_DPATH="${SHITSPOTTER_DEIMV2_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/DEIMv2}"
export SHITSPOTTER_SAM2_REPO_DPATH="${SHITSPOTTER_SAM2_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/segment-anything-2}"
export SHITSPOTTER_MASKDINO_REPO_DPATH="${SHITSPOTTER_MASKDINO_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/MaskDINO}"
export FOUNDATION_V3_MODEL_DPATH="${FOUNDATION_V3_MODEL_DPATH:-$DVC_DATA_DPATH/models/foundation_detseg_v3}"
export FOUNDATION_V3_TRAIN_KWCOCO_FPATH="${FOUNDATION_V3_TRAIN_KWCOCO_FPATH:-$DVC_DATA_DPATH/train_imgs5747_1e73d54f.kwcoco.zip}"
export FOUNDATION_V3_VALI_KWCOCO_FPATH="${FOUNDATION_V3_VALI_KWCOCO_FPATH:-$DVC_DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip}"
export FOUNDATION_V3_TEST_KWCOCO_FPATH="${FOUNDATION_V3_TEST_KWCOCO_FPATH:-$DVC_DATA_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip}"
```

### 2. Initialize external repos and install Python deps

This uses the `tpl/` git submodules by default and installs the local Python
package plus the extra runtime/test utilities used by the foundation pipeline.
The setup script intentionally preserves your existing Torch stack and current
OpenCV provider instead of forcing the exact upstream DEIMv2 / MaskDINO pins.

```bash
git -C "$SHITSPOTTER_DPATH" submodule update --init --recursive --depth 1 \
    tpl/DEIMv2 \
    tpl/segment-anything-2 \
    tpl/MaskDINO

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/setup_environment.sh"
```

The current local deviations from upstream install instructions are tracked in
[UPSTREAM_ENVIRONMENT_OVERRIDES.md](/home/agent/code/shitspotter/experiments/foundation_detseg_v3/UPSTREAM_ENVIRONMENT_OVERRIDES.md).

### 3. Download the default foundational weights into the expected locations

This is the step that makes the rest of the README work without placeholders.
It downloads:

- DEIMv2 distilled backbone init files into `$SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts/`
- DEIMv2 pretrained S/M detector checkpoints into `$FOUNDATION_V3_MODEL_DPATH/deimv2/`
- SAM2.1 Base+ and Large checkpoints into `$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/`

```bash
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/download_foundation_assets.sh"
```

After that command, the important files should exist at:

```bash
ls "$SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts/vitt_distill.pt"
ls "$SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts/vittplus_distill.pt"
ls "$FOUNDATION_V3_MODEL_DPATH/deimv2/deimv2_dinov3_s_coco.pth"
ls "$FOUNDATION_V3_MODEL_DPATH/deimv2/deimv2_dinov3_m_coco.pth"
ls "$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt"
ls "$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_large.pt"
```

### 4. Run CPU smoke tests

```bash
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/cpu_smoke_tests.sh"
```

## Fine-tune the DEIMv2 detector on ShitSpotter

Training does not require a package file. It reads kwcoco, exports deterministic
COCO json under the workdir, generates the upstream DEIMv2 config, and runs the
upstream training code.

### 5. Fine-tune from the downloaded DEIMv2-M pretrained detector checkpoint

```bash
export DEIMV2_INIT_CKPT="$FOUNDATION_V3_MODEL_DPATH/deimv2/deimv2_dinov3_m_coco.pth"
export WORKDIR="${WORKDIR:-$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/deimv2_m}"

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
```

That script defaults to:

- `TRAIN_FPATH=$FOUNDATION_V3_TRAIN_KWCOCO_FPATH`
- `VALI_FPATH=$FOUNDATION_V3_VALI_KWCOCO_FPATH`
- `VARIANT=deimv2_m`
- `WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/deimv2_m`
- `ENABLE_RESIZE_PREPROCESS=True`
- `RESIZE_MAX_DIM=640`
- `RESIZE_OUTPUT_EXT=.jpg`
- `TRAIN_BATCH_SIZE=24`
- `VAL_BATCH_SIZE=48`
- `USE_AMP=True`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

If you unset `DEIMV2_INIT_CKPT`, the upstream S/M config will still use the
downloaded distilled backbone init file from `tpl/DEIMv2/ckpts/`. That is still
pretrained initialization, just not a full detector checkpoint resume/tune.

With resize preprocessing enabled, the script keeps your input env vars pointed
at the original hashed kwcoco bundles, but it first materializes temporary
resized training and validation bundles under:

```bash
ls "$WORKDIR/preprocessed_kwcoco/"
```

The detector then trains against those preprocessed bundles instead of the
full-resolution phone images. This is meant to reduce the decode+resize
bottleneck without forcing you to manually manage a second set of dataset paths.

These defaults are intentional for the resized-data path on a single 24 GB GPU.
In the current ShitSpotter setup, `TRAIN_BATCH_SIZE=24` used about `20 / 24 GB`
and produced roughly `70-80%` GPU utilization, which was much healthier than
the earlier overly conservative settings. The upstream COCO configs still assume
larger aggregate batch sizes, so if you need a smaller or larger run, override
them directly before calling the script:

```bash
export TRAIN_BATCH_SIZE=16
export VAL_BATCH_SIZE=32
export USE_AMP=True
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
```

If you want to skip or rebuild the preprocessing step:

```bash
export ENABLE_RESIZE_PREPROCESS=False
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
```

```bash
export FORCE_RESIZE_PREPROCESS=True
export RESIZE_MAX_DIM=512
export TRAIN_BATCH_SIZE=24
export VAL_BATCH_SIZE=48
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
```

### 6. Evaluate the tuned detector with box AP

Detector-only box AP currently comes from the DEIMv2 validation loop itself.
The generated DEIMv2 config uses a `CocoEvaluator` with `iou_types=['bbox']`, so
the detector training run reports bbox AP on the validation split while it
trains. In other words, the current detector-only evaluation path is the native
DEIMv2 validation metric inside the training job, not a separate ShitSpotter
experiment wrapper script.

Useful files to inspect after training:

```bash
ls "$WORKDIR"
ls "$WORKDIR/summary"
ls "$WORKDIR/best_stg2.pth"
```

Use that detector-side validation signal to pick the checkpoint you want to
carry into the detector+segmenter evaluation path below.

### 7. Build a package for your trained detector

Replace `best_stg2.pth` with whichever checkpoint you want to deploy.

```bash
export DEIMV2_TRAINED_CKPT="$WORKDIR/best_stg2.pth"
export DEIMV2_SAM2_TRAINED_PACKAGE="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/deimv2_sam2_trained.yaml"

python -m shitspotter.algo_foundation_v3.cli_package build \
    "$DEIMV2_SAM2_TRAINED_PACKAGE" \
    --backend deimv2_sam2 \
    --detector_preset deimv2_m \
    --segmenter_preset sam2.1_hiera_base_plus \
    --detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT" \
    --segmenter_checkpoint_fpath "$SAM2_BPLUS_CKPT" \
    --metadata_name deimv2_sam2_trained
```

You can then reuse `run_deimv2_sam2_on_vali.sh`, `run_deimv2_sam2_on_test.sh`,
and `run_bootstrap_new_cohort.sh` by setting:

```bash
export PACKAGE_FPATH="$DEIMV2_SAM2_TRAINED_PACKAGE"
```

## Fine-tune SAM2 on ShitSpotter masks

The repo now has a real SAM2 fine-tuning path. It exports the kwcoco masks into
the static-image SA1B-style format that the upstream SAM2 trainer already
supports, generates a SAM2 Hydra config under the local SAM2 repo, and launches
the official trainer. The default path uses `sam2.1_hiera_base_plus` because
that is the upstream SAM2.1 training template currently available in the local
repo.

Like detector training, SAM2 fine-tuning does not require a package file to
start training. The package is only for inference after you have a tuned
checkpoint you want to deploy.

The SAM2 experiment script intentionally uses `SAM2_*` environment variable
names instead of generic names like `WORKDIR` and `TRAIN_BATCH_SIZE`. This
avoids accidental leakage from a previous detector-training shell session.

### 8. Fine-tune SAM2.1 Base+ from the downloaded checkpoint

```bash
export SAM2_INIT_CKPT="$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt"
export SAM2_WORKDIR="${SAM2_WORKDIR:-$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/sam2.1_hiera_base_plus}"

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_sam2_segmenter.sh"
```

That script defaults to:

- `TRAIN_FPATH=$FOUNDATION_V3_TRAIN_KWCOCO_FPATH`
- `VALI_FPATH=$FOUNDATION_V3_VALI_KWCOCO_FPATH`
- `SAM2_VARIANT=sam2.1_hiera_base_plus`
- `SAM2_INIT_CKPT=$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt`
- `SAM2_WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/sam2.1_hiera_base_plus`
- `SAM2_RESOLUTION=1024`
- `SAM2_TRAIN_BATCH_SIZE=1`
- `SAM2_NUM_TRAIN_WORKERS=8`
- `SAM2_NUM_EPOCHS=20`
- `SAM2_NUM_GPUS=1`
- `SAM2_BASE_LR=5e-6`
- `SAM2_VISION_LR=3e-6`

The generated SAM2 bundle lives under:

```bash
ls "$SAM2_WORKDIR/prepared_data/sam2/train/images"
ls "$SAM2_WORKDIR/prepared_data/sam2/train/annotations"
ls "$SAM2_WORKDIR/prepared_data/sam2/train/train.txt"
ls "$SAM2_WORKDIR/generated_configs/train_sam2.yaml"
```

The expected tuned checkpoint lands here:

```bash
ls "$SAM2_WORKDIR/checkpoints/checkpoint.pt"
```

If you already know which detector checkpoint you want to pair with this tuned
segmenter, you can have the training script emit a runnable combined package at
the end of fine-tuning:

```bash
export DEIMV2_TRAINED_CKPT="$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/deimv2_m/best_stg2.pth"
export SAM2_PACKAGE_OUT="$SAM2_WORKDIR/tuned_segmenter_package.yaml"

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_sam2_segmenter.sh"
```

Otherwise, just build the package later once you know the detector checkpoint.

If you want to keep only the target category when exporting masks into the SAM2
training bundle, make that explicit:

```bash
export SAM2_CATEGORY_NAMES='["poop"]'
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_sam2_segmenter.sh"
```

## Evaluate detector+segmenter packages with the repo metric tooling

This is the main ShitSpotter comparison path. These validation/test runs go
through the foundation prediction backend, write kwcoco predictions, and then
feed the repo's existing evaluation / aggregation tooling. This is the path to
use when you want the combined detector+segmenter system judged on box metrics
and mask/pixel metrics rather than just detector bbox AP inside the upstream
trainer.

### 9. Evaluate the tuned detector with zero-shot SAM2

First build a package that pairs your tuned detector with the downloaded
zero-shot SAM2 checkpoint:

```bash
export DEIMV2_TRAINED_CKPT="$WORKDIR/best_stg2.pth"
export SAM2_BPLUS_CKPT="$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt"
export DEIMV2_SAM2_TRAINED_PACKAGE="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/deimv2_sam2_trained.yaml"

python -m shitspotter.algo_foundation_v3.cli_package build \
    "$DEIMV2_SAM2_TRAINED_PACKAGE" \
    --backend deimv2_sam2 \
    --detector_preset deimv2_m \
    --segmenter_preset sam2.1_hiera_base_plus \
    --detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT" \
    --segmenter_checkpoint_fpath "$SAM2_BPLUS_CKPT" \
    --metadata_name deimv2_sam2_trained
```

Then run validation and test:

```bash
export PACKAGE_FPATH="$DEIMV2_SAM2_TRAINED_PACKAGE"
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_vali.sh"

export PACKAGE_FPATH="$DEIMV2_SAM2_TRAINED_PACKAGE"
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_test.sh"
```

And aggregate:

```bash
TARGET_DPATH="$DVC_EXPT_DPATH/_foundation_detseg_v3" \
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/aggregate_foundation_results.sh"
```

### 10. Compare zero-shot SAM2 against tuned SAM2 with the same detector

Zero-shot comparison package:

```bash
export PACKAGE_FPATH="$DEIMV2_SAM2_TRAINED_PACKAGE"
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_vali.sh"
```

Tuned-SAM package:

```bash
export WORKDIR_DETECTOR="${WORKDIR_DETECTOR:-$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/deimv2_m}"
export DEIMV2_TRAINED_CKPT="$WORKDIR_DETECTOR/best_stg2.pth"
export SAM2_TUNED_CKPT="$SAM2_WORKDIR/checkpoints/checkpoint.pt"
export DEIMV2_SAM2_TUNED_SEG_PACKAGE="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/deimv2_sam2_tuned_segmenter.yaml"

python -m shitspotter.algo_foundation_v3.cli_package build \
    "$DEIMV2_SAM2_TUNED_SEG_PACKAGE" \
    --backend deimv2_sam2 \
    --detector_preset deimv2_m \
    --segmenter_preset sam2.1_hiera_base_plus \
    --detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT" \
    --segmenter_checkpoint_fpath "$SAM2_TUNED_CKPT" \
    --metadata_name deimv2_sam2_tuned_segmenter

export PACKAGE_FPATH="$DEIMV2_SAM2_TUNED_SEG_PACKAGE"
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_vali.sh"
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_test.sh"
```

Then aggregate both result families with the usual repo tooling:

```bash
TARGET_DPATH="$DVC_EXPT_DPATH/_foundation_detseg_v3" \
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/aggregate_foundation_results.sh"
```

## Train and evaluate the MaskDINO baseline

The baseline path is intentionally parallel to the detector+segmenter path, but
it does not have a good off-the-shelf ShitSpotter-ready checkpoint in this repo.
The normal flow is train first, then package the trained checkpoint for
prediction/evaluation.

### 11. Train MaskDINO-R50

```bash
export WORKDIR="${WORKDIR:-$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/maskdino_r50}"

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_maskdino_baseline.sh"
```

### 12. Build a package for the trained MaskDINO checkpoint

```bash
export MASKDINO_CKPT="$WORKDIR/model_final.pth"
export MASKDINO_PACKAGE="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/maskdino_r50_local.yaml"

python -m shitspotter.algo_foundation_v3.cli_package build \
    "$MASKDINO_PACKAGE" \
    --backend maskdino \
    --baseline_preset maskdino_r50 \
    --baseline_checkpoint_fpath "$MASKDINO_CKPT" \
    --metadata_name maskdino_r50_local
```

### 13. Run MaskDINO on validation and test

```bash
export PACKAGE_FPATH="$MASKDINO_PACKAGE"
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_maskdino_on_vali.sh"

export PACKAGE_FPATH="$MASKDINO_PACKAGE"
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_maskdino_on_test.sh"
```

### 14. Aggregate results again

```bash
TARGET_DPATH="$DVC_EXPT_DPATH/_foundation_detseg_v3" \
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/aggregate_foundation_results.sh"
```

## Bootstrap a new cohort of phone images

This is the intended annotation-seeding workflow, and it comes last on purpose:
the expectation is that you first tune and compare the detector / segmenter
variants above, then pick the package you trust for a new cohort.

```bash
export COHORT_DPATH=/path/to/new/cohort
export PACKAGE_FPATH="$DEIMV2_SAM2_TUNED_SEG_PACKAGE"

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_bootstrap_new_cohort.sh"
```

If you want to bootstrap with the tuned detector plus zero-shot SAM2 instead,
point `PACKAGE_FPATH` at `$DEIMV2_SAM2_TRAINED_PACKAGE` instead.

If you already have a prediction kwcoco file and only want the LabelMe sidecars:

```bash
python -m shitspotter.algo_foundation_v3.cli_export_labelme \
    /path/to/pred.kwcoco.zip \
    --only_missing True
```

## What the package step is for

The package file is a small YAML manifest for inference. It records:

- which backend to run
- which detector / segmenter / baseline checkpoint files to load
- which preset/config variant to use
- default postprocess settings such as score threshold, NMS, crop padding, and polygon simplify

The package step is not required to fine-tune the DEIMv2 detector or the SAM2
segmenter. Training uses the kwcoco inputs plus the training CLI flags, not the
package file.

## Notes on external weights

- DEIMv2 S/M training configs expect local backbone init files under
  `$SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts/`. The download helper places them there.
- Reusable DEIMv2 detector checkpoints are stored under
  `$FOUNDATION_V3_MODEL_DPATH/deimv2/` so they behave like the older repo
  convention of keeping model assets under the data DVC tree.
- SAM2 checkpoints are stored under
  `$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/`, which matches the upstream SAM2
  repo layout.
- If you want to skip local SAM2 checkpoints entirely, the package/runtime can
  fall back to the Hugging Face model ids in the preset registry. The local
  checkpoint path is still the easiest fully explicit path for reproducible
  experiment scripts.
