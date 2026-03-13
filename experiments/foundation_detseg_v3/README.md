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

## Copy / paste environment

These defaults match the usual ShitSpotter dev-machine layout with the repo
checked out under `$HOME/code/shitspotter`, DVC paths discovered via
`geowatch_dvc`, and external model repos living under `tpl/` as git submodules.

```bash
export SHITSPOTTER_DPATH="${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}"
export DVC_DATA_DPATH="${DVC_DATA_DPATH:-$(geowatch_dvc --tags="shitspotter_data")}"
export DVC_EXPT_DPATH="${DVC_EXPT_DPATH:-$(geowatch_dvc --tags="shitspotter_expt")}"

export SHITSPOTTER_DEIMV2_REPO_DPATH="${SHITSPOTTER_DEIMV2_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/DEIMv2}"
export SHITSPOTTER_SAM2_REPO_DPATH="${SHITSPOTTER_SAM2_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/segment-anything-2}"
export SHITSPOTTER_MASKDINO_REPO_DPATH="${SHITSPOTTER_MASKDINO_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/MaskDINO}"
```

## Setup

This will initialize the foundation-model git submodules in `tpl/` when using
the default paths. If you override the repo paths above, the script will clone
those repos into your chosen locations instead.

```bash
git -C "$SHITSPOTTER_DPATH" submodule update --init --recursive --depth 1 \
    tpl/DEIMv2 \
    tpl/segment-anything-2 \
    tpl/MaskDINO

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/setup_environment.sh"
```

## CPU smoke tests

```bash
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/cpu_smoke_tests.sh"
```

## Build package files

DEIMv2 + SAM2:

```bash
export DEIMV2_CKPT=/path/to/deimv2_checkpoint.pth
export SAM2_CKPT=/path/to/sam2_checkpoint.pt

python -m shitspotter.algo_foundation_v3.cli_package build \
    "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/deimv2_sam2_local.yaml" \
    --backend deimv2_sam2 \
    --detector_preset deimv2_m \
    --segmenter_preset sam2.1_hiera_base_plus \
    --detector_checkpoint_fpath "$DEIMV2_CKPT" \
    --segmenter_checkpoint_fpath "$SAM2_CKPT" \
    --metadata_name deimv2_sam2_local
```

MaskDINO:

```bash
export MASKDINO_CKPT=/path/to/maskdino_model_final.pth

python -m shitspotter.algo_foundation_v3.cli_package build \
    "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/maskdino_r50_local.yaml" \
    --backend maskdino \
    --baseline_preset maskdino_r50 \
    --baseline_checkpoint_fpath "$MASKDINO_CKPT" \
    --metadata_name maskdino_r50_local
```

## Train DEIMv2 detector

```bash
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
```

## Run DEIMv2 + SAM2 on validation

```bash
PACKAGE_FPATH="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/deimv2_sam2_local.yaml" \
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_vali.sh"
```

## Run DEIMv2 + SAM2 on test

```bash
PACKAGE_FPATH="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/deimv2_sam2_local.yaml" \
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_test.sh"
```

## Train MaskDINO baseline

```bash
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_maskdino_baseline.sh"
```

## Run MaskDINO on validation

```bash
PACKAGE_FPATH="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/maskdino_r50_local.yaml" \
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_maskdino_on_vali.sh"
```

## Run MaskDINO on test

```bash
PACKAGE_FPATH="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/maskdino_r50_local.yaml" \
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_maskdino_on_test.sh"
```

## Bootstrap a new cohort of phone images

```bash
export COHORT_DPATH=/path/to/new/cohort
export PACKAGE_FPATH="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/deimv2_sam2_local.yaml"

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/run_bootstrap_new_cohort.sh"
```

## Export LabelMe sidecars from a kwcoco prediction file

```bash
python -m shitspotter.algo_foundation_v3.cli_export_labelme \
    /path/to/pred.kwcoco.zip \
    --only_missing True
```

## Aggregate evaluation results

```bash
TARGET_DPATH="$DVC_EXPT_DPATH/_foundation_detseg_v3" \
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/aggregate_foundation_results.sh"
```
