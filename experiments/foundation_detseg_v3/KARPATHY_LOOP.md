# Foundation V3 Karpathy Loop

This is a tighter, repeatable loop for improving the foundation-v3 stack
without conflating detector errors, segmenter errors, and package/runtime glue.

The key lesson from the latest run is:

- tuned DEIMv2 reported about `0.457` bbox AP inside its own validation loop
- the combined `deimv2_sam2` package scored about `0.001` AP on validation

That gap is too large to treat as "the models are just weak". It means we
should optimize the system in stages and refuse to let a later stage hide
whether an earlier stage is already working.

## Principles

Use fixed gates:

1. Detector-only: does the tuned detector still have a good box metric?
2. Segmenter-only: given ground-truth boxes, does SAM2 produce usable masks?
3. Combined package: when we compose the two, does the end-to-end metric stay
   in the same ballpark or collapse?

Only move to the next stage when the previous one looks sane.

Use a fixed "casebook":

- full validation split for numbers
- a small hand-picked visual subset for fast inspection
- the same subset every iteration until a clear improvement appears

## Shell Setup

```bash
export SHITSPOTTER_DPATH="${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}"
export DVC_DATA_DPATH="${DVC_DATA_DPATH:-$(geowatch_dvc --tags="shitspotter_data")}"
export DVC_EXPT_DPATH="${DVC_EXPT_DPATH:-$(geowatch_dvc --tags="shitspotter_expt")}"

export SHITSPOTTER_DEIMV2_REPO_DPATH="${SHITSPOTTER_DEIMV2_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/DEIMv2}"
export SHITSPOTTER_SAM2_REPO_DPATH="${SHITSPOTTER_SAM2_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/segment-anything-2}"

export FOUNDATION_V3_TRAIN_KWCOCO_FPATH="${FOUNDATION_V3_TRAIN_KWCOCO_FPATH:-$DVC_DATA_DPATH/train_imgs5747_1e73d54f.kwcoco.zip}"
export FOUNDATION_V3_VALI_KWCOCO_FPATH="${FOUNDATION_V3_VALI_KWCOCO_FPATH:-$DVC_DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip}"
export FOUNDATION_V3_TEST_KWCOCO_FPATH="${FOUNDATION_V3_TEST_KWCOCO_FPATH:-$DVC_DATA_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip}"
```

## Loop A: Tune The Detector

Train DEIMv2 and trust its native bbox validation metric first.

```bash
export DEIMV2_INIT_CKPT="$DVC_DATA_DPATH/models/foundation_detseg_v3/deimv2/deimv2_dinov3_m_coco.pth"
export WORKDIR="$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/deimv2_m"

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
```

Record:

- `best_stat`
- epoch of `best_stat`
- checkpoint path you intend to deploy
- 3 to 10 qualitative examples from validation

Gate:

- do not tune SAM2 yet if detector bbox AP regresses meaningfully

## Loop B: Tune The Segmenter In Isolation

Tune SAM2, but score it using ground-truth boxes as prompts. This isolates mask
quality from detector quality.

Train:

```bash
export SAM2_INIT_CKPT="$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt"
export SAM2_WORKDIR="$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/sam2.1_hiera_base_plus"

bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/train_sam2_segmenter.sh"
```

Build a package for the detector checkpoint you currently trust plus the SAM2
checkpoint you want to test:

```bash
export DEIMV2_TRAINED_CKPT="$WORKDIR/best_stg2.pth"
export SAM2_TRAINED_CKPT="$SAM2_WORKDIR/checkpoints/checkpoint.pt"
export PACKAGE_FPATH="$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/deimv2_sam2_loop.yaml"

python -m shitspotter.algo_foundation_v3.cli_package build \
    "$PACKAGE_FPATH" \
    --backend deimv2_sam2 \
    --detector_preset deimv2_m \
    --segmenter_preset sam2.1_hiera_base_plus \
    --detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT" \
    --segmenter_checkpoint_fpath "$SAM2_TRAINED_CKPT" \
    --metadata_name deimv2_sam2_loop
```

Run GT-box prompted SAM2 on validation:

```bash
export GTBOX_VALI_DPATH="$DVC_EXPT_DPATH/_foundation_detseg_v3/sam2_gtboxes_vali"

python -m shitspotter.algo_foundation_v3.cli_predict_gtboxes \
    "$FOUNDATION_V3_VALI_KWCOCO_FPATH" \
    --package_fpath "$PACKAGE_FPATH" \
    --dst "$GTBOX_VALI_DPATH/pred.kwcoco.zip"

python -m kwcoco eval \
    --true_dataset "$FOUNDATION_V3_VALI_KWCOCO_FPATH" \
    --pred_dataset "$GTBOX_VALI_DPATH/pred.kwcoco.zip" \
    --out_dpath "$GTBOX_VALI_DPATH/eval" \
    --out_fpath "$GTBOX_VALI_DPATH/eval/detect_metrics.json" \
    --confusion_fpath "$GTBOX_VALI_DPATH/eval/confusion.kwcoco.zip" \
    --draw False \
    --iou_thresh 0.5
```

Gate:

- if GT-box SAM2 is poor, do not blame the detector
- inspect those predictions visually before changing detector settings

## Loop C: Compose Detector And Segmenter

Only after loops A and B look sane, run the full package:

```bash
bash "$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/hack_oneoff_trained_deimv2_sam2_eval.sh"
```

Interpretation:

- detector good, GT-box SAM2 good, combined bad:
  focus on composition and postprocess
- detector good, GT-box SAM2 bad:
  focus on SAM2 tuning and prompting
- detector bad:
  fix detector first

## Loop D: Sweep Postprocess Before Retraining

Before launching another expensive training job, sweep the cheap knobs:

- `score_thresh`
- `nms_thresh`
- `crop_padding`
- `polygon_simplify`
- `min_component_area`
- `keep_largest_component`

Suggested first sweep:

```bash
score_thresh in {0.05, 0.10, 0.20}
nms_thresh in {0.3, 0.5, 0.7}
crop_padding in {0, 16, 32, 64}
polygon_simplify in {0.0, 1.0, 2.0}
min_component_area in {0, 16, 32}
```

Do this on validation first. Only send the best few configurations to test.

## Visual Inspection Loop

For every promising or suspicious run:

1. Export a small set of prediction sidecars.
2. Look at false positives, false negatives, and obviously bad polygons.
3. Write down the failure mode in one sentence.
4. Make the next change target that failure mode only.

Useful command:

```bash
python -m shitspotter.algo_foundation_v3.cli_export_labelme \
    "$GTBOX_VALI_DPATH/pred.kwcoco.zip" \
    --score_thresh 0.0
```

Or export from the full composed prediction bundle instead.

## What To Log Each Iteration

For each run, record:

- detector checkpoint
- SAM2 checkpoint
- package path
- postprocess settings
- detector native bbox AP
- GT-box SAM2 validation AP
- full composed validation AP
- 3 representative good cases
- 3 representative bad cases
- one sentence for the suspected bottleneck

## Current Best Guess

Given the latest numbers, the most likely issue is not "the detector forgot
everything". The detector's native validation metric was decent, but the
composed package collapsed. That points to one of these:

- SAM2 fine-tuned checkpoint is poor even when prompted with decent boxes
- box-to-mask prompting is mismatched to the validation distribution
- postprocess is discarding or distorting otherwise usable masks
- the combined metric is dominated by a specific failure mode that is easy to
  see visually but impossible to diagnose from one scalar AP number

So the next rational step is:

1. run GT-box SAM2 validation
2. inspect those masks visually
3. sweep postprocess on validation
4. only retrain after those cheaper checks tell us which stage is weak
