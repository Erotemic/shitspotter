# v4 vs kit — apples-to-apples re-evaluation (2026-05-22)

## TL;DR

Every kit-trained run (v6/v7/v8) was compared against v4's self-reported
AP from a different eval pipeline. When v4's checkpoints are evaluated
with the **kit's eval driver** (the same one v6/v7/v8 use), v4 scores
substantially higher than the previously-recorded numbers — and the kit
runs come in well below v4 on both cells.

The root cause is **not** a kit code bug: the kit's
`run_kwcoco_eval` is correct and consistent. The root cause is a
**recipe-level path bug**: v6/v7/v8 trained on a "simplified" source
bundle that contains only 25 % of v4's training images.

## Corrected headline table (all numbers from the kit's eval driver,
v9 simplified test set, score_thresh=0.001)

| Cell        | v4 self-reported | **v4 (kit eval)** | v6 (kit) | v7 (kit) | v8 (kit) | Δ v7 vs v4 | Δ v8 vs v4 |
|-------------|------------------|-------------------|----------|----------|----------|------------|------------|
| pico@416    | 0.406            | **0.4548**        | 0.386    | 0.398    | 0.387    | **−0.057** | **−0.068** |
| n@640       | 0.520            | **0.5553**        | —        | 0.535    | 0.515    | **−0.020** | **−0.040** |

The "v7 n@640 WIN by +0.015" recorded in [`v7 EVAL.md`](mobile_app_training_v7/EVAL.md)
(commit `7dec7c6`) was wrong — based on the v4 self-report. v7 actually
**loses to v4 by 0.020** on the kit's consistent eval.

The "v6 pivot validated within DETR noise" recorded in
[`v6 EVAL.md`](mobile_app_training_v6/EVAL.md) (commit `434fc8d`) was
wrong — the actual gap on pico is **0.057**, way outside noise.

## What changed: the apples-to-apples test

Ran v4's `best_stg2.pth` through the kit's `run_kwcoco_eval`
(`score_thresh=0.001` since `31836c5`), with v4's own train.yml's
`__include__` patched to point at the kit's `tpl/DEIMv2/configs/`
(which is at the same DEIMv2 SHA `377e10a` as v4's — confirmed below).

| Cell    | v4 ckpt + kit eval |
|---------|-------------------:|
| pico@416 | 0.4548 |
| n@640    | 0.5553 |

This is the baseline every kit run should be compared against. The
v4 self-reported numbers (0.406 / 0.520) came from v4's
`04_eval_on_test.sh` → shitspotter `algo_foundation_v3 cli_predict_boxes`,
a *different* eval driver that under-reports AP by ~0.03–0.05 on this
test set.

## What was ruled out as the cause of the kit's underperformance

| Suspect | Status | Evidence |
|---|---|---|
| DEIMv2 SHA drift | **No** | `shitspotter/tpl/DEIMv2 = kit/tpl/DEIMv2 = 377e10a` |
| Upstream config drift | **No** | `diff -q` on `deimv2_hgnetv2_pico_coco.yml` and `..._n_coco.yml` returned empty |
| COCO-pretrained init | **No** | both runs use `deimv2_pico_coco.pth`, md5 ce738340… |
| Kit's score_thresh | **Fixed earlier** (`31836c5` lowered 0.30 → 0.001) — produced the 0.343 → 0.386 jump on v6 |
| Kit's eval pipeline   | **No** | now consistently applied to both v4's and the kit's checkpoints |

## What IS the cause: training-data path bug

Source bundles compared:

| | v4 source bundle | v6/v7/v8 source bundle |
|---|---|---|
| Path | `dvc-repos/shitspotter_dvc/train_imgs10671_b277c63d.kwcoco.zip` | `dvc-repos/shitspotter_dvc/simplified_train_imgs7350_4f0174d0.kwcoco.zip` |
| Images | **10 671** | **2 564** (filename misleading) |
| Poop annotations | **7 782** | **4 736** |
| Images-with-poop | **3 696** | **2 564** |

After quadrant tiling (`tile_grid=2, tile_overlap=0.20, tile_output_dim=640`):

| | v4 tile bundle | v6 tile bundle (same kit code) |
|---|---:|---:|
| Total tile images | 53 355 | 12 820 |
| Total annotations | 22 672 | 14 200 |
| Full-frame images | 10 671 | 2 564 |
| Tile images       | 42 684 | 10 256 |

The kit's `tile.py` correctly amplified by ~5× (1 full + 4 tiles per
source). The deficit is upstream — v6 fed it a 4× smaller source.

## The fix

One-line path change in each v6/v7/v8/v9/v10 `run.sh`:

```diff
- RAW_TRAIN=$RAW_DPATH/simplified_train_imgs7350_4f0174d0.kwcoco.zip
- RAW_VALI=$RAW_DPATH/simplified_vali_imgs1258_07ec447d.kwcoco.zip
+ RAW_TRAIN=$RAW_DPATH/train_imgs10671_b277c63d.kwcoco.zip
+ RAW_VALI=$RAW_DPATH/vali_imgs1258_577e331c.kwcoco.zip
```

The kit's `tile.py --category_name poop` will filter the multi-class
bundle to poop-bearing annotations, matching v4's behavior.

## Recommended next experiment: v6.1

Re-run v6 (`pico@416 fixed-policy`) with the corrected source bundle.
Wall-clock ~3 h. If the kit-eval AP lands within ±0.01 of v4's 0.455,
the kit pivot is genuinely validated and we can proceed to v7.1
(multiscale, same corrected data) which should beat v4 on n@640 for
real.

Cost to re-establish the entire ladder on corrected data:
- v6.1 pico@416: ~3 h
- v7.1 pico@416 + n@640: ~3 + 5 = 8 h
- v8.1 (if pursued): much more (~145 h), but we should probably skip
  given v8.0 regressed and the time-to-result is poor

Total budget to lock baselines: **~11 GPU-hours**, vs ~145 h sunk into
v8.0. Cheap relative to the existing investment.

## Why this was missed

1. v4's `04_eval_on_test.sh` used a different eval driver than the
   kit's `run_kwcoco_eval`. We never cross-validated.
2. v6's recipe path was picked plausibly ("simplified" sounds like the
   right input for a simple single-class detector) without confirming
   image counts.
3. The kit's tile step printed image counts but they didn't trigger any
   warning — there's no "expected vs actual" check.

## Kit gaps worth queuing (not for now)

- `tile.py` could emit a stats summary (input / output / dropped) to
  catch order-of-magnitude unexpected losses.
- `run_kwcoco_eval` could write the source-bundle SHA into the
  metrics JSON so manifests are unambiguous.
- A `kwcoco-detector-kit check-data` subcommand that prints
  source-bundle stats given a recipe.yaml.
