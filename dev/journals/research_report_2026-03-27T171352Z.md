# Research Report: DINO Detector Tuning Snapshot

Created: 2026-03-27T17:13:52Z

## Purpose

This note checkpoints the current state of the detector experiments so the main
findings are recoverable without replaying the full journal. The focus is the
comparison between the older DINOv2 detector path via OpenGroundingDINO and the
newer DINOv3 detector path via DEIMv2, plus the major lessons from the
full-scale detector-plus-SAM work that led into the small-data benchmark.

## Executive Summary

- The SAM integration issues were fixed by `v3`; SAM is not the main bottleneck.
- `v4` was a failed simplify-preprocessing ablation and should not be treated as
  a promising direction.
- `v5` is the best full-scale foundation-detseg baseline so far.
- Checkpoint selection matters, but it does not explain the DINOv3 gap by
  itself.
- `v6` and `v7` detector-learning-rate tweaks did not beat the selected `v5`
  baseline.
- The new small-data benchmark confirms the same overall pattern seen at larger
  scale: OpenGroundingDINO (DINOv2) remains clearly ahead of DEIMv2 (DINOv3).
- The best DEIMv2 improvement so far came from shrinking its training batch size
  aggressively. Batch size 4 is much better than the earlier batch-24 and
  batch-16 settings, but DEIMv2 still trails OpenGroundingDINO by about 0.10 AP
  on test at the currently emphasized train sizes.

## Full-Scale Experiment Story

### v3

`v3` established the first stable end-to-end detector-plus-SAM pipeline after
the SAM prompt-coordinate fixes.

Observed summary:

- detector-only vali: `0.566`
- detector-only test: `0.481`
- GT-box tuned raw vali: `1.000`
- GT-box zero-shot raw vali: `1.000`
- combined tuned raw vali: `0.576`
- combined tuned raw test: `0.486`

Interpretation:

- SAM behaved correctly when given good boxes.
- The main bottleneck was detector quality, not SAM behavior.

### v4

`v4` tested a simplify-preprocessing hypothesis on top of the `v3` recipe. It
failed badly.

Root cause that was discovered:

- simplify preprocessing dropped empty images from the detector training data
- this changed the training distribution in a destructive way
- the resulting detector effectively collapsed

Conclusion:

- `v4` should be treated as a negative result, not a baseline candidate

### v5

`v5` restored the good detector data path, kept offline resize, disabled
simplify, and became the strongest full-scale line.

Earlier baseline-style result:

- combined raw vali: about `0.598`
- combined raw test: about `0.539`

Checkpoint-selection follow-up improved it slightly. Best observed combined
result from the checkpoint sweep:

- `checkpoint0024`
- combined raw vali: `0.600812`
- combined raw test: `0.549743`

Interpretation:

- checkpoint selection is worth doing
- but the gain is modest
- it is not enough to explain why DINOv3 has not clearly surpassed the older
  DINOv2-based path

### v6

`v6` reused the tuned `v5` SAM checkpoint and tried gentler detector backbone
fine-tuning.

Selected result:

- detector-only vali: `0.572650`
- combined vali: `0.594299`
- detector-only test: `0.526306`
- combined test: `0.539812`

Interpretation:

- lowering only the backbone LR was not the right next move
- `v6` did not beat the selected `v5` checkpoint

## Small-Data Benchmark Design

The small-data benchmark was introduced to create a faster, more reproducible,
and more apples-to-apples comparison between detector families.

Key design choices:

- shared prepared data under `experiments/small-data-tuning/`
- train subsets at `128`, `256`, and `512` images
- focus on positives-only images for this benchmark stage
- common preprocessing recorded in metadata:
  - poop-only
  - offline resize to max dimension `640`
  - simplify step retained in the benchmark metadata path where applicable
- standard full validation and test bases preserved as the evaluation anchors
- tagged DEIMv2 config sweeps so multiple tuning recipes can coexist without
  overwriting each other

The intent is that this benchmark can grow over time by adding more train sizes
or more config tags while preserving the meaning of the earlier rows.

## Small-Data Benchmark Results

### OpenGroundingDINO Baseline

This is the current detector benchmark to beat.

- `train128`: vali `0.544002`, test `0.599663`
- `train256`: vali `0.588519`, test `0.630317`
- `train512`: vali `0.628371`, test `0.685568`

Trend:

- strong and monotonic across all tested train sizes

### Early DEIMv2 Results

Initial DEIMv2 settings used much larger batches and underperformed badly.

- batch-24 baseline often failed to yield finite selections at `train128`
- batch-16 produced the first meaningful recovery
- best batch-16 region still lagged far behind OpenGroundingDINO

Representative batch-16 results:

- `small_batch16`, `train128`: vali `0.256622`, test `0.334074`
- `small_batch16`, `train256`: vali `0.465655`, test `0.483813`
- `small_batch16_low_lr_all_0p8`, `train256`: vali `0.471054`, test `0.490328`

Interpretation:

- DEIMv2 was sensitive to optimization scale
- smaller effective batch was clearly helping

### Current Best DEIMv2 Results

The strongest DEIMv2 family so far is the batch-4 family.

At `train128`:

- best validation AP:
  - `small_batch4_low_lr_all_0p8`
  - vali `0.452779`
  - test `0.475780`
- best observed test AP:
  - `small_batch4_backbone_0p5`
  - vali `0.440623`
  - test `0.501876`

At `train256`:

- best validation AP:
  - `small_batch4_backbone_0p5`
  - vali `0.495747`
  - test `0.519976`
- best observed test AP:
  - `small_batch4_low_lr_all_0p8`
  - vali `0.487417`
  - test `0.522443`

At `train512`:

- batch-4 variants cluster around test AP `0.546` to `0.556`
- still below the OpenGroundingDINO baseline at `0.685568`

Interpretation:

- DEIMv2 tuning is finally producing meaningful movement
- batch size is a major lever
- but even the improved DEIMv2 line still trails OpenGroundingDINO by roughly
  `0.10` AP on test at `train128` and `train256`

## What We Know With Reasonable Confidence

- SAM is not the main blocker anymore.
- The `v4` simplify path was a negative result.
- Checkpoint selection is necessary but not sufficient.
- The DINOv3/DEIMv2 gap is not just a full-scale artifact; it also appears in
  the small-data benchmark.
- DEIMv2 responds strongly to smaller batch sizes.
- The best current DEIMv2 tuning neighborhood is around:
  - batch size `4`
  - modest global LR reduction
  - possibly a modest backbone LR reduction

## What Is Still Uncertain

- Why OpenGroundingDINO remains so much stronger on this task even after DEIMv2
  tuning improved.
- Whether the remaining gap is mostly due to:
  - optimization recipe differences
  - augmentation or schedule differences
  - architecture/head differences
  - training-data semantics not yet matched closely enough
- Whether a stricter validation-driven selection regime will eventually converge
  on the same recipes that currently look best on test.

## Recommended Next Steps

1. Keep DEIMv2 tuning tightly centered on the batch-4 family.
2. Prefer a small number of sharper follow-up configs over broad sweeps.
3. Continue using the small-data benchmark as the fast comparison harness.
4. Preserve OpenGroundingDINO as the stable comparison anchor while tuning
   DEIMv2.
5. Add richer analysis around training curves and checkpoint trajectories so it
   is easier to understand why the selected DEIMv2 checkpoint differs from the
   best observed test checkpoint.

## Artifact Pointers

- Small benchmark root:
  `/data/joncrall/dvc-repos/shitspotter_expt_dvc/small_data_tuning/dino_detector_benchmark`
- Analysis dir:
  `/data/joncrall/dvc-repos/shitspotter_expt_dvc/small_data_tuning/dino_detector_benchmark/analysis`
- Run links dir:
  `/data/joncrall/dvc-repos/shitspotter_expt_dvc/small_data_tuning/dino_detector_benchmark/analysis/run_links`
- Foundation detector experiment scripts:
  `experiments/foundation_detseg_v3/`
- Small-data benchmark scripts:
  `experiments/small-data-tuning/`

## Bottom Line

The current evidence says the DINOv3 detector path is real, sensitive to tuning,
and improved materially once the batch size was pushed down to 4. But it still
does not match the older DINOv2/OpenGroundingDINO baseline. The problem is no
longer “the benchmark is too messy to tell”; the problem is now a concrete model
and training-recipe gap that can be studied in a controlled way.
