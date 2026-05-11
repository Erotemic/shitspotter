# mobile_app_training_v5 — design

## Problem statement

v4 trains DEIMv2 detectors at a fixed input resolution. The training
data is "the full image, downsized to fit the input grid" plus "2×2
overlapping tiles from the full-resolution image". The detector sees
objects at *the apparent scale they happen to land at* in each
image — there's no explicit mechanism to teach it that a given
physical object can be far away (small) or close (large) within the
same scene.

The deployed phone app needs the detector to work across at least
three operating modes that all feed it different effective scales:

```
FAST_FULL_FRAME       full frame, model input 320 or 416
BALANCED_FULL_FRAME   full frame, model input 640
ROI_HIGH_RES          a region cropped from a higher-res capture
TILED_SEARCH          one of N overlapping tiles per frame
```

A model that has only seen objects at one apparent-size distribution
during training will systematically underperform when deployed in the
modes that produce a different distribution.

## Two ideas, one pipeline

### 1. Multi-scale fixed-size tile extraction

For each source image, generate **multiple downscaled copies** at a
fixed list of scales (default `1.0, 0.66, 0.40, 0.25`). From each
downscaled copy, cut **fixed-size** (default 320×320) sliding-window
tiles. Each output tile is one model input.

Effects:

* The same physical poop appears in multiple tiles, at different
  apparent sizes — large in the 1.0×-scale tile, small in the
  0.25×-scale tile. The detector learns to handle both.
* Every tile is identical in resolution, so DEIMv2's fixed-input
  HGNetv2 encoder (the same one v4 trains) works without any per-
  batch dynamic resize trick.
* The pool size grows ~4× per scale, so total tile count scales as
  `image_count × num_scales × num_grid_positions_per_scale`. With
  defaults that's still tractable (tens of thousands of tiles for
  the 10k-image v9 train split).

Annotations are warped through the downscale + crop and clipped to
the tile. Tiles whose surviving GT area falls below
`min_gt_area_frac × tile_area` are tagged **negative**; the rest are
**positive**. The split is recorded as image-level `tile_role`
metadata so downstream stages can route the two pools differently.

### 2. Round-based hard-negative mining

For DETR-family detectors trained on cluttered outdoor scenes, the
hard cases are often:

* Grass texture that fires a false poop detection.
* Tree bark, mulch, leaves at certain scales.
* Shadows under low sun.

These tiles, in a pure-positive training run, never enter the
gradient at all (they're negatives — no GT). A naïve fix is to
include random negatives, but most of them are easy (large flat
sidewalk or sky regions). Those don't teach the model anything new.

**Hard-negative mining** instead asks the model itself which
negatives it's currently getting wrong, and uses those as the next
round's negative half. Concretely:

```
round 0:
  train on positives + a uniform random sample of negatives
  (ratio neg_over_pos = 3.0 by default)

round 1:
  run trained model on ALL negative tiles
  pick the top-K with max_pred_score >= 0.30   <- "hard negatives"
  train on positives + those hard negatives
  starting from round 0's checkpoint

round 2:
  same, mining from round 1's model

...
```

This is the standard hard-negative-mining loop, just structured as
explicit rounds rather than online sampling. The explicit-round form
is easier to debug (each round's checkpoint and mining stats are
preserved on disk) and easier to dispatch to v4's existing
single-run trainer.

## Why round-based rather than online mining

We considered online mining (re-score every K iterations, push hard
negs into the next batch via a custom dataloader). Rejected because:

1. **DEIMv2's trainer is upstream code.** Injecting an online miner
   requires patching `tpl/DEIMv2/engine/solver/det_solver.py`, which
   couples v5 to a specific upstream commit.
2. **Round artefacts are debuggable.** Each round has a workdir, a
   policy.json, a mined-negs kwcoco, a score histogram sidecar. We
   can look at the *delta* between rounds and reason about whether
   mining is helping.
3. **Iteration cost is bounded.** A round is `train ~20 epochs +
   mine`, predictable wall-clock-wise. Online mining couples
   miner latency to the trainer's iteration speed.

## Where v5 inherits from v4

Everything model-side:

* DEIMv2 variant choices (deimv2_n, deimv2_pico, deimv2_s).
* The trainer wrapper (`_train_deimv2_variant.sh`), including the
  HGNetv2 fixed-size invariant (`V4_TRAIN_POLICY=fixed`),
  RLIMIT_NOFILE handling, the per-`(variant, input_h)` batch table,
  the OOM fail-hint, and the YAML indent fix.
* The ONNX exporter (`03_export_onnx.sh`) including the modelspec
  sidecar.
* The kwcoco-based evaluator (`04_eval_on_test.sh`) against the
  v9-canonical simplified test GT.
* The eligibility manifest semantics (HOST_PROMISING / PHONE_ELIGIBLE
  / candidate_kind=real|smoke).

v5 dispatches to these scripts by setting `V4_ROOT` to a per-round
subdirectory. Each round writes a v4-shaped workdir at
`$V5_ROOT/rounds/round<N>/v4_root/runs/<candidate_id>/`.

## What v5 contributes

Three new Python modules under `experiments/mobile_app_training_v5/`:

* `v5_tile.py` — multi-scale fixed-size tile extractor.
* `v5_merge.py` — combine positives + (round-0 random or round-N+ hard)
  negatives into a single training kwcoco.
* `v5_mine.py` — load a trained DEIMv2 checkpoint, score every
  negative tile, emit a kwcoco of the hard ones.

Plus four shell drivers that compose v5 around v4's trainer:

* `01_make_multiscale_tile_dataset.sh` — calls `v5_tile.py` for train + vali.
* `02_train_round.sh` — merge + dispatch v4 trainer for one round.
* `03_mine_hard_negatives.sh` — mining pass after a round.
* `run_round_loop.sh` — the round-by-round driver.

`run_all.sh` ties everything together and is the recommended
entrypoint.

## Open questions / decisions tagged for follow-up

* **Tile size vs. export size.** Currently both default to 320, which
  matches the smallest export resolution in v4's sweep. For larger
  deploy resolutions (416, 512, 640), should v5 retile? Or train at
  320 and rely on test-time multi-scale? Untested; the round loop
  is parameterised on `V5_TILE_SIZE` so retiling is one knob away.
* **Hard-neg score threshold.** 0.30 matches v4's default deploy
  score threshold. The right answer probably depends on the round's
  PR curve. A future refinement: pick the threshold each round so
  the resulting hard-neg count lands in `[0.5 × pos_count, 5 × pos_count]`.
* **Stop criterion.** v5 currently runs a fixed `V5_NUM_ROUNDS=3`.
  A more principled stop: train round N, compare AP to round N-1,
  stop if it didn't improve. Not implemented; saved for v6.
* **Cross-scale annotation duplication.** A poop at the 1.0 scale and
  the 0.25 scale is the same physical object. The detector sees both
  as positive examples — which is good for scale invariance but
  could double-count IoU during loss. Currently we don't deduplicate;
  the v9 simplified GT path uses cluster-level merge but only at
  the source-image level. Worth measuring.

## File layout

See [README.md](README.md) for the operational layout. The DESIGN.md
file is the rationale; the README is the operating manual.
