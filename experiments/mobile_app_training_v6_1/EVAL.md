# v6.1 evaluation — source-bundle fix delivered ~60% of the gap

## Headline numbers

| Cell     | v4 (kit eval) | v6.0  | **v6.1**  | v7    | v8    | Δ v6.1 vs v6.0 | Δ v6.1 vs v4 |
|----------|---------------|-------|-----------|-------|-------|----------------|--------------|
| pico@416 | 0.4548        | 0.386 | **0.4211**| 0.398 | 0.387 | **+0.035**     | **−0.034**   |

| Metric                | v6.1 measured |
|-----------------------|---------------|
| Test AP @ IoU=0.5     | **0.4211**    |
| Desktop CPU mean (ms) | 14.4          |
| Desktop CPU p50 (ms)  | 14.4          |
| Desktop CPU p99 (ms)  | 14.8          |
| Eligibility class     | HOST_PROMISING|
| Eval pipeline         | kit's `run_kwcoco_eval` (score_thresh=0.001) |

## Verdict — partial confirmation

The source-bundle hypothesis from
[`../V4_VS_KIT_APPLES_TO_APPLES.md`](../V4_VS_KIT_APPLES_TO_APPLES.md)
predicted that fixing the recipe's source-bundle path would close most
of the kit-vs-v4 gap. It delivered **+0.035 AP** (about **60% of the
0.057-gap to close**), so the hypothesis is **partially confirmed**:
the source bundle was a load-bearing variable, but not the whole story.

A −0.034 residual still separates v6.1 from v4's kit-eval baseline.
Per the [research journal's 2026-05-24 entry](../RESEARCH_JOURNAL.md),
this result conflates **three** variables we deliberately combined for
time savings:

1. The corrected source bundle (`train_imgs10671` vs `simplified_*7350`)
2. The DEIMv2 submodule bump (`377e10a` → `aeabc7e`, including the
   "DDP loss-key alignment" change)
3. The kit's multi-class refactor + universal-tile architecture

We cannot attribute the +0.035 to any single one in isolation, only
that the combination produced this.

## What we still don't fully understand

Candidates for the remaining −0.034 to v4:

- v4's `tile_kwcoco.py` may handle tile-boundary annotations
  differently from the kit's `tile.py` quadrant mode (clipping vs
  dropping, JPEG quality, etc.).
- v4's `_train_deimv2_variant.sh` had additional config logic
  (`num_top_queries` clamping, etc.) that the kit may not replicate
  exactly.
- DEIMv2 `aeabc7e`'s "DDP loss-key alignment" change is single-GPU
  irrelevant for us, but other changes in `377e10a..aeabc7e` could
  affect training math.

The pragmatic question is whether **the residual is worth chasing**
given v6.1 is now reproducible, traceable, and within ~10% relative
of v4 — and v7 (multi-scale) typically adds +0.012 over a fixed-policy
baseline on this data.

## Provenance (auto-stamped, first run with this enabled)

From `policy.json`:

```json
"provenance": {
    "kit_sha": "5b9f49d1b3a1be4853b8ba58ad5012edf56b1588",
    "kit_dirty": false,
    "deimv2_sha": "aeabc7e400e59b9d2b427ccb2556561ab3bbe26d",
    "deimv2_dirty": false,
    "opengroundingdino_sha": "9ddf10371a46ddca080b9319306185bc704325e5",
    "opengroundingdino_dirty": false,
    "source": "file"
}
```

(NB: `detect_metrics.json`'s provenance + eval_inputs blocks did NOT
land in this run due to a stale `category_name` reference in
`run_kwcoco_eval` post the multi-class merge. Fixed in kit commit
`059f60c`; future eval runs are clean. The training-side stamp in
`policy.json` worked correctly.)

## Decision

- [x] **Take the +0.035 win and proceed.** The kit pivot is now
      reproducible at 0.421 AP — close enough to v4's 0.455 that v7.1
      (multiscale on top of the corrected bundle) is the right next
      experiment. v7's prior multiscale gain over v6.0 was +0.012; if
      that stacks, v7.1 lands at ~0.433, putting us within ~0.02 of
      v4 — solidly inside DETR run-to-run noise.
- [ ] Chase the −0.034 residual. Possible, but probably not worth
      the GPU time relative to v7.1's expected gain.

## Run identity

- Recipe: `experiments/mobile_app_training_v6_1/recipe.yaml`
- Workspace: `/data/joncrall/kcd/v6_1/`
- Manifest: `/data/joncrall/kcd/v6_1/manifest.tsv`
- Kit commit: `5b9f49d`
- DEIMv2: `aeabc7e`, Open-GroundingDino: `9ddf1037`
- Source bundle: `train_imgs10671_b277c63d.kwcoco.zip` (10 671 source
  images, ~53 K tiles after quadrant g2 amplification — matching v4's
  53 355)
