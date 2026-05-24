# v8 evaluation — REGRESSION vs v7 (and also vs v4)

> **⚠ TABLE UPDATED 2026-05-22.** The "Δ v4" column below used v4's
> self-reported AP (0.406 / 0.520). Under consistent kit-eval, v4 is
> actually **pico=0.4548 / n=0.5553**, making the true Δ vs v4
> **pico=−0.068 / n=−0.040**. The "regression vs v7" verdict is
> unchanged. The recipe here also uses the wrong source bundle (see
> v6's superseded note). See [`../V4_VS_KIT_APPLES_TO_APPLES.md`](../V4_VS_KIT_APPLES_TO_APPLES.md).

## Headline numbers

| Cell        | v4 base | v7 best | **v8 (3 rds)** | Δ v7→v8 | Δ v4→v8 | desk ms |
|-------------|---------|---------|----------------|---------|---------|---------|
| pico@416    | 0.406   | 0.398   | **0.387**      | −0.011  | −0.019  | 13.7    |
| n@640       | 0.520   | 0.535   | **0.515**      | **−0.020** | −0.005 | 49.6 |

n@640's v7 win is **lost**. Both cells are below v4. Test went DOWN even
though val trajectories went UP.

## Per-round val AP@0.5 (multi-scale tile pool — not directly comparable to v7's val)

| Cell        | round 0 | round 1 | round 2 | gain | shape |
|-------------|---------|---------|---------|------|-------|
| pico@416    | 0.796   | 0.808   | 0.810   | +0.014 | plateauing |
| n@640       | 0.799   | 0.834   | 0.834   | +0.035 | plateau after rd 1 |

## Why val improved but test regressed

Two compounding factors:

1. **Hard-neg mining bias.** Each round trains on the prior model's
   false positives. The decision boundary shifts conservative — better
   at rejecting the mined tile distribution, worse at recall on the
   actual test distribution. Classic mining failure mode on a small
   dataset where recall is the load-bearing factor.

2. **v8 dropped multiscale.** `round-loop` hardcodes `--train_policy
   fixed`. v7 = multiscale + COCO init. v8 = fixed + COCO init + 3
   rounds of mining. The mining didn't gain enough to offset the lost
   multiscale.

So this experiment **didn't isolate "mining vs no mining"** — it
compared "multiscale" against "fixed + mining". Not the same hypothesis
we set out to test.

## Decision

- [x] **v7 remains the best baseline.** Mining as currently configured
      hurts on test. v10 ship cut uses v7-style (multiscale, no mining)
      unless v9 lands ≥ +0.02 AP.
- [ ] **Skip v9 if v8 had won.** v8 didn't win, so v9 is unblocked.
- [ ] **Optional future v8' = multiscale + mining.** A `round-loop`
      patch to accept `--train_policy multiscale_*` would let us test
      whether mining helps *on top of* multiscale instead of in lieu
      of it. Not chasing this now; v9 is the higher-EV experiment.

## Cost incurred

- pico@416: ~20 GPU-hours (3 rounds × ~7 hrs/round on a 3090)
- n@640: ~45 GPU-hours (3 rounds × ~15 hrs/round)
- Total: ~65 GPU-hours, plus a couple of round-loop bugs found and fixed.

The bug haul is a real positive: kit commits `b0db63c` (merge.py abs
paths), `017ca44` (mine.py abs paths + round_loop resume), `78c5654`
(use_gateway per variant), and `9028a52`/`b5aeaec` (mining budget +
O(N) stratification) all landed because v8 surfaced them.

## Artifacts

- pico@416 final round: `/data/joncrall/kcd/v8/deimv2_pico_416x416/rounds/round2/runs/deimv2_pico_416x416/`
  - best_stg2.pth, export/deimv2_h416_w416.onnx, eval/v8_deimv2_pico_416x416_round2/eval/detect_metrics.json
- n@640 final round: `/data/joncrall/kcd/v8/deimv2_n_640x640/rounds/round2/runs/deimv2_n_640x640/`
  - best_stg2.pth, export/deimv2_h640_w640.onnx, eval/v8_deimv2_n_640x640_round2/eval/detect_metrics.json

## Next step

Advance to v9 distillation **based on v7's recipe** (multiscale policy,
COCO init), with the v9 OGDino bbox teacher providing pseudo-GT on the
v6 tiled training pool. See [`../mobile_app_training_v9/`](../mobile_app_training_v9/).
