# mobile_app_training_v8 — round-based hard-negative mining

> **Morning kickoff routine (copy-paste, then walk away):**
> ```bash
> cd ~/code/shitspotter
> git fetch aivm-aivm-2404-fe98f795 && git merge --ff-only aivm-aivm-2404-fe98f795/main
> ./reproduce/mobile_quality_push.sh build        # ~30s, kit-layer rebuild
> ./reproduce/mobile_quality_push.sh compare      # confirm v7 numbers landed
> # If v7 looks reasonable, kick off v8 (8-10 h, both cells, 3 rounds each):
> ./reproduce/mobile_quality_push.sh v8 2>&1 | tee /tmp/v8_run.log
> ```
> Spot-check around T+15 min (multi-scale tile step done, round 0 started)
> and T+30 min (DEIMv2 launch banner shows the COCO init loaded). After
> that you can walk away until evening.
> 
> First-run caveat: this is the **first end-to-end run** of the kit's
> `round-loop` path on this dataset. Most likely failure modes are kit-side
> bugs in tile-split-by-role or the round merge. If the tile step fails or
> round 0 doesn't start within 20 min, ctrl-C and ping me — don't burn the
> day waiting on a hung pipeline.


v5's idea, finally executed at scale through `kwcoco-detector-kit round-loop`.
Round 0 trains on positives + a random sample of negatives. Each subsequent
round runs the prior round's model on the full negative pool, picks the
top-K false-positive tiles, and trains on positives + those hard negatives
starting from the prior checkpoint.

| Cell        | v7 multiscale AP | v8 hard-neg AP | Δ |
|-------------|------------------|----------------|---|
| pico@416    | TBD              | TBD            | TBD |
| n@640       | TBD              | TBD            | TBD |

Unlike v6/v7/v9/v10, v8 is **not** a single-shot recipe — it's a
fixed-count loop. There's no `recipe.yaml`; the loop knobs live in
`run.sh` and the run params (rounds, hard-neg cap, mining threshold)
are exported as env vars.

## Why a loop instead of a recipe

The kit's `round-loop` subcommand is its own state machine: it owns
the pos/neg merge per round, the mining pass at the end of each
round, and the next-round training kickoff. Wrapping it in the
single-shot `recipe-run` would just inline its logic. v8 calls it
directly.

## Quick start — morning kickoff, hands-off all day

The whole sequence is one command via the driver. Expected runtime
~8-10 h for both cells × 3 rounds × 30 epochs/round on a single 3090.

```bash
cd ~/code/shitspotter
./reproduce/mobile_quality_push.sh v8 2>&1 | tee /tmp/v8_run.log
```

Then walk away. Each cell streams its own per-round log under
`/data/joncrall/kcd/v8/{deimv2_pico_416x416,deimv2_n_640x640}/rounds/`.
Round 0 fine-tunes from the matching `deimv2_<variant>_coco.pth` (the
COCO-pretrained checkpoint, same one v6 used). Rounds 1+ resume from
the prior round's `best_stg2.pth` automatically.

### Single-cell variant (pico only, ~3-4 h)

```bash
V8_CELLS="pico:416" ./reproduce/mobile_quality_push.sh v8
```

### Mining budget — preventing 16-hour rounds

The multi-scale tile pool produces ~1.8M negative tiles. Scoring every
one per round at ~19 Hz on a 3090 = ~16 h **per round**. With 3 rounds
that's 48 h just for the mining passes.

v8 now caps mining at `V8_MINE_MAX_CANDIDATES` (default 50000) using a
`stratified_by_image` sample so each source image contributes roughly
equally to the candidate pool. ~30x speedup with negligible recall
loss (we only care about the TOP-K hard negatives at the end anyway,
and 50k>>5k is plenty of headroom). Override:

```bash
V8_MINE_MAX_CANDIDATES=100000 ./reproduce/mobile_quality_push.sh v8
# or, to revert to legacy "score everything":
V8_MINE_MAX_CANDIDATES=0      ./reproduce/mobile_quality_push.sh v8
```

The v8.0 run that hit the 16h mining ETA had this budget = 0; the
2026-05-18 kit commit added the knob with default 50000 going forward.

### GPU note

The Pixel 5 mobile cells are single-GPU by design. **Do not** try to
spread one training run across both GPUs on a host where the second
3090 is on only 2 PCIe lanes — DDP all-reduce is bottlenecked by the
slowest peer and you'll lose throughput. The driver pins to GPU 0
implicitly (the kit defaults `CUDA_VISIBLE_DEVICES=0` when unset, per
[`_env.default_cuda_visible_devices`](../../tpl/kwcoco_detector_kit/kwcoco_detector_kit/_env.py)).
GPU 1 stays free for ad-hoc work.

## Prerequisites

- v7 done. v8 inherits whichever cells improved in v7 as its starting
  variants (override via `V8_CELLS="pico:416 n:640"` if you want to
  force both regardless).
- Multi-scale tile bundles produced once by v8's tile step (separate
  pos and neg). These are written under `/data/joncrall/kcd/v8/data/`
  by `run.sh`.

## Success criterion

At least one cell beating its v7 number by **+2 AP** (false-positive
suppression effect on cluttered outdoor scenes is the load-bearing
gain here).

See `EVAL.md` for the post-run write-up.
