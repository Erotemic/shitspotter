# mobile_app_training_v8 — round-based hard-negative mining

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

## Quick start (inside the docker image)

```bash
docker run --gpus=all -it --rm \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v8/run.sh
```

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
