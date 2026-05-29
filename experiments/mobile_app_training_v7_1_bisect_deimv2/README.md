# DEIMv2 bisect — does the SHA bump explain the residual?

A diagnostic experiment, not a quality push. Re-runs v7.1 pico@416
under DEIMv2 `377e10a` (v4's SHA) to isolate the bump's contribution.

## What this tests

v7.1 (on DEIMv2 `aeabc7e`) lands at:
- pico@416: 0.4329 (v4 kit-eval = 0.4548, Δ = −0.022)
- n@640:    0.5350 (v4 kit-eval = 0.5553, Δ = −0.020)

The ~0.02 residual on **both** cells is consistent. The source bundle
was already ruled out for n@640 (v7 = v7.1 on n). Candidates that
could affect both cells equally include the DEIMv2 SHA bump. This
experiment isolates one of them.

## Setup

Kit commit [`a8ca45e`](https://github.com/Erotemic/kwcoco_detector_kit/commit/a8ca45e)
rolls `tpl/DEIMv2` from `aeabc7e` → `377e10a` (v4's SHA). Two commits
on the upstream DEIMv2 fork separate them:

- `aeabc7e setup_print: prefix every emitted line with an ISO-8601 timestamp`
- (the DDP loss-key alignment commit, single-GPU-irrelevant)

Cosmetic + DDP-targeted, *should* have no effect on single-GPU pico
training. This experiment verifies that intuition.

## Decision matrix

| Bisect pico AP | Interpretation | Next |
|----------------|----------------|------|
| ≥ 0.45         | DEIMv2 bump caused the residual. Surprising given the diff. | Restore main to aeabc7e (so sealions works); pin shitspotter recipes to a kit branch with 377e10a until upstream releases something we'd actually want |
| ~0.45 (Δ ≥ +0.015 vs v7.1) | DEIMv2 bump partially caused it. | Same as above. |
| ~0.43 ± 0.01 (no movement) | DEIMv2 isn't the cause. Look elsewhere (multi-class refactor, tile semantics, train config detail). | Restore main; investigate other suspects or move on to v9 distillation. |
| < 0.42         | DEIMv2 rollback HURT. Unexpected. | Restore main; document; move on. |

## Quick start

```bash
# After rebuilding the image so the bisect kit + DEIMv2 377e10a get
# baked in:
docker run --gpus=all -it --rm \
    --shm-size=32g \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/home/joncrall/data/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/home/joncrall/data/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/shitspotter_v4:/data/joncrall/shitspotter_v4:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v7_1_bisect_deimv2/run.sh 2>&1 \
    | tee /tmp/v7_1_bisect.log
```

Wall-clock: ~3 GPU-hours on a single 3090.

## Cleanup after the experiment

Once the bisect lands and EVAL.md is filled in, the next kit commit
on main should restore `tpl/DEIMv2` to `aeabc7e` so the sealions line
isn't blocked. Don't forget — `a8ca45e` is marked "temporary" in its
message specifically because of this.
