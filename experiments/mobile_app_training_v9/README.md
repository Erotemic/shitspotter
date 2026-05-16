# mobile_app_training_v9 — distillation from v9 OGDino bbox teacher

v9 uses the trained OpenGroundingDINO + SAM2 package
([`v9_opengroundingdino_sam2_1_hiera_base_plus_tuned.yaml`](../foundation_detseg_v3/packages/v9_opengroundingdino_sam2_1_hiera_base_plus_tuned.yaml),
AP=0.766 on the simplified test GT) as a **bbox-only teacher**. SAM2 is
not in the loop — the student is a detector, so only box-level
supervision matters.

| Cell        | v8 AP | v9 distilled AP | Δ |
|-------------|-------|-----------------|---|
| pico@416    | TBD   | TBD             | TBD |
| n@640       | TBD   | TBD             | TBD |

## Distillation strategy (offline pseudo-GT)

1. `kwcoco-detector-kit pseudo-label` runs the teacher on the v6
   train tile bundle, producing a pseudo-GT kwcoco.
2. `run.sh` merges human-GT and pseudo-GT annotations into a single
   training kwcoco, tagging pseudo annotations with a `from_teacher`
   field so the trainer can downweight them if needed (default 1.0).
3. `recipe-run` trains the student against the merged bundle, same
   recipe shape as v6.

This is the **simpler half** of distillation — no teacher-forward
pass during training. The student only sees the teacher's boxes as
extra GT. Tradeoff: no soft-label probabilities, but no runtime
teacher cost either.

## Why bbox-only teacher

Confirmed with the user: OGDino's bbox head is the higher-quality
signal anyway; SAM2's mask refinement does not change the boxes it
conditions on. See
[`dev/journals/lessons_learned.md`](../../dev/journals/lessons_learned.md)
for the full reasoning when it lands.

## Quick start (inside the docker image)

```bash
docker run --gpus=all -it --rm \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v9/run.sh
```

## Prerequisites

- v8 done (carries the round-mined checkpoints forward).
- v9 OGDino package present on the
  `/data/joncrall/dvc-repos/shitspotter_expt_dvc/` mount.
- OGDino MSDeformAttention extension built into the image (already
  done by the dockerfile).

## Success criterion

At least +3 AP on `n@640` over its v8 number. pico@416 is harder to
distill into (capacity gap); +2 AP on pico is acceptable.
