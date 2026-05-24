# mobile_app_training_v6_1 — pico@416 rebase on corrected source bundle

v6.1 is a one-variable redo of v6.0 with the **correct source bundle**.
Every other knob is held identical to v6.0 so the AP delta directly
measures the data-path bug's contribution.

## Why this exists

v6.0/v7.0/v8.0 all trained on `simplified_train_imgs7350_*.kwcoco.zip`,
which despite the misleading "7350" in its filename contains only
**2 564 images** — vs the **10 671** in v4's
`train_imgs10671_b277c63d.kwcoco.zip`. After kit quadrant tiling (5×
amplification), v6.0 trained on 12 820 tiles while v4 trained on 53 355.

Under the kit's apples-to-apples eval driver, v4 scores
**pico=0.4548 / n=0.5553**, not the 0.406 / 0.520 from v4's own
self-report. Every kit run came in below those corrected numbers. See
[`../V4_VS_KIT_APPLES_TO_APPLES.md`](../V4_VS_KIT_APPLES_TO_APPLES.md)
for the full diagnostic.

## What v6.1 expects to show

If the source-bundle hypothesis is correct, v6.1 should hit
**~0.45 ± 0.02 AP** on the simplified test set (matching v4's
kit-eval number). If it does, the kit pivot is genuinely validated.
If it doesn't, there's another factor we haven't found yet.

## Quick start

```bash
docker run --gpus=all -it --rm \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/home/joncrall/data/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/home/joncrall/data/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/shitspotter_v4:/data/joncrall/shitspotter_v4:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v6_1/run.sh
```

Or through the driver:

```bash
./reproduce/mobile_quality_push.sh build         # if image not current
docker run --gpus=all -it --rm \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/home/joncrall/data/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/home/joncrall/data/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/shitspotter_v4:/data/joncrall/shitspotter_v4:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v6_1/run.sh 2>&1 | tee /tmp/v6_1.log
```

Expected wall-clock: ~3 GPU-hours on a single RTX 3090 (same as v6.0).

## What v6.1 keeps from v6.0

- Same kit version (any post-`b5aeaec`)
- Same hyperparameters (80 epochs, batch=48, lr=7.5e-4, etc.)
- Same COCO-pretrained init (`deimv2_pico_coco.pth`)
- Same `score_thresh=0.001` for eval (via the kit's `31836c5` fix)
- Same v9 simplified test set as the eval target

## What v6.1 changes from v6.0

Exactly one thing: the source bundle in `run.sh`.

```diff
- RAW_TRAIN=$RAW_DPATH/simplified_train_imgs7350_4f0174d0.kwcoco.zip
- RAW_VALI=$RAW_DPATH/simplified_vali_imgs1258_07ec447d.kwcoco.zip
+ RAW_TRAIN=$RAW_DPATH/train_imgs10671_b277c63d.kwcoco.zip
+ RAW_VALI=$RAW_DPATH/vali_imgs1258_577e331c.kwcoco.zip
```

And the workspace is `/data/joncrall/kcd/v6_1/` so v6.0's outputs are
preserved for comparison.

## After the run

Fill in `EVAL.md` with:
- v6.1 test AP via `./reproduce/mobile_quality_push.sh compare`
- v6.1 wall-clock for the `experiments/COMPUTE_COST.md` table

If v6.1 lands at ~0.45 AP, the data-bundle hypothesis is confirmed and
we proceed to **v7.1** (multiscale + corrected bundle, both cells). If
not, we have a different question to investigate.
