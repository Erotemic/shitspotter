# mobile_app_training_v7_1 — multiscale on the corrected bundle

v7.1 stacks v7's multi-scale training policy on top of v6.1's
validated, corrected-source-bundle tile pool. Both cells (pico@416 and
n@640). No re-tiling; reuses v6.1's tile bundles directly.

## Hypothesis

v7's multi-scale policy added +0.012 AP over v6.0 on pico@416. If that
gain stacks linearly on v6.1's +0.035 from the corrected bundle:

| Step | pico@416 expected | n@640 expected |
|------|-------------------|----------------|
| v6.1 measured     | 0.421             | — (first run)  |
| +0.012 multiscale | 0.433             | (analogous lift over v4) |
| v4 baseline       | 0.4548            | 0.5553         |

A v7.1 pico@416 result in the 0.43–0.44 range puts us solidly within
DETR run-to-run noise of v4. n@640 is a first measurement on the
corrected bundle — we expect it to come in at v4-comparable too.

## What v7.1 keeps from v7.0

- Multi-scale train policies (`multiscale_320_512` for pico,
  `multiscale_512_768` for n).
- Per-cell COCO-pretrained init checkpoint.
- 80 epochs, AMP on, EMA on (DEIMv2 defaults).

## What v7.1 changes from v7.0

Exactly one thing: source bundle.

```diff
- train_kwcoco: /data/joncrall/kcd/v6/data/train_tile_g2.kwcoco.zip     # 12,820 tiles (wrong bundle)
+ train_kwcoco: /data/joncrall/kcd/v6_1/data/train_tile_g2.kwcoco.zip   # ~53K tiles (corrected, == v4)
```

Workspace is `/data/joncrall/kcd/v7_1/` so v7.0's outputs are
preserved.

## Quick start (inside the docker image)

```bash
# Assumes v6.1 has already run (its tile bundles are at
# /data/joncrall/kcd/v6_1/data/).
docker run --gpus=all -it --rm \
    --shm-size=32g \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/home/joncrall/data/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/home/joncrall/data/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/shitspotter_v4:/data/joncrall/shitspotter_v4:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v7_1/run.sh 2>&1 | tee /tmp/v7_1.log
```

Expected wall-clock: ~8 GPU-hours on a single 3090 (pico ~3 h +
n@640 ~5 h, sequential). Both checkpoints get auto-stamped provenance
in their `policy.json` and (now with kit `059f60c`) in their
`detect_metrics.json`.
