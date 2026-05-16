# mobile_app_training_v7 — multi-scale training

Same model, same training data, **multi-scale resolution policy** during
training. Both target cells.

| Cell        | v4 fixed AP | v7 multiscale AP | Δ |
|-------------|-------------|------------------|---|
| pico@416    | 0.406       | TBD              | TBD |
| n@640       | 0.520       | TBD              | TBD |

## Why

v4 ran the `*_fixed_*` rows of the matrix but not the matching
`*_multiscale_*` rows — the latter are still `NOT_READY` in
`/data/joncrall/shitspotter_v4/manifest.tsv`. There's free AP sitting
there; v7 collects it.

## Quick start (inside the docker image)

```bash
docker run --gpus=all -it --rm \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v7/run.sh
```

`run.sh` is the same shape as v6's — it reuses v6's tiled bundles
(symlinked or already-present at `/data/joncrall/kcd/v6/data/`) and
hands the recipe to `kwcoco-detector-kit recipe-run`.

## Prerequisites

- v6 done and validated (the tile bundles under
  `/data/joncrall/kcd/v6/data/` are the input). If v6 hasn't run,
  v7's `run.sh` will tile from scratch into `/data/joncrall/kcd/v7/data/`
  instead.

## Success criterion

Either cell beating the v4 fixed-policy baseline by **at least +1 AP**.
Both cells beating it is the goal but not required to proceed; we
keep whichever cells improve and roll those forward into v8.
