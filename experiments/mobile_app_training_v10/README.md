# mobile_app_training_v10 — best-of-everything ship candidates

v10 combines whatever worked in v7-v9 into a single recipe per cell,
trained for longer with EMA, and freezes the ship candidates.

| Cell        | v4 baseline | v10 final | Δ      | Status   |
|-------------|-------------|-----------|--------|----------|
| pico@416    | 0.406       | TBD       | TBD    | TBD      |
| n@640       | 0.520       | TBD       | TBD    | TBD      |

## What goes into the v10 recipe

Fill in `recipe.yaml` **after** v9 is evaluated, copying:

- The winning `train_policy` (multiscale vs fixed) from v7's EVAL.md
- The winning distillation training data path from v9's EVAL.md
- The winning round-count from v8's EVAL.md (if mining helped, run
  v10 as one more round on top of the v9-distilled checkpoint)
- 1.5–2× the epoch count of the best round
- EMA toggled on (the kit's deimv2 trainer exposes this in
  `_train_deimv2_variant.sh`; mirror via a recipe knob if added)

There is no v10 recipe.yaml committed yet — it's intentionally written
after the upstream cells' EVAL.md files are filled in. That keeps v10
honest: it's a synthesis, not a prediction.

## Quick start (inside the docker image)

v10 is the first shitspotter recipe to opt into the kit's
WebDataset training-input path (see kwcoco-detector-kit ADR-0001).
That requires a one-time data-prep step to produce the shard tree
from v6.1's train bundle. Then the recipe + sweep run as usual.

```bash
# 0. Build the WDS shards from v6.1's train bundle (one-time;
#    idempotent; FORCE_RESHARD=1 to rebuild).
#
#    Runs on the host. The script self-wraps in shitspotter:latest
#    because /data/joncrall/kcd/v6_1/ is root-owned (created by an
#    earlier in-container run), so the host user can't write to it
#    directly. Override the image with SHITSPOTTER_IMAGE=... ;
#    skip the wrap with SKIP_DOCKER=1 if you have host write access.
bash experiments/mobile_app_training_v10/00_build_wds_shards.sh

# 1. Run the recipe (after recipe.yaml is filled in).
docker run --gpus=all -it --rm \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v10/run.sh
```

Step 0 needs `kwcoco_dataloader` on dev/0.1.3 or later installed
in the `shitspotter:latest` image. The shitspotter Dockerfile
(`dockerfiles/shitspotter.dockerfile`) installs it from the kit's
submodule `tpl/kwcoco_dataloader` (which `setup_staging.py` pulls
via `recurse_submodules: true`). If you haven't rebuilt your image
since the dev/0.1.3 merge landed, do so first:

```bash
bash reproduce/mobile_quality_push.sh build
```

The script fails fast with a clear "rebuild the image" message if
the import is missing.

The shards land under `$(dirname $TRAIN_KWCOCO)/shards/` by
default; the recipe just points `data.train_wds_shards` at that
directory.

### When does WDS actually win?

Per the kwcoco_dataloader cross-storage bench (journal entry
2026-05-29_ssd_cross_storage.md): WebDataset is a **storage
strategy, not a raw-throughput strategy**.

- Cold rotational HDD with adequate shards: ~1.7× baseline (wins).
- Warm SSD / NVMe: ~0.5× baseline (loses — the WDS path has a
  ~1.5× per-sample CPU overhead that doesn't pay back without the
  sequential-read win).
- Small datasets (<~5K samples): WDS parallelism caps at shard
  count via `split_by_worker`. Shitspotter's ~53K-tile train
  bundle yields ~20 shards across the two buckets (poop +
  &lt;empty&gt;), which is adequate.

Before opting v10 into webdataset, check whether
`/data/joncrall/kcd/v6_1/data/train_tile_g2.kwcoco.zip` lives on
spinning storage. If it's already on a warm SSD, leave
`tile_store: kwcoco_jpeg` (the default) — webdataset won't help.

## Success criterion

Either cell beating its v4 number by **at least +5 AP** is the ship
gate. If only one cell hits +5, ship that one and downgrade the other
to "future work."

## Deliverables for the ship cut

After the v10 numbers land:

1. Copy the winning ONNX into `tpl/shitspotter-phone-app/`'s
   model assets dir per the phone app's `006_adding_a_new_model.md`.
2. Update the phone app's `ModelRegistry` (`DEIMV2_PICO_416` /
   `DEIMV2_N_640`) to reference the new ONNX. The
   modelspec sidecar from the kit's export already has the right
   shape and postprocess params.
3. Commit the EVAL.md, the modelspec, and the recipe in one commit.
4. Tag the shitspotter repo with `mobile-v10-{cell}-{date}`.
