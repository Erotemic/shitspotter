# mobile_app_training_v10 — best-of-everything ship candidates

v10 combines whatever worked in v7-v9 into a single recipe per cell,
trained for longer with EMA, and freezes the ship candidates.

| Cell        | v4 baseline | v10 final | Δ      | Status                   |
|-------------|-------------|-----------|--------|--------------------------|
| pico@416    | 0.406       | 0.478     | +0.072 | SHIP (clears +5 AP bar)  |
| n@640       | 0.520       | 0.511     | -0.009 | below v4 (not a winner)  |

v10 final = `test_ap` (AP@0.5, kwcoco detect_metrics) from
`/media/joncrall/flash1/kcd-ssd/v10/manifest.json` (run completed 2026-06-01
10:38 UTC). Both cells are desktop-eligible (pico p50 14.1 ms, n@640 p50
39.7 ms; budget 80 ms) and classed `HOST_PROMISING` — the Pixel 5 on-device
bench has not run yet (`device_eligible: TODO`), which is the remaining gate
before pico@416 is ship-ready.

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

## Quick start

v10 reads tiles via the default kwcoco_jpeg path on SSD-backed
storage. The SSD lives at `/media/joncrall/flash1/kcd-ssd/`. On
warm SSD/NVMe, kwcoco_jpeg beats WebDataset by ~2x per the
cross-storage bench (see "Storage tier" below). There's no shard
pre-build step on this path.

```bash
# 0. Copy v6.1's tile bundles from the HDD-backed staging to the SSD.
#    One-time step; the bundles are ~few GB combined. Run as the user
#    that owns the SSD mount (no docker needed for the rsync itself).
mkdir -p /media/joncrall/flash1/kcd-ssd/v6_1
rsync -aP /data/joncrall/kcd/v6_1/data /media/joncrall/flash1/kcd-ssd/v6_1/

# 1. Dry-run (validates recipe + prints resolved sweep_data; no GPU):
bash experiments/mobile_app_training_v10/run.sh --dry_run

# 2. Real run. Script self-wraps shitspotter:latest with all required
#    mounts (GPU, shm, DVC, V4 pretrained, SSD workspace, live
#    shitspotter source for recipe edits).
bash experiments/mobile_app_training_v10/run.sh

# Force re-train of completed cells:
bash experiments/mobile_app_training_v10/run.sh --force_train
```

If the image isn't built yet:

```bash
bash reproduce/mobile_quality_push.sh build
```

Override knobs (see `run.sh` header for the full list): `KCD_SSD_DPATH`
points elsewhere if your SSD isn't at `/media/joncrall/flash1/kcd-ssd/`;
`SHITSPOTTER_IMAGE` for a custom image tag; `SKIP_DOCKER=1` to skip the
docker wrap entirely.

### Storage tier (why kwcoco_jpeg on SSD)

Per kwcoco_dataloader's `2026-05-29_ssd_cross_storage.md` journal,
WebDataset is a **storage strategy, not a raw-throughput strategy**:

- Cold rotational HDD with adequate shards: WDS ~1.7× baseline (wins).
- Warm SSD / NVMe: WDS ~0.5× baseline (loses — the WDS path has a
  ~1.5× per-sample CPU overhead that doesn't pay back without the
  sequential-read win).
- Small datasets (<~5K samples): WDS parallelism caps at shard
  count via `split_by_worker`. Shitspotter's ~53K-tile train
  bundle yields ~20 shards across the two buckets (poop +
  &lt;empty&gt;), which is adequate — only relevant if you fall
  back to the HDD path.

v10 stages v6.1's bundle on SSD, so kwcoco_jpeg wins. If you ever
need to run from rotational storage, the optional
`00_build_wds_shards.sh` builds the WDS shard tree and the recipe
flips back to `tile_store: webdataset` + `train_wds_shards: <path>`
(see ADR-0001 in the kit).

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
