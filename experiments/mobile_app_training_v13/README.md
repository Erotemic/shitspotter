# mobile_app_training_v13 — multi-scale tiled detection (train + on-phone pyramid)

Goal: make the detector robust across **apparent object scale**, because the
real use case is a person (or robot) walking around a yard — a poop is tiny
when far away and large when close, and we must **never drop the detection as
the camera moves**. This needs (a) a model trained on objects at many apparent
sizes, and (b) a phone inference path that feeds the model crops where the poop
is actually resolvable, instead of one 640² downscale of a 4032×3024 frame.

## The key constraint that reframes everything

**pico's input size is fixed.** DEIMv2's HGNetv2 variants emit
`base_size_repeat=None` and "do NOT support dynamic input" (see
`kwcoco_detector_kit/trainers/deimv2.py`). That is why v10/v11's
`multiscale_512_768` policy silently collapsed to fixed-640 — the network-input
multiscale jitter is a no-op for pico. So **multi-scale robustness must come
from the training DATA (tiles/crops at many scales), not from input-size
jitter.** Conveniently, that is exactly what the phone will feed at runtime, so
train-time and inference-time scale distributions match.

(Side note on the v11 distill result: 0.552 vs 0.588 is a regression, but a
full-weight pseudo-GT merge is a plausible misconfig, so treat it as "not a
win," not a proven dead end. It is parked, not refuted.)

## We have the tooling already

`kwcoco-detector-kit tile` (`data/tile.py`) supports three modes:
- **`multiscale`** — fixed-size tiles cut from N downscaled source copies
  (`source_scales` default `1.0,0.66,0.40,0.25`). The *same object appears at
  different apparent sizes* across tiles. This is the far→near objective.
- **`quadrant`** — NxN overlapping full-res crops (`tile_grid`, `tile_overlap`),
  each resized to `tile_output_dim`; `keep_full` also emits the downsized whole
  frame. This mirrors the phone's grid passes + the coarse whole-frame pass.
- **`full_only`** — whole image resized (the 1×1 letterbox fallback).

The VIAME sealions project already drives `mode=multiscale` at scale
(`projects/viame_sealions_2026/scripts/_launch_tiles.sh`). NB: sealions found
*input-size* multiscale hurt (OOM/instability → fixed 640); that is a different
axis from our *data-side* multiscale, which is what we want.

**Generic composer (added to the kit, not duplicated here):** the `tile` CLI
emits one mode per call, so composing a multi-pass corpus was project-specific
shell. That is now a reusable kit op — **`kwcoco-detector-kit tile-corpus`**
(`kwcoco_detector_kit/data/tile_corpus.py`) — which runs an ordered list of
passes from a spec and unions them. shitspotter and sealions both invoke it
with their own [corpus_spec.yaml](corpus_spec.yaml); no tiling code lives in
this experiment dir.

## Training-data plan (v13 bundle)

Build a training bundle whose scale distribution matches the phone pyramid, by
unioning tile sets from the raw 4032×3024 annotated images
(`/data/joncrall/dvc-repos/shitspotter_dvc/train_imgs10671_b277c63d.kwcoco.zip`):

1. **`multiscale`** tiles, `tile_size = model_input` (640, then 768),
   `source_scales=1.0,0.66,0.40,0.25` — the core far→near continuum. At scale
   1.0 these are high-res crops (dense/close view); at 0.25 they approach a
   wide field of view. `keep_negative=True` for hard negatives + balancing.
2. **`quadrant` grid=2**, `keep_full=True` — explicit 2×2 crops + the downsized
   whole frame (matches the phone's L2 + coarse passes). This is the existing
   v6.1 `train_tile_g2` recipe; reuse it.
3. (optional) **`quadrant` grid=4** — denser crops so the phone's fine pass is
   in-distribution without going to a 64-tile build at train time.

These passes are declared in [corpus_spec.yaml](corpus_spec.yaml) and composed
into one bundle by the generic `tile-corpus` builder. The corpus build
([build_corpus.sh](build_corpus.sh)) and training ([run.sh](run.sh)) are
**separate steps**, each runnable in or out of the container — no bespoke
tiling script.

Because every emitted tile is resized to the fixed model input, the model only
ever sees its native input size — the *scale variety is in the pixels*, which
is what an FPN-bearing, scale-**covariant** (not invariant) detector needs.

## Resolution

- Train **pico@640** on the multi-scale bundle first (proven-stable input).
- Then **pico@768** (`recipe.yaml`), tile_size matched to 768. Desktop ~47 ms,
  Pixel 5 ~2 FPS (est.) — still inside budget.
- **1024**: likely too much for pico's attention/memory and ~1 FPS on a Pixel 5
  (640→1024 ≈ 2.56× tokens). Treat as a desktop-only curiosity, not a ship
  target. **768 is the intended ceiling.**
- We also have FPS headroom for a **larger architecture** (deimv2_n, maybe
  deimv2_s) at 640 — n/s have *not* been retrained in the v6–v12 push (only
  pico@416 + n@640-v4). A higher-capacity arm is a parallel lever, but v13's
  focus is scale, not capacity.

## On-phone tiled-pyramid inference (consumer-app plan)

Full design in the app repo:
[`tpl/scatspotter_app/docs/007_tiled_pyramid_inference.md`](../../tpl/scatspotter_app/docs/007_tiled_pyramid_inference.md).
Summary — a user-selectable **compute lever** with progressively finer tilings,
results fused across passes:

| Level | Tiling of the captured frame | Effective resolution | Cost |
|-------|------------------------------|----------------------|------|
| L0 | 1×1, letterboxed (current) | lowest | cheapest (tight-resource fallback) |
| L1 | 1×2 overlapping near-square crops covering the full frame (no letterbox) | low | cheap |
| L2 | 2×2 overlapping grid, each cell → model input | medium | moderate |
| L3 | 8×8 grid, each cell → model input | high | expensive |

- Each tile's boxes are mapped back to full-frame coords and **fused** across
  passes (NMS / weighted-box-fusion on overlaps).
- The user picks which levels run (e.g. L0+L3, skipping the middle); levels
  compose.
- Future (not v1): run the expensive pass only every few seconds while the
  cheap pass runs every frame.

## Eval

- Score at each pyramid level + fused, on the v9 simplified test GT.
- **Size-stratified AP** is the headline (small poops are the point) — reuse
  `experiments/size_stratified_eval.py`.
- Sanity: a "move closer/farther" check — the same poop should stay detected as
  the crop scale changes.

## Build & run — everything in the container, as separate steps

All host work runs inside the image; the default is **rebuild the image** (code
baked in), not bind-mount code. `reproduce/in_docker.sh` is a transparent docker
prefix — drop it to run the same command on the host.

```bash
# 0. rebuild the image so the current kit (tile-corpus + fixes) is baked in
reproduce/mobile_quality_push.sh build

# 1. build the multi-scale corpus  (SEPARATE step; CPU; the new tile-corpus op)
reproduce/in_docker.sh bash experiments/mobile_app_training_v13/build_corpus.sh

# 2. train pico@768 on the corpus  (SEPARATE step; GPU). Run inside tmux.
reproduce/in_docker.sh bash experiments/mobile_app_training_v13/run.sh
```

Runs foreground (`-it`) — persist it with tmux. (`DETACH=1 NAME=v13` is
available for non-tmux/CI use, then `docker logs -f v13`.) Run any piece on the
host instead by dropping the `reproduce/in_docker.sh` prefix. Set-and-forget
(one container, both steps) is just:
`reproduce/in_docker.sh bash -lc 'experiments/mobile_app_training_v13/build_corpus.sh && experiments/mobile_app_training_v13/run.sh'`.

## Status / next steps

- [x] Generic `tile-corpus` builder added to the kit (reusable by any project).
- [x] `corpus_spec.yaml` + separate `build_corpus.sh` / `run.sh` steps.
- [x] `reproduce/in_docker.sh` generic docker-prefix wrapper.
- [ ] Rebuild image → step 1 (corpus) → step 2 (train pico@768). (Try pico@640
      by swapping the three sizes in corpus_spec.yaml + recipe input_hw.)
- [ ] App: implement L1/L2/L3 tiling + box fusion behind a settings lever
      (see `tpl/scatspotter_app/docs/007_tiled_pyramid_inference.md`).
- [ ] Eval per-level + size-stratified; pick the ship configuration. A tiled-
      inference *eval* path is itself a candidate generic kit op (future).
