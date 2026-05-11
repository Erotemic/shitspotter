# mobile_app_training_v4 — DEIMv2 detectors for the Pixel 5 phone app

This experiment family trains and exports detectors with **live on-device
inference on a Pixel 5** as the deploy target.

The goal is to produce ONNX detectors that the new KMP+Compose phone app
([`tpl/shitspotter-phone-app/`](../../tpl/shitspotter-phone-app/)) can
swap into its model registry, while keeping a clear path back to the
v9 OpenGroundingDINO teacher in
[`experiments/foundation_detseg_v3/`](../foundation_detseg_v3/).

See [DESIGN.md](DESIGN.md) for the full rationale and the lessons we are
*choosing not to relearn*. This README is the operational quick-start.

---

## Models and roles

| Variant     | Role                                  | Default input | Notes                                           |
|-------------|---------------------------------------|---------------|-------------------------------------------------|
| `deimv2_n`  | **Primary Pixel-5 candidate**         | 320 × 320     | HGNetv2-N, ~3.6 M params, ~6.8 GFLOPs.          |
| `deimv2_pico` | Speed fallback                      | 320 × 320     | HGNetv2-Pico, ~1.5 M params, ~5.2 GFLOPs.       |
| `deimv2_s`  | Quality reference / future teacher    | 640 × 640     | DINOv3-backed ViT-Tiny. Not the live model.     |

Train order (in priority): **N first, Pico second, S last**.

The v9 OpenGroundingDINO + tuned SAM2 package is treated as a frozen
**teacher / quality reference**, not the deploy target. v9 is a 0.766 AP
detector at full resolution; the phone-deploy story does not need to
match that number on-device, only to come close enough that the live
pipeline is usable.

---

## Inputs we depend on

* **Splits** (read-only, on the DVC root)
  * `train_imgs10671_b277c63d.kwcoco.zip`
  * `vali_imgs1258_577e331c.kwcoco.zip`
  * `test_imgs121_d39956b1.kwcoco.zip`
* **DEIMv2 repo**: `tpl/DEIMv2/` (submodule). Pretrained COCO detector
  checkpoints are downloaded by `00_setup.sh` into
  `$V4_ROOT/pretrained/deimv2/`.
* **DINOv3 distilled backbones** (`vitt_distill.pt`, `vittplus_distill.pt`)
  required by the `deimv2_s` / `deimv2_m` configs. Same downloader.
* **Teacher (optional, evaluation only)**:
  `experiments/foundation_detseg_v3/packages/v9_opengroundingdino_sam2_1_hiera_base_plus_tuned.yaml`.

All derived artifacts go under `$V4_ROOT` (defaults to
`$HOME/data/shitspotter_v4`). The DVC roots stay read-only — see
[`dev/journals/lessons_learned.md`](../../dev/journals/lessons_learned.md)
for the reason.

---

## Coarse-to-fine training data

`01_make_tile_augmented_kwcoco.sh` builds a kwcoco bundle that mixes:

* the **full image** resized so the long side ≤ `V4_RESIZE_MAX_DIM`
  (default 1280) — feeds the `BALANCED_FULL_FRAME` phone-app mode;
* **2 × 2 overlapping tiles** cut from the full-resolution image, each
  resized so the long side ≤ `V4_TILE_OUTPUT_DIM` (default 640) — feeds
  the `ROI_HIGH_RES` and `TILED_SEARCH` phone-app modes.

Annotations (xywh boxes today, polygons next) are warped into the tile
coordinate frame and dropped if the visible fraction falls below
`min_keep_fraction` (default 0.30) so we don't teach the detector that a
half-poop is a full poop.

This single bundle is the training input for **all three** v4 variants —
they share the same dataset but train at different input sizes.

---

## Quick start

The whole pipeline assumes the `shitspotter` Python env is active and
the DEIMv2 submodule is initialised.

```bash
# 0. one-time setup: install pretrained COCO checkpoints, smoke-test env
bash experiments/mobile_app_training_v4/00_setup.sh

# 1. tile-augmented kwcoco (cheap, CPU only — runs fine in the VM)
bash experiments/mobile_app_training_v4/01_make_tile_augmented_kwcoco.sh

# 2. fine-tune the three variants — RUN ON THE HOST GPU MACHINE.
#    Each script honors V4_INPUT_HW / V4_TRAIN_BATCH / V4_NUM_EPOCHS.
bash experiments/mobile_app_training_v4/02_train_deimv2_n.sh
bash experiments/mobile_app_training_v4/02_train_deimv2_pico.sh
bash experiments/mobile_app_training_v4/02_train_deimv2_s.sh

# 3. ONNX export per variant. Defaults pick the variant's default input
#    size; override with positional H W.
bash experiments/mobile_app_training_v4/03_export_onnx.sh deimv2_n
bash experiments/mobile_app_training_v4/03_export_onnx.sh deimv2_pico
bash experiments/mobile_app_training_v4/03_export_onnx.sh deimv2_s

# 4. Evaluate on the v9-canonical simplified test split (compare
#    directly to the 0.766 v9 AP).
bash experiments/mobile_app_training_v4/04_eval_on_test.sh deimv2_n
bash experiments/mobile_app_training_v4/04_eval_on_test.sh deimv2_pico
bash experiments/mobile_app_training_v4/04_eval_on_test.sh deimv2_s

# 5. Sanity-check that ONNX matches the .pth on a still image.
python experiments/mobile_app_training_v4/05_desktop_onnx_parity.py \
    --pth_ckpt   $V4_ROOT/runs/deimv2_n_tile_g2_320x320/best_stg2.pth \
    --pth_config $V4_ROOT/runs/deimv2_n_tile_g2_320x320/generated_configs/train.yml \
    --onnx       $V4_ROOT/runs/deimv2_n_tile_g2_320x320/export/deimv2_n_h320_w320.onnx \
    --image      $SHITSPOTTER_DPATH/tpl/poop_models/dog.jpg

# 6. Bench the ONNX on a desktop CPU as an early signal.
python experiments/mobile_app_training_v4/06_benchmark_onnx_desktop.py \
    --onnx $V4_ROOT/runs/deimv2_n_tile_g2_320x320/export/deimv2_n_h320_w320.onnx \
    --image $SHITSPOTTER_DPATH/tpl/poop_models/dog.jpg \
    --warmup 5 --iters 50

# 7. Wire the export into the phone app — see 07_register_in_phone_app.md.
```

### Knobs you will reach for

| Variable                | Default                  | Purpose                                                |
|-------------------------|--------------------------|--------------------------------------------------------|
| `V4_ROOT`               | `~/data/shitspotter_v4`  | Writable workspace for all v4 artifacts.               |
| `V4_RESIZE_MAX_DIM`     | `1280`                   | Long-side cap on the kept full-frame image.            |
| `V4_TILE_GRID`          | `2`                      | NxN tile grid per image.                               |
| `V4_TILE_OVERLAP`       | `0.20`                   | Fractional overlap between adjacent tiles.             |
| `V4_TILE_OUTPUT_DIM`    | `640`                    | Long-side cap on each tile after resize.               |
| `V4_INPUT_HW`           | `"320 320"` (N/Pico)     | Model input size; override per training script.        |
| `V4_TRAIN_BATCH`        | `128` (N/Pico), `32` (S) | Total training batch size.                             |
| `V4_NUM_EPOCHS`         | `60`/`80`/`30`           | N / Pico / S epoch budget.                             |
| `V4_NUM_GPUS`           | `1`                      | Multi-GPU via `torch.distributed.run` when > 1.        |
| `FORCE_RETRAIN`         | `0`                      | Bypass the "checkpoint already exists" short-circuit.  |
| `FORCE_TILE_REBUILD`    | `False`                  | Re-run `01_*.sh` even if outputs exist.                |
| `FORCE_REPRED`          | `0`                      | Re-run prediction in `04_*.sh` even if pred exists.    |
| `DOWNLOAD_PRETRAINED`   | `True`                   | Skip the gdown step in `00_setup.sh`.                  |

---

## Where to run what

| Stage                          | VM | Host GPU box | Phone |
|--------------------------------|----|--------------|-------|
| `00_setup.sh`                  | ✓  | ✓            |       |
| `01_make_tile_augmented_*.sh`  | ✓ (CPU-bound) | ✓ |        |
| `02_train_*.sh`                |    | ✓            |       |
| `03_export_onnx.sh`            | ✓ (CPU OK) | ✓     |       |
| `04_eval_on_test.sh`           |    | ✓ (GPU recommended) |    |
| `05_desktop_onnx_parity.py`    | ✓  | ✓            |       |
| `06_benchmark_onnx_desktop.py` | ✓  | ✓            |       |
| `07_register_in_phone_app.md`  |    | ✓            | ✓     |

The VM has no GPU, so the heavy training and the GPU-backed
detector eval need to run on the host machine. Everything else is fine
inside the VM.

---

## Comparison target

The v9 OpenGroundingDINO + tuned SAM2 package reaches **AP = 0.766** on
the simplified test GT (IoU = 0.5). v4 variants are *expected* to land
below that — DEIMv2-N has 1/100th the parameters of OpenGroundingDINO
+ DINOv2 — but they need to be close enough to be useful live, while
running 10–30× faster.

Numbers from `04_eval_on_test.sh` are recorded in
`$V4_ROOT/eval/<variant>_<tag>_<HxW>/eval/detect_metrics.json` and are
read by the same `nocls_measures.ap` selector v9 uses, so the comparison
is apples-to-apples.

---

## Coordination with other agents

* The Codex agent also touches this repo. Updates to this directory
  must mention the change in `CHANGELOG.md` and add a journal entry
  under `dev/journals/` per
  [`AGENTS.md`](../../AGENTS.md).
* Hard invariants discovered during this work belong in
  `dev/benchmark-candidates/` (see the existing
  `app-deployment-questions.md`).
* The phone-app side is read in this repo via `tpl/shitspotter-phone-app/`
  (in-tree, not a submodule, by deliberate exception). All proposed
  changes there live in the prescriptive
  [`07_register_in_phone_app.md`](07_register_in_phone_app.md) — we do
  not modify Kotlin from this experiment; the next agent does.

---

## What this directory deliberately does NOT do

* It does **not** retrain SAM2. The v4 detectors are evaluated against
  the same zero-shot `sam2.1_hiera_base_plus` checkpoint as v9, since
  the question is about the detector head, not the segmenter.
* It does **not** modify
  `shitspotter/algo_foundation_v3/`. The v4 path bypasses
  `cli_train.py` and calls DEIMv2's own `train.py` directly with a
  generated config so the v3 stack stays a stable artifact.
* It does **not** modify the in-tree phone app code; that change is
  prescribed in step 7 and belongs to a follow-up host-side commit.

If you find yourself reaching for the v3 codebase or the phone-app
Kotlin, stop and ask — those changes belong in their own commits with
their own journal entries.
