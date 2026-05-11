# mobile_app_training_v4 — DEIMv2 detectors for the Pixel 5 phone app

This experiment family solves a constrained optimization problem:

```text
maximize:  detection quality on the v9 simplified test GT
subject to: Pixel 5 live detection >= 10 FPS in the intended mode,
            local-only inference, fixed-shape ONNX export, app preview
            stays responsive, failure-case capture still works.
```

Resolution is a **first-class axis** alongside architecture. We train and
evaluate a Pareto frontier over (variant × export resolution × train
resolution policy), exporting one ONNX per cell. The phone app picks
the highest-quality candidate that meets its on-device gates — see
[`07_register_in_phone_app.md`](07_register_in_phone_app.md).

See [DESIGN.md](DESIGN.md) for the full rationale and the lessons we are
*choosing not to relearn*. This README is the operational quick-start.

---

## Candidate matrix

A "candidate" is uniquely identified by
`(variant, export_h, export_w, train_resolution_policy, tile_training_policy)`.
The default sweep walks one row per cell:

| variant       | export size | train policy        | role                              |
|---------------|-------------|---------------------|-----------------------------------|
| `deimv2_n`    | 320×320     | `multiscale_256_416`| primary phone candidate, cheapest |
| `deimv2_n`    | 416×416     | `multiscale_320_512`| likely Pareto winner              |
| `deimv2_pico` | 320×320     | `multiscale_256_416`| speed fallback                    |
| `deimv2_pico` | 416×416     | `multiscale_320_512`| higher-res fallback               |
| `deimv2_n`    | 512×512     | `multiscale_384_640`| if N has speed headroom           |
| `deimv2_pico` | 512×512     | `multiscale_384_640`| higher-res small model            |
| `deimv2_n`    | 640×640     | `multiscale_512_768`| upper bound for N                 |
| `deimv2_s`    | 640×640     | `multiscale_512_768`| quality reference / teacher       |

Order is **cheapest-first** so a sweep that already hits the FPS gate
can be stopped before training the heavier rows. Each cell becomes its
own ONNX, its own modelspec sidecar, and its own `ModelSpec` entry in
the phone app.

The v9 OpenGroundingDINO + tuned SAM2 package is treated as a frozen
**teacher / quality ceiling**, not the deploy target. v9 reaches
AP=0.766 on the simplified test GT at full resolution.

### Train vs. export resolution

These are deliberately separate fields on every candidate:

```text
export resolution         fixed, explicit, benchmarked, exported to ONNX
train resolution policy   may include random multi-scale resize augmentation
```

`train_resolution_policy` strings the scripts accept:

| String                 | DEIMv2 collate                     | Scales seen during training            |
|------------------------|------------------------------------|----------------------------------------|
| `fixed`                | `base_size_repeat=None`            | single scale = export size             |
| `multiscale`           | base=max(H,W), repeat=12           | ±25% band around export size, 32-px    |
| `multiscale_<S>`       | base=`<S>`, repeat=12              | ±25% band centered on `<S>`            |
| `multiscale_<lo>_<hi>` | base picked so band ≈ [lo, hi]     | band targeting `[lo, hi]`              |

The trainer dumps the *resolved effective config* and the *exact scale
list it will sample from* to `<workdir>/policy.json` and
`<workdir>/resolved_effective_config.yml` after generation so you can
verify what DEIMv2 will actually do — not just what we asked for.

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

## Quick start (recommended path: sweep)

The whole pipeline assumes the `shitspotter` Python env is active and
the DEIMv2 submodule is initialised.

```bash
# 0a. Once per shell: export every env var the v4 scripts read. Override
#     anything by exporting it BEFORE the source line, e.g.
#         V4_ROOT=/scratch/v4 PYTHON_BIN=python3 source ...setup_env.sh
cd ~/code/shitspotter   # adjust to wherever you cloned the repo
source experiments/mobile_app_training_v4/setup_env.sh

# 0b. One-time setup: install pretrained COCO checkpoints, smoke-test env
bash experiments/mobile_app_training_v4/00_setup.sh

# 1. Tile-augmented kwcoco (cheap, CPU only — runs fine in the VM)
bash experiments/mobile_app_training_v4/01_make_tile_augmented_kwcoco.sh

# 2. Sweep — train + export + eval + bench every cell of the candidate
#    matrix. Cheapest-first; pass V4_SWEEP_KEEP_GOING=1 to continue past
#    failed cells. RUN ON THE HOST GPU MACHINE.
bash experiments/mobile_app_training_v4/02_sweep.sh

# 3. Aggregate the per-cell outputs into a single eligibility manifest
#    and select the highest-quality candidate that satisfies the
#    desktop CPU latency proxy gate.
"$PYTHON_BIN" experiments/mobile_app_training_v4/eligibility_manifest.py \
    --auto \
    --max_desktop_ms 80 \
    --out "$V4_ROOT/manifest.tsv" \
    --out_json "$V4_ROOT/manifest.json"
```

The manifest TSV has one row per candidate with all the columns the
phone-app integration step needs (`candidate_id`, `variant`,
`export_input_h/w`, `train_resolution_policy`, `train_resolution_min/max/choices`,
`onnx_path`, `modelspec_path`, `test_ap_simplified`,
`desktop_latency_ms_p50/mean/p99`, `pixel5_*`, `phone_model_id`).
`pixel5_*` columns read `TODO` until a host-side run fills them in.

## Quick start (fine-grained: per-variant)

If you want to investigate one cell in isolation:

```bash
# train one specific cell — these defaults match the per-variant
# entrypoints: V4_INPUT_HW + V4_TRAIN_POLICY + the cheap-first epoch
# budget already encoded in 02_train_*.sh.
bash experiments/mobile_app_training_v4/02_train_deimv2_n.sh
bash experiments/mobile_app_training_v4/02_train_deimv2_pico.sh
bash experiments/mobile_app_training_v4/02_train_deimv2_s.sh

# Override export resolution and/or train policy:
V4_INPUT_HW="416 416" V4_TRAIN_POLICY=multiscale_320_512 \
    bash experiments/mobile_app_training_v4/02_train_deimv2_n.sh

# Then per cell:
bash experiments/mobile_app_training_v4/03_export_onnx.sh deimv2_n tile_g2_multiscale_320_512 416 416
bash experiments/mobile_app_training_v4/04_eval_on_test.sh deimv2_n tile_g2_multiscale_320_512 416 416

# Sanity-check that ONNX matches the .pth on a still image.
"$PYTHON_BIN" experiments/mobile_app_training_v4/05_desktop_onnx_parity.py \
    --pth_ckpt   "$V4_ROOT/runs/deimv2_n_tile_g2_multiscale_320_512_416x416/best_stg2.pth" \
    --pth_config "$V4_ROOT/runs/deimv2_n_tile_g2_multiscale_320_512_416x416/generated_configs/train.yml" \
    --onnx       "$V4_ROOT/runs/deimv2_n_tile_g2_multiscale_320_512_416x416/export/deimv2_n_h416_w416.onnx" \
    --image      "$SHITSPOTTER_DPATH/tpl/YOLOX/assets/dog.jpg"

# Desktop CPU bench
"$PYTHON_BIN" experiments/mobile_app_training_v4/06_benchmark_onnx_desktop.py \
    --onnx  "$V4_ROOT/runs/deimv2_n_tile_g2_multiscale_320_512_416x416/export/deimv2_n_h416_w416.onnx" \
    --image "$SHITSPOTTER_DPATH/tpl/YOLOX/assets/dog.jpg" \
    --warmup 5 --iters 50

# Wire the export into the phone app — see 07_register_in_phone_app.md.
```

### Knobs you will reach for

| Variable                  | Default                       | Purpose                                                 |
|---------------------------|-------------------------------|---------------------------------------------------------|
| `V4_ROOT`                 | `~/data/shitspotter_v4`       | Writable workspace for all v4 artifacts.                |
| `V4_RESIZE_MAX_DIM`       | `1280`                        | Long-side cap on the kept full-frame image.             |
| `V4_TILE_GRID`            | `2`                           | NxN tile grid per image.                                |
| `V4_TILE_OVERLAP`         | `0.20`                        | Fractional overlap between adjacent tiles.              |
| `V4_TILE_OUTPUT_DIM`      | `640`                         | Long-side cap on each tile after resize.                |
| `V4_INPUT_HW`             | `"320 320"` (N/Pico) / `"640 640"` (S) | Export resolution for this run.                |
| `V4_TRAIN_POLICY`         | `multiscale_256_416` / `..._512_768` | Training resolution policy (`fixed` to disable).  |
| `V4_MULTISCALE_REPEAT`    | `12`                          | How often the base size is repeated in the scale list.  |
| `V4_MULTISCALE_STOP_EPOCH`| `V4_NUM_EPOCHS - 4`           | Last epoch that uses multi-scale; afterwards = base.    |
| `V4_TRAIN_BATCH`          | `128` (N/Pico), `32` (S)      | Total training batch size.                              |
| `V4_NUM_EPOCHS`           | `60`/`80`/`30`                | N / Pico / S epoch budget.                              |
| `V4_NUM_GPUS`             | `1`                           | Multi-GPU via `torch.distributed.run` when > 1.         |
| `V4_SWEEP_CELLS`          | (default Pareto matrix)       | Override the sweep matrix; one `<v> <h> <w> <pol>` row. |
| `V4_SWEEP_KEEP_GOING`     | `0`                           | Continue the sweep past failed cells.                   |
| `V4_SWEEP_DO_{EXPORT,EVAL,BENCH}` | `1`                   | Disable individual stages of the sweep.                 |
| `FORCE_RETRAIN`           | `0`                           | Bypass the "checkpoint already exists" short-circuit.   |
| `FORCE_TILE_REBUILD`      | `False`                       | Re-run `01_*.sh` even if outputs exist.                 |
| `FORCE_REPRED`            | `0`                           | Re-run prediction in `04_*.sh` even if pred exists.     |
| `DOWNLOAD_PRETRAINED`     | `True`                        | Skip the gdown step in `00_setup.sh`.                   |

### Eligibility manifest fields

The TSV emitted by [eligibility_manifest.py](eligibility_manifest.py) has
one row per candidate with these columns:

```text
candidate_id, variant,
export_input_h, export_input_w,
train_resolution_policy, train_resolution_min, train_resolution_max,
train_resolution_choices, tile_training_policy,
checkpoint_path, onnx_path, modelspec_path,
test_ap_simplified,
desktop_latency_ms_p50, desktop_latency_ms_mean, desktop_latency_ms_p99,
desktop_eligible,
pixel5_latency_ms, pixel5_fps, pixel5_eligible,
phone_model_id, status, reasons
```

`pixel5_*` fields read `TODO` until you supply an on-device benchmark
TSV via `--pixel5_index`. The desktop CPU latency is a *proxy* for the
on-device gate, not a substitute.

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
