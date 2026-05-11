# mobile_app_training_v4 — design notes

This is the longer-form rationale for why v4 looks the way it does.
Pair it with the operational [README.md](README.md).

## Problem statement (constrained optimization)

```text
maximize:  detection quality on the v9 simplified test GT
subject to:
  Pixel 5 live detection >= 10 FPS in the intended mode
  exportable to ONNX / mobile runtime (fixed-shape first; dynamic later)
  local-only inference
  app preview remains responsive
  failure-case capture still works
  supports full-frame and ROI/tiled inference modes
```

We train and evaluate a **Pareto frontier over both architecture and
input resolution.** Resolution is a first-class axis, not just a
hyperparameter.

Each deployable model has a **fixed export resolution** for phone
benchmarking, but **training may use bounded multi-resolution
augmentation** to improve scale robustness within a band around the
deploy size.

A model is not selected because it is smallest or fastest. It is
selected because it is the highest-quality candidate that satisfies the
Pixel 5 deployment gates.

### Eligibility rule (mechanical)

```text
A candidate is eligible only if it:
  1. trains successfully;
  2. exports to ONNX (fixed-shape);
  3. passes PyTorch <-> ONNX parity;
  4. runs in the phone app;
  5. reaches >= 10 FPS on Pixel 5 in its intended mode;
  6. keeps preview responsive;
  7. preserves failure-case capture.

Among eligible candidates, choose the one with the best AP on the v9
simplified test GT (IoU=0.5).
```

Gates 1–3 are checked by the v4 scaffold itself (training, export,
parity scripts). Gate 4–7 must be filled in by the host-side phone-app
step. The desktop CPU latency benchmark (`06_benchmark_onnx_desktop.py`)
is a *proxy* for gate 5 — it cannot replace it.

### Three axes in tension

1. **Latency budget** — 100 ms/frame total includes camera I/O, color
   conversion, model forward, postprocess, and overlay update. The
   model itself needs to be comfortably under ~50–70 ms on a 2020 mid-
   range mobile SoC.
2. **Detection quality** — small, low-contrast objects in cluttered
   outdoor scenes. We've been collecting since 2021; the v9 split is
   10 671 / 1 258 / 121 train/vali/test images.
3. **Deploy ergonomics** — ONNX-first, single-ABI APK, must-not-bloat
   the existing 29 MB debug build.

The scaffold optimizes (1) jointly with (2) by sweeping
(architecture × export resolution × train resolution policy) and
recording per-cell quality + desktop latency + (eventually) on-device
latency in a single eligibility manifest.

## Lessons we are *not* relearning

These come from
[`dev/journals/lessons_learned.md`](../../dev/journals/lessons_learned.md)
and the canonical-metric note in
[`shitspotter/algo_foundation_v3/`](../../shitspotter/algo_foundation_v3/).

* **Simplified-GT AP is the canonical detection metric.** The raw test
  GT has ~16 instances/image vs ~2 in train/vali; raw-GT AP for v9 was
  0.090, simplified-GT was 0.766. We re-use the v9 simplified test
  bundle so v4 numbers are directly comparable. → enforced in
  `04_eval_on_test.sh`.
* **DVC roots are read-only.** All v4 derived artifacts live under
  `$V4_ROOT`. → enforced in `common.sh`.
* **ONNX shape mismatches fail at first run, not at construction.** We
  ship the modelspec sidecar JSON next to every export so the phone
  app side has a single source of truth for input H/W/normalisation. →
  written by `03_export_onnx.sh`.
* **Coordinate boundaries between buffer / display / model.** The
  phone app already has the rotation + letterbox machinery; the v4
  modelspec encodes `LETTERBOX` so the existing path keeps working. →
  declared in `03_export_onnx.sh`.
* **APK ABI explosion.** Stays at one ABI (arm64-v8a) — that's a
  property of the phone-app gradle config, not of v4, but step 7
  reminds us not to break it.

## Candidate identity

A "candidate" in v4 is uniquely identified by:

```text
variant                  e.g. deimv2_n, deimv2_pico, deimv2_s
export_input_h           fixed export H (pixels)
export_input_w           fixed export W (pixels)
train_resolution_policy  e.g. fixed | multiscale | multiscale_320_512
tile_training_policy     e.g. tile_g2_overlap0.20_out640
```

Each unique 5-tuple produces:

* its own training workdir under `$V4_ROOT/runs/`,
* its own ONNX export under `<workdir>/export/*.onnx`,
* its own modelspec sidecar (`*.modelspec.json`),
* its own eval row in the eligibility manifest, and
* its own `ModelSpec` entry in the phone app.

The phone app should be able to choose between candidates at runtime
the same way it currently picks between YOLOX-nano-poop, custom-v5,
and simple-v3.

The default sweep covers:

```text
deimv2_n     320 × 320   multiscale_256_416
deimv2_n     416 × 416   multiscale_320_512
deimv2_pico  320 × 320   multiscale_256_416
deimv2_pico  416 × 416   multiscale_320_512
deimv2_n     512 × 512   multiscale_384_640
deimv2_pico  512 × 512   multiscale_384_640
deimv2_n     640 × 640   multiscale_512_768
deimv2_s     640 × 640   multiscale_512_768
```

Cells are ordered cheapest-first so a sweep that hits the FPS gate
early can be stopped before training the heavier rows.

## Train vs. export resolution

These two are deliberately separate fields on every candidate:

```text
export resolution         fixed, explicit, benchmarked, exported to ONNX
train resolution policy   may include random multi-scale resize augmentation
```

DEIMv2's `BatchImageCollateFunction` already supports stochastic per-
batch resizing centered on `base_size` with `base_size_repeat`
controlling how many times each scale is repeated before switching
(see `tpl/DEIMv2/engine/data/dataloader.py:generate_scales`). The
policy strings map to that mechanism:

```text
fixed                    base_size_repeat=None — single scale
multiscale               band of ±25% around base_size = max(H,W),
                         32-px granularity, base_size repeated 12 times
multiscale_<S>           ±25% band around <S>
multiscale_<lo>_<hi>     band sized to target [lo, hi]; base picked as
                         (lo+hi)/2 rounded to 32
```

The trainer dumps the *resolved effective config* and the *exact scale
list it will sample from* to `<workdir>/policy.json` and
`<workdir>/resolved_effective_config.yml` so we can verify what DEIMv2
actually does — not just what we asked for. This is the answer to
"don't just set `eval_spatial_size` and assume training uses it."

## Why DEIMv2 over alternatives

The v9 result established that the foundation-detector family
(OpenGroundingDINO with DINOv2 + BERT) is the strongest current
performer on this dataset. DEIMv2 is the closest *small* relative:

* Same lineage (DETR family).
* The S/M variants use a DINOv3-distilled ViT-Tiny backbone, which is
  the most plausible 2026 path for the teacher → student story.
* The Atto/Femto/Pico/N variants drop down to HGNetv2 backbones with
  COCO AP from 23.8 (Atto) to 43.0 (N). Their on-device latency
  estimates (1.1–2.3 ms on RTX) translate to phone-realistic 30–80 ms
  budgets.
* Upstream ships the export pipeline we need (`tools/deployment/export_onnx.py`).

Alternatives considered and not pursued in this iteration:

* **YOLOX-nano** — already in the app; trained on cropped patches and
  visibly OOD on full-frame inputs (`dog.jpg` returns 0.757 confidence
  garbage). Keeping for backwards compat, not iterating on it.
* **GroundingDINO / OpenGroundingDINO** — too heavy for live phone use.
  Reserved as the v9 teacher.
* **SAM2 / SAM-style segmenters** — useful for refinement, useless as
  a live detector at 10 FPS. SAM2 stays in the v3/v9 evaluation path.

## Why "tile-augmented" training data

The phone app's GOAL.md and the second advisor brief (in the prompt
that triggered this work) call for several detection modes:

```text
FAST_FULL_FRAME       — full preview, model input 320 or 416
BALANCED_FULL_FRAME   — full 1280x720 preview, model input 640
ROI_HIGH_RES          — crop a region, feed to model
TILED_SEARCH          — 2x2 overlapping tiles, one per frame
FREEZE_FRAME_HIGH_RES — high-res still, possibly tiled
```

A detector trained only on full-image downsampled inputs sees boxes that
occupy a small fraction of the input grid. When the same detector is
fed a high-res tile at inference time, the *same physical poop* now
occupies a much larger fraction of the input grid — that's a
distribution shift the model has not been trained on.

`01_make_tile_augmented_kwcoco.sh` solves this by mixing both views
into the same training bundle:

* Each source image emits **one resized full frame** (long side ≤
  1280) plus **N×N overlapping tiles**, each cut from the *full-
  resolution* source and resized to ≤ 640.
* GT boxes are warped into each tile's coordinate frame and clipped.
* Annotations whose visible fraction in a tile drops below
  `min_keep_fraction` (default 0.30) are dropped.

This way a single trained checkpoint serves all the planned phone-app
modes without needing distinct fine-tunes per mode.

The `2 × 2` grid with 20 % overlap is intentional — at the v9 split
(10 671 train images), it produces ~5× the original count, which keeps
training time tractable on a single 24 GB GPU but still gives the
model meaningful tile-scale supervision.

## Why no separate distillation step (yet)

The advisor brief includes a "DEIMv2-S as teacher / DEIMv2-N as
student" track. We deliberately defer the explicit distillation pass
for two reasons:

1. **Establish the baseline first.** We do not yet have v4 numbers for
   any DEIMv2 variant on this split; spending compute on a
   distillation harness before knowing the teacher–student gap is
   premature.
2. **The training data already encodes the teacher.** Because v9
   bootstraps the labels we train on (via the existing
   `experiments/foundation_detseg_v3/` bootstrap pipeline), v4 is
   already learning from a v9-curated dataset.

Once v4 baselines are in, follow-up will add an `08_distill_*.sh`
script that uses the v9 package or a tuned `deimv2_s` to label held-out
or augmented frames and adds them as soft / pseudo targets. That step
is intentionally not in this directory yet.

## Open questions / decisions tagged for follow-up

* **Default input size for N**: 320 vs 416 vs 640. The HGNetv2 N config
  defaults to 640. We've set the v4 default at 320 to maximise the
  chance of hitting 10 FPS on Pixel 5; if the AP drop relative to v9
  is too steep, bump to 416 or 640 and re-bench.
* **Mosaic + heavy augmentation interactions.** The upstream Pico/N
  configs use heavy mosaic + RandomZoomOut + RandomIoUCrop.
  Combined with our tile augmentation, that may double-augment small
  objects. If AP plateaus low, a second-pass training script that
  disables mosaic for the last N epochs is the obvious knob.
* **Class-agnostic NMS in DEIMv2.** The built-in DEIMv2 postprocessor
  emits top-K per query; we may not need an additional NMS pass, but
  the existing app applies one uniformly. The phone-app integration
  doc proposes keeping NMS for parity with other backends.
* **YOLOX retire.** Once a v4 export consistently beats YOLOX-nano on
  the simplified test, we should consider deprecating the YOLOX entry
  from the registry. Not in scope for this iteration.

## File layout

```text
experiments/mobile_app_training_v4/
├── README.md                          — operational quick-start
├── DESIGN.md                          — this file
├── common.sh                          — env vars + helpers, sourced everywhere
├── 00_setup.sh                        — env check + pretrained download
├── 01_make_tile_augmented_kwcoco.sh   — tile bundle + simplified MSCOCO json
├── tile_kwcoco.py                     — tile augmentation worker (Python)
├── _train_deimv2_variant.sh           — shared training library
├── 02_train_deimv2_n.sh               — entrypoint: deimv2_n
├── 02_train_deimv2_pico.sh            — entrypoint: deimv2_pico
├── 02_train_deimv2_s.sh               — entrypoint: deimv2_s
├── 03_export_onnx.sh                  — wraps DEIMv2 export_onnx + modelspec
├── 04_eval_on_test.sh                 — kwcoco simplified-GT AP
├── 05_desktop_onnx_parity.py          — torch ↔ onnx parity guard
├── 06_benchmark_onnx_desktop.py       — desktop CPU latency
└── 07_register_in_phone_app.md        — prescriptive phone-app changes
```

There is **no top-level entrypoint** that runs the whole pipeline.
That's deliberate — every step is restart-safe individually, and a
single "run all" script tends to mask errors mid-run. The Karpathy
loop README in v3 says the same.
