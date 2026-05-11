# 2026-05-11 — mobile_app_training_v4 scaffold

## Trigger

User asked for a new training-run scaffold focused on producing the best
possible detector for the new Pixel-5 phone app
(`tpl/shitspotter-phone-app/`). Two prior advisor briefs were attached:

1. **DEIMv2 model size** — train DEIMv2-N first (primary phone
   candidate), DEIMv2-Pico second (speed fallback), DEIMv2-S as
   teacher / quality reference. Do NOT start with DEIMv2-S as the live
   model.
2. **Coarse-to-fine inference** — the app needs `FAST_FULL_FRAME`,
   `BALANCED_FULL_FRAME`, `ROI_HIGH_RES`, `TILED_SEARCH`, and
   `FREEZE_FRAME_HIGH_RES` modes. A detector trained only on
   downsampled full frames will be OOD on tile inputs.

VM has no GPU; user will run training on the host. Output requested in
`experiments/mobile_app_training_v4/`.

## What landed

```
experiments/mobile_app_training_v4/
├── README.md                          — operational quick-start
├── DESIGN.md                          — full rationale + lesson refs
├── common.sh                          — env vars, helpers (sourced everywhere)
├── 00_setup.sh                        — env check, gdown pretrained DEIMv2
├── 01_make_tile_augmented_kwcoco.sh   — tile bundle + simplified MSCOCO json
├── tile_kwcoco.py                     — tile augmentation worker
├── _train_deimv2_variant.sh           — shared training library
├── 02_train_deimv2_n.sh               — primary Pixel-5 candidate
├── 02_train_deimv2_pico.sh            — speed fallback
├── 02_train_deimv2_s.sh               — DINOv3-backed reference
├── 03_export_onnx.sh                  — wraps export_onnx.py + writes modelspec
├── 04_eval_on_test.sh                 — kwcoco simplified-GT AP (v9-comparable)
├── 05_desktop_onnx_parity.py          — torch ↔ onnx parity guard
├── 06_benchmark_onnx_desktop.py       — desktop CPU latency
└── 07_register_in_phone_app.md        — prescriptive phone-app changeset
```

Plus a CHANGELOG entry pointing to this journal.

## Design choices worth recording

* **Bypass `shitspotter/algo_foundation_v3/cli_train.py`.** Its
  `variant` field has `choices=['deimv2_m', 'deimv2_s']`, so adding
  Pico / N would mean editing the v3 stack — a stable artifact behind
  the v9 result. Instead, `_train_deimv2_variant.sh` writes its own
  DEIMv2 train.yml that `__include__`s the upstream config and
  invokes `tpl/DEIMv2/train.py` directly. The v3 path is untouched.
* **Stage everything under `$V4_ROOT`.** The DVC roots are read-only
  per `project_readonly_filesystems.md`. `$V4_ROOT` defaults to
  `$HOME/data/shitspotter_v4` and is created by `common.sh`.
* **Tile-augmented training data instead of per-mode fine-tunes.** The
  same kwcoco bundle mixes downsized full frames (long side ≤ 1280)
  with 2 × 2 overlapping tiles cut from full-resolution sources
  (each ≤ 640). `min_keep_fraction=0.30` drops half-cut annotations.
  All three v4 variants train on this single bundle.
* **No SAM2 retrain in v4.** The detector is the bottleneck for live
  inference; v4 evaluates against the same zero-shot
  `sam2.1_hiera_base_plus` checkpoint as v9 so detector AP differences
  show through.
* **Modelspec sidecar at export time.** `03_export_onnx.sh` writes
  `<onnx>.modelspec.json` alongside the ONNX file with the same
  fields as the Kotlin `ModelSpec` data class. The phone-app side can
  consume this directly and the spec doubles as a single source of
  truth for input H/W/normalisation, dodging the "ORT defers shape
  validation until first run()" failure mode from
  `dev/journals/lessons_learned.md` §1.
* **No distillation script yet.** The advisor brief mentions
  S-as-teacher / N-as-student. We deliberately defer that until we
  have v4 baselines — see DESIGN.md §"Why no separate distillation
  step (yet)".
* **No phone-app code change here.** Adding
  `PostprocessType.DEIMV2` requires a Kotlin edit; that lands in a
  separate commit on the host machine, prescribed in step 7.

## Validation status (VM)

What ran on the VM:

* `bash -n` syntax-check on all 9 shell scripts → all pass.
* `python3 ast.parse` + module import on all 3 Python files → all OK.
* `tile_kwcoco.TileKwcocoCLI()` instantiates with expected defaults.

What did **not** run on the VM (no GPU, no shitspotter Python env on
PATH):

* End-to-end `01_*` tile generation against the real v9 train split.
* Any `02_*` training pass.
* `03_export_onnx.sh`, `04_eval_on_test.sh`,
  `05_desktop_onnx_parity.py`, `06_benchmark_onnx_desktop.py`.

These need to run on the user's host machine. README and DESIGN both
document the host/VM split explicitly.

## Comparison target

v9 OpenGroundingDINO + tuned SAM2 reaches AP = 0.766 on simplified test
GT (IoU=0.5). v4 numbers will be reported by `04_eval_on_test.sh`
against the same simplified test bundle so the comparison is
apples-to-apples. Expectation: N/Pico will land below v9 but be 10–30×
faster live; S should approach v9 quality.

## Follow-ups

* Run the pipeline on the host and record numbers in
  `$V4_ROOT/eval/.../eval/detect_metrics.json`.
* Add `08_distill_*.sh` once N/Pico baselines are known.
* On-phone benchmark + the Kotlin changeset from step 7 — those are
  separate journal entries when they land.
