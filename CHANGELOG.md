# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Version 0.0.1] - 

### Added:

* (2026-05-12): `mobile_app_training_v4`: landed the two fixes for the
  Q5/Q6 failure modes surfaced by sweep `20260511T134137Z`, and made
  the sweep restart gracefully.
  - `_train_deimv2_variant.sh` now emits a top-level `num_top_queries`
    override in the generated `train.yml`, clamped to
    `min(300, num_queries × num_classes)` per variant. Unblocks
    `deimv2_pico/femto/atto` on single-class shitspotter. The trainer
    error-helper block names the new symptom inline.
  - `04_eval_on_test.sh` pre-filters both the v9 simplified test GT
    and the predicted kwcoco file to drop annotations whose `bbox` is
    missing, `None`, or not a length-4 sequence, then passes the
    filtered files to `kwcoco eval`. Caches filtered outputs under
    `$EVAL_DPATH/`.
  - `02_sweep.sh`, `03_export_onnx.sh`, and `04_eval_on_test.sh` are
    now idempotent — re-running the sweep on the same `V4_ROOT` skips
    stages whose artifacts already exist (`best_stg2.pth`, ONNX
    ≥ 256 KiB, `detect_metrics.json`, `*.bench.json`). A cell with
    every enabled stage already complete is reported as
    `ok_resumed`. Force-rerun per stage with
    `V4_SWEEP_FORCE_{TRAIN,EXPORT,EVAL,BENCH}=1` (or the per-stage
    `FORCE_RETRAIN`/`FORCE_REEXPORT`/`FORCE_REEVAL`/`FORCE_REBENCH`
    flags individually).
  - New `V4_SWEEP_RETRY_FAILED=<prior_index.tsv>` mode filters the
    cell list to those whose prior status is not `ok`/`ok_resumed`
    — exactly the right tool to re-run only the cells broken by a
    just-fixed bug.
  - New tests: `tests/mobile_app_training_v4/test_num_top_queries_clamp.py`
    (9 parametrised cases) — locks the variant→`num_queries` table
    and the clamp invariant `num_top_queries ≤ num_queries × num_classes`.

* (2026-05-12): `mobile_app_training_v4`: documented two new sweep
  failure modes surfaced by the first long Pareto sweep
  (`sweeps/20260511T134137Z`): (a) `num_top_queries=300 > num_queries
  × num_classes` for the deimv2_pico variants on single-class
  shitspotter (RuntimeError out of range in DEIMv2 postprocessor
  topk), (b) `kwcoco coco_eval` `KeyError: 'bbox'` on the n@320
  predictions when the predictor emits an ann without a `bbox`
  field. Lessons added to `dev/journals/lessons_learned.md`
  §2026-05-12; new benchmark candidates Q5 (`num_top_queries` clamp)
  and Q6 (predict-side bbox invariant) added to
  `dev/benchmark-candidates/pipeline-bootstrap-questions.md`. Dated
  journal at
  `dev/journals/2026-05-12_v4_sweep_pico_topk_and_eval_bbox.md`.
  Scripts NOT YET updated to apply the fixes; the in-flight n@640
  cell will finish first.

* (2026-05-12): `mobile_app_training_v4`: added `08_status.sh` —
  scans `$V4_ROOT/{runs,eval,sweeps}` and prints one row per candidate
  (checkpoint present, ONNX size with stub flag, simplified-GT AP,
  desktop bench mean, sweep status). Documents the canonical on-disk
  layout in `README.md` under "Where am I in the sweep?" so a multi-
  hour sweep's progress is visible without reading the trainer log.

* (2026-05-11): scaffolded the `experiments/mobile_app_training_v4/`
  workflow targeting Pixel-5 live detection. Trains DEIMv2-N (primary
  candidate), DEIMv2-Pico (speed fallback), and DEIMv2-S (DINOv3-backed
  quality reference / future teacher) on a tile-augmented version of
  the v9 split (`train_imgs10671_b277c63d` + 2x2 overlapping tiles per
  source image). Includes ONNX export with a `*.modelspec.json` sidecar,
  v9-canonical simplified-GT eval re-using `cli_predict_boxes`, a
  torch↔ONNX parity guard, a desktop CPU latency benchmark, and a
  prescriptive `07_register_in_phone_app.md` describing the
  `PostprocessType.DEIMV2` change required on the phone-app side. The
  v4 pipeline does NOT modify `shitspotter/algo_foundation_v3/` or the
  in-tree phone app — it is self-contained and stages every artifact
  under `$V4_ROOT` (default `$HOME/data/shitspotter_v4`). Journal entry:
  `dev/journals/2026-05-11_mobile_app_training_v4.md`.

* (2026-05-10): scaffolded the v2 phone app under
  `tpl/shitspotter-phone-app/` (Kotlin Multiplatform + Compose
  Multiplatform, Android-first). Milestone 0/1/2 are functionally
  complete from the VM side, Milestone 3 is in progress; Pixel 5
  validation is the only remaining step.
  - Milestone 0 decision record:
    `tpl/shitspotter-phone-app/docs/000_stack_decision.md`.
  - Milestone 1 skeleton: CameraX `STRATEGY_KEEP_ONLY_LATEST` analysis
    loop, Compose overlay + HUD with FPS/inference latency/dropped
    frames/build commit, in-app score-threshold slider, model-selection
    chip row, on-device failure-case capture (JPEG + JSON + optional
    note), Compose-for-Desktop still-image harness, frame-directory
    replay source. Android debug APK is 29 MB (arm64-v8a).
  - Milestone 2 real-model backends: ORT-Android (NNAPI EP w/ FP16 +
    CPU fallback, records the actual delegate that loaded), ORT-JVM
    (CPU). Three ModelSpecs registered: yolox-nano-poop-cropped-v1,
    shitspotter-custom-v5-epoch115, shitspotter-custom-v2-epoch126.
    Input-shape validation rejects spec/model mismatches at construction
    time. AndroidModelLoader resolves a ModelSpec to a file via
    external-files-dir → cache → APK assets. Last-frame JPEG capture
    honours camera rotationDegrees. Detection boxes are rotated into
    display orientation before reaching the overlay.
  - Milestone 3 backend comparison: `BackendComparison` runs N backends
    back-to-back; `CompareCli` accepts repeated `--model=<path>`,
    `--score-threshold`, `--no-stub`, `--out=<json>`, `--help`;
    `describe --model=<path>` dumps ONNX input/output shapes.
  - Python parity: `scripts/python_reference_compare.py` re-implements
    the same letterbox + YOLOX postprocess in NumPy; full comparison
    archived at
    `tpl/shitspotter-phone-app/docs/004_kotlin_python_parity.md`.
  - 122 commonTest + desktopTest cases all green. Operator checklist at
    `tpl/shitspotter-phone-app/docs/001_build_run_validate.md`,
    benchmark schema at
    `tpl/shitspotter-phone-app/docs/002_benchmarks_template.md`,
    next-agent backlog at
    `tpl/shitspotter-phone-app/docs/003_known_limitations.md`,
    Python-vs-Kotlin parity at
    `tpl/shitspotter-phone-app/docs/004_kotlin_python_parity.md`,
    runtime architecture at
    `tpl/shitspotter-phone-app/docs/005_runtime_architecture.md`. Run
    journal at `dev/journals/2026-05-10_phone_app_kmp_scaffold.md`.
    APKs: 29 MB debug, 20 MB release (R8 + arm64-v8a).

### Fixed:

* (2024-03-01): fixed issue in make splits where the 2/3 image protocol
  selection was backwards. Also improved protocol usage. Fixed issue where
  after and negative images were never used.

## [Version 0.0.1] - 

### Added
* Initial version
