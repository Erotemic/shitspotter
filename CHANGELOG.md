# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Version 0.0.1] - 

### Added:

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
  - 117 commonTest + desktopTest cases all green. Operator checklist at
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
