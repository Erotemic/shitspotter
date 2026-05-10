# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Version 0.0.1] - 

### Added:

* (2026-05-10): scaffolded the v2 phone app under
  `tpl/shitspotter-phone-app/` (Kotlin Multiplatform + Compose
  Multiplatform, Android-first). Milestone 0 decision record at
  `tpl/shitspotter-phone-app/docs/000_stack_decision.md`. Milestone 1
  skeleton (CameraX + overlay + HUD + failure-case capture + desktop
  still-image harness) builds and the Android debug APK assembles.
  Milestone 2 ONNX backends (Android + JVM) are wired and load the
  YOLOX-nano poop model when present; falls back to the stub detector
  when the model file is absent so the pipeline still exercises.
  Validation/build commands are in
  `tpl/shitspotter-phone-app/docs/001_build_run_validate.md` and the
  run journal is at `dev/journals/2026-05-10_phone_app_kmp_scaffold.md`.

### Fixed:

* (2024-03-01): fixed issue in make splits where the 2/3 image protocol
  selection was backwards. Also improved protocol usage. Fixed issue where
  after and negative images were never used.

## [Version 0.0.1] - 

### Added
* Initial version
