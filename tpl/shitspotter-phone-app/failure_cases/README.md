# `failure_cases/` — local-only failure reports

This directory is **gitignored** (see `.gitignore`). It exists in the
tree so that:

- a fresh clone has the directory ready when the desktop harness or
  Android device first writes into it;
- documentation can link to a stable path.

## What lives here at runtime

The Android app writes captures to
`/sdcard/Android/data/io.kitware.shitspotter/files/failure_cases/`
on the device, **not** here. To pull them onto a workstation, run
`scripts/sync_failure_cases.sh` from a machine with the phone
plugged in. The pulled tree lands in
`./pulled_failure_cases/<DATE>/failure_cases/`.

The desktop harness writes here directly (relative to the working
directory the GUI was launched from) — only when the user clicks
"Save failure" in the desktop UI.

## Format

Each capture is a sub-directory `YYYYMMDD_HHMMSS_<uuid>/` containing:

```text
image.jpg          # camera frame, ARGB → JPEG, rotated to display
metadata.json      # FailureCaseMetadata (timestamp, model_id,
                   # delegate, latency, fps, failure_type, …)
detections.json    # List<Detection> at capture time
user_note.txt      # only when the user typed a note
```

The schema is locked by [`SerializationTest`](../composeApp/src/commonTest/kotlin/io/kitware/shitspotter/core/SerializationTest.kt).
Older captures (without `userNote`) round-trip cleanly because the
JSON parser is configured with `ignoreUnknownKeys = true` and the
field has a default.

## What you should NOT put here

- model weights (`*.onnx`, `*.tflite`, etc.) — they belong in the
  `local_models/` folder or in the device's external-files dir
- sample / test images for development — use `tpl/YOLOX/assets/dog.jpg`
  or any other in-repo image
- benchmark JSON reports — those go in `docs/benchmarks/`
