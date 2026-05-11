# mobile_app_training_v4 — preemptive audit

Audit performed 2026-05-11 against commit `a833f6c` (before this audit
landed). Driven against a hand-built smoke fixture under
`/data/tmp/v4_smoke/` plus `kwcoco subset` of the real splits.

The audit had three concerns:

1. Find pipeline-internal bugs that haven't surfaced because we hadn't
   run end-to-end.
2. Find external-tool dep bugs that the v4 pipeline trips into.
3. Build the test infrastructure to catch regressions of both kinds.

## TL;DR

* **10 audit findings**, of which **6 are real bugs** (4 in v4 itself, 2
  in upstream tools) — all fixed in this PR.
* **31 pytest tests** added under `tests/mobile_app_training_v4/`,
  covering tile augmentation, train-policy parser, eligibility manifest
  state machine, and the v4_mock model.
* **End-to-end smoke** `train -> export -> eval -> bench -> manifest`
  now runs in <30s on CPU using the new `v4_mock_tiny` first-class
  variant. Final eligibility class: `HOST_PROMISING`.

## Findings — fixed in this PR

### F1. `kwimage.imresize(..., interpolation='area')` crashes without cv2
[v4 / fixed]

`kwimage` falls back to `skimage.transform.resize` when cv2 isn't
installed. The skimage backend doesn't recognise `interpolation='area'`
and raises `NotImplementedError`. **`tile_kwcoco._resize_with_long_side`
now catches this and falls back to `'linear'`** (slightly worse
aliasing on downsampling, still correct).

The skimage-vs-cv2 backend behaviour gap is a real kwimage UX issue
worth filing upstream — `'area'` is the standard correct downsampling
interpolation and should be honored or mapped to the closest
equivalent.

### F2. `simplify_kwcoco` has a hidden `geowatch` import dependency
[upstream / worked around]

`shitspotter.cli.simplify_kwcoco` does `from geowatch.utils.util_kwimage
import find_low_overlap_covering_boxes` inside `main()`. When
`geowatch` isn't installed, the import crashes the CLI. `geowatch`
itself hard-imports `osgeo` (GDAL Python bindings) at package init,
which can't be installed via pip without system libs.

The v4 tile script now soft-falls back to a straight `cp` of the tile
bundle when `simplify_kwcoco` fails — set `V4_FORCE_SIMPLIFY=1` to
make it fatal. **Worth filing two upstream issues:**

* `shitspotter.cli.simplify_kwcoco` should import the geowatch helper
  inside the function it's used in (lazy), and either declare geowatch
  as a hard dep OR handle ImportError gracefully.
* `geowatch.__init__` shouldn't hard-fail on missing osgeo when
  importing utility submodules.

### F3. `02_sweep.sh` could mark failed cells as `status=ok`
[v4 / fixed in earlier review-pass; verified here]

The reviewer flagged that the original sweep printed `status=ok`
unconditionally after `run_cell` returned. The fix in `825abf0` (per-
stage exit-code check, `set +e` guard around `run_cell` to survive
parent `set -e`, status reflects worst stage) is **verified by the
new pytest suite** — see the `fail_train` / `fail_export` /
`fail_eval` / `fail_bench` paths.

### F4. `eligibility_manifest --sweep_index` silently used wrong V4_ROOT
[v4 / fixed]

When invoked with `--sweep_index <tsv>` and *without* `--v4_root` or
`$V4_ROOT` set in env, `_find_eval_ap` defaulted v4_root to
`~/data/shitspotter_v4`, which was wrong if the sweep wrote to a
different root. Result: `test_ap_simplified` came back as `None`,
candidate landed in `NOT_READY` with `status=no_eval` — a false
negative.

**Fix:** `_find_eval_ap` now also tries to infer v4_root from the
workdir's grandparent (which by the v4 layout convention is V4_ROOT).
Pytest `test_class_HOST_PROMISING_when_all_host_gates_pass` would have
caught the regression.

### F5. `04_eval_on_test.sh` mock dispatcher preferred v9 cached test GT
[v4 / fixed]

The mock dispatcher's test-GT discovery preferred
`$DVC_EXPT_DPATH_RO/.../test.simplified.kwcoco.zip` over the configured
`V4_TEST_FPATH`. On hosts where v9 had previously run, the mock would
try to evaluate against the v9 simplified test GT — which has
annotations missing bbox fields, crashing kwcoco eval with a
`KeyError: 'bbox'`. **Fix:** mock dispatcher now uses `V4_TEST_FPATH`
verbatim (override with `V4_MOCK_TEST_KWCOCO`).

The DEIMv2 path's existing v9 fallback is unchanged — it's correct
there because the v9 path applies `ensure_true_bboxes` to patch the
GT before predicting.

### F6. `kwcoco subset --select_images "..."` requires `jq` Python pkg
[upstream / documented]

`kwcoco subset --select_images ".id <= 8"` raises
`ModuleNotFoundError: No module named 'jq'` because the generic image
query is implemented via the Python `jq` library. The error message
*does* surface this clearly, but the dep isn't declared by kwcoco.

**Workaround:** the deprecated `--gids 1,2,3,4,...` flag works without
jq. The conftest fixture uses that form. Worth filing — kwcoco should
either declare jq as an install_requires or document it in the
`--select_images` help text more visibly.

## Findings — documented but not blocking

### F7. `kwcoco` deprecation warnings on every load
[upstream / noisy but non-blocking]

* `Images.coco_images_iter()` deprecated (hit inside simplify_kwcoco)
* `"img_root"` dataset member deprecated (hit on every load of zips
  saved by older kwcoco versions, including the v9 split files)

Both will become errors in kwcoco 1.0.0. Worth a separate cleanup
pass against the entire shitspotter codebase + the saved kwcoco
files on the DVC mount. Out of scope for v4.

### F8. `delayed_image` warns "DelayedLoad may not be efficient without gdal"
[upstream / non-blocking]

Hit on every kwcoco image load. Performance hint only. The host runs
have GDAL; the smoke env doesn't. Out of scope.

### F9. `04_eval_on_test.sh` inline package hardcodes `device: cuda:0`
[v4 / known]

The DEIMv2 inline-preset branch in `04_eval_on_test.sh` writes
`device: cuda:0` into the package YAML. On CPU-only hosts this would
fail the predict step. Currently only hit by the deimv2_n/pico path,
which is only ever run on a GPU host. **Acceptable for now**; if we
ever want to eval the DEIMv2 detectors on CPU we should respect a
`V4_DEVICE` env override.

### F10. `tile_kwcoco.py` re-encodes JPEG even when it's a no-op
[v4 / known, low priority]

When the source image is already smaller than `full_dim`, we skip the
resize but still re-encode the JPEG (because we always write into the
asset_dname/ subdir). For the tile bundle this is correct (we want
self-contained assets), but the second JPEG decode-encode cycle is
quality loss for no benefit. **Acceptable for now**; small.

## What the test suite covers

```
tests/mobile_app_training_v4/
├── conftest.py                     synthetic_kwcoco fixture (always
│                                   available), real_subset_train
│                                   fixture (DVC-mount-gated)
├── test_tile_kwcoco.py             8 tests: tile extents, bbox clipping,
│                                   resize math, end-to-end on the
│                                   synthetic bundle
├── test_train_policy_parser.py     7 tests: every documented
│                                   train_resolution_policy form +
│                                   the rounding-to-32 corner cases,
│                                   exercised by sourcing the bash
│                                   block in a controlled shim
├── test_eligibility_manifest.py    11 tests: every state-machine
│                                   transition (NOT_READY x4 paths,
│                                   HOST_PROMISING x2, PHONE_ELIGIBLE,
│                                   PHONE_INELIGIBLE), winner-picking,
│                                   phone_model_id format
└── test_v4_mock.py                 5 tests: model output shapes,
                                    initial-gate-off invariant,
                                    gate-flips-with-few-steps, train
                                    -> export -> ORT inference parity
```

Run with:

```bash
source experiments/mobile_app_training_v4/setup_env.sh
"$PYTHON_BIN" -m pytest tests/mobile_app_training_v4/ -q
# 31 passed in <10s on CPU
```

## End-to-end smoke command (for the host)

This is the one-liner that exercises every script except the DEIMv2
trainer itself, on a tiny dataset, in seconds:

```bash
source experiments/mobile_app_training_v4/setup_env.sh
V4_ROOT=/tmp/v4_smoke \
    V4_TRAIN_FPATH=...smoke_train.kwcoco.zip \
    V4_VALI_FPATH=...smoke_vali.kwcoco.zip \
    V4_TEST_FPATH=...smoke_test.kwcoco.zip \
    V4_SWEEP_CELLS="v4_mock_tiny 256 256 fixed" \
    V4_NUM_EPOCHS=4 V4_LR=1.0 \
    bash experiments/mobile_app_training_v4/run_all.sh
```

Produces a `HOST_PROMISING` candidate with a real (small) ONNX, a
working modelspec.json sidecar, and a populated eligibility manifest.

## Upstream bugs worth filing

| # | repo                | summary                                                              |
|---|---------------------|----------------------------------------------------------------------|
| 1 | kwimage             | `imresize(interp='area')` raises NotImplementedError without cv2     |
| 2 | shitspotter         | `simplify_kwcoco` has hidden `geowatch` import; should be lazy/optional |
| 3 | geowatch            | hard import of `osgeo` at package init blocks pip-only installs       |
| 4 | kwcoco              | `subset --select_images` needs `jq` python pkg; not declared/documented |
| 5 | kwcoco              | `Images.coco_images_iter()` deprecated; will break in 1.0 (in shitspotter callers) |
| 6 | kwcoco              | `img_root` deprecation noise on every load of older zips             |

The audit's outcome is that v4 is in good shape to start GPU runs
once geowatch + cv2 are present on the host (both are part of the
foundation_v3 requirements already), with regression coverage in
place to catch future plumbing bugs in seconds.
