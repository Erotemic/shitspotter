# Lessons learned

Distilled postmortems per [`../AGENT_BENCHMARK_DISCIPLINE.md`](../AGENT_BENCHMARK_DISCIPLINE.md)
§Journals. Less formal than `benchmark-candidates/`; if an entry
crystallises into a clear invariant, also create a benchmark
candidate.

---

## 2026-05-11 (afternoon): mobile_app_training_v4 — pipeline-bootstrap deps cluster

### Trigger

Continued v4 sweep on the host (torch 2.11+cu130, python 3.13.13,
geowatch+gdal installed). The morning's audit fixed the obvious
wrapper-API bugs. The afternoon surfaced a *second* category: the
ten-or-so undeclared / version-incompatible deps the v4 sweep needs
present on the host before any DEIMv2 cell can finish its first
forward pass. They cascade — fixing one reveals the next, each
consuming a sweep restart.

### Symptoms (in the order they hit)

| # | error | what was missing |
|---|-------|------------------|
| 1 | `gdown: unrecognized arguments: --fuzzy` | gdown 6.x dropped `--fuzzy` |
| 2 | `cv2.error: Failed to allocate 44947419955200 bytes` | kwimage.imresize positional 2 is `scale`, not `dsize` |
| 3 | `cv2.error: imwrite() got an unexpected keyword argument 'imwrite_params'` | kwimage.imwrite forwards \*\*kwargs to cv2 |
| 4 | `gdown` writes 4 KB HTML stub on Drive throttle, cached as success | min-size guard missing |
| 5 | `kwimage NotImplementedError('area')` | skimage backend doesn't support area interp |
| 6 | `from osgeo import gdal: No module named 'osgeo'` | geowatch hard-imports osgeo at `__init__` |
| 7 | `pip ... does not provide upload-time metadata` | pip 25+ rejects girder index (PEP 700) |
| 8 | `RuntimeError: operator torchvision::nms does not exist` | torch/torchvision ABI mismatch (env-local) |
| 9 | `ModuleNotFoundError: 'onnxscript'` | torch.onnx.export imports it at function-call time |
| 10 | `ModuleNotFoundError: 'faster_coco_eval'` | DEIMv2's requirements.txt isn't installed transitively |
| 11 | `ModuleNotFoundError: 'tensorboard'` | same — next item in DEIMv2 deps |
| 12 | `_train_deimv2_variant.sh: V4_TRAIN_BATCH must be set by the caller` | sweep dispatch dropped the per-variant defaults |
| 13 | `TypeError: CocoDetection.__init__() got an unexpected keyword argument 'collate_fn'` | YAML 4-space indent put `collate_fn` under `dataset` instead of as its sibling |

### Root cause — common to most of the cluster

The v4 pipeline composes ~6 third-party libraries (DEIMv2, kwimage,
kwcoco, geowatch, torch, gdown), each with its own undeclared / lazy
runtime deps and version-bound API. A "fresh host" runs the pipeline
through O(10) thin layers before reaching the work — and any one
missing piece kills it. Worse, fixing one and restarting reveals the
next, so each iteration costs a full sweep restart unless you
front-load the audit.

### Fix — front-loaded environment audit

`00_setup.sh` is now responsible for *all three* dep families:

* gdown (used by step 0)
* onnxscript / onnx / onnxruntime (used by export + bench + parity)
* the entire `tpl/DEIMv2/requirements.txt` (used by every DEIMv2 cell)

Each is a probe-then-install loop that's a no-op when present. With
this in place, **all 13 errors above happen at `00_setup.sh`** if at
all, not 30 seconds into the first GPU cell of an 8-cell sweep.

The non-dep bugs (12, 13) get caught by the new pytest suite + the
end-to-end smoke run via `V4_RUN_ALL_SMOKE=1`. Both should run before
any long sweep on a fresh host.

### Durable lesson

There are two distinct error-shapes in cross-library pipelines:

1. **Wrapper-API misuse** (yesterday's lesson — wrong kwarg name,
   removed CLI flag, positional-vs-keyword confusion). Solution:
   pin keyword args at every wrapper call site. Already in place.
2. **Undeclared transitive runtime deps** (today's lesson — geowatch
   needs osgeo, DEIMv2 needs faster_coco_eval, torch.onnx needs
   onnxscript, etc.). Solution: a probe-everything `00_setup.sh`
   that walks every transitive `requirements.txt` it knows about
   and fails loud + early if anything is missing.

Both classes share a meta-lesson: **the cost of one bug is "next
sweep iteration"** — i.e. minutes-to-hours of wasted GPU time.
Front-load the discovery into a 30-second pre-flight and treat the
pre-flight as part of the deliverable, not a developer afterthought.

### Late finding (#14): HGNetv2 hybrid encoder doesn't support multi-scale

After all twelve cluster bugs landed and the trainer survived past
config build, the first DEIMv2-N cell crashed inside the encoder:

```text
RuntimeError: The size of tensor a (121) must match the size of
tensor b (100) at non-singleton dimension 1
```

121 = 11×11 (input 352, stride 32); 100 = 10×10 (input 320, stride
32). The encoder pre-bakes positional embeddings at
`eval_spatial_size` (we set 320) and doesn't dynamically interpolate
per batch. Multi-scale collate jittered the input to 352, the
pre-baked pos_embed didn't match, → tensor mismatch deep in the
encoder's `with_pos_embed`.

Confirmed by reading upstream configs:

* `deimv2_hgnetv2_n_coco.yml`, `deimv2_hgnetv2_pico_coco.yml` —
  `base_size_repeat: ~` (multiscale disabled)
* `deimv2_dinov3_s_coco.yml` — `base_size_repeat: 20` (multiscale on)

Upstream knows. The HGNetv2 hybrid encoder is fixed-size; only the
DINOv3-backed variants support per-batch resize.

**Fix:** per-variant default in `_train_deimv2_variant.sh` —
HGNetv2 (Atto/Femto/Pico/N) defaults to `V4_TRAIN_POLICY=fixed`,
DINOv3 (S/M/L/X) defaults to `multiscale`. The sweep matrix updated
to match. **Multi-resolution coverage for HGNetv2 cells comes from
the SWEEP** (multiple cells at different fixed sizes) **+ the tile
augmentation** (which already mixes 320..1280 px content per
training image). This is closer to what the user originally asked
for anyway: "different deployable models at different fixed
resolutions".

Lesson: **upstream model configs encode hard architectural
constraints** (here, "this encoder needs a fixed input size"). When
generating configs programmatically, mirror the upstream defaults
unless you understand why you're departing. We departed and ate a
sweep restart for it.

### YAML-indent bug (#13) deserves separate mention

`MULTISCALE_BLOCK` was a `cat <<EOF` heredoc inserted into a larger
heredoc via `$MULTISCALE_BLOCK`. The inner block had 4-space indent
intending to land at "child of `train_dataloader`" (2 spaces in YAML)
+ "sibling of `dataset`" (which is 2 spaces). Got the math wrong —
4-space indent landed it as a child of `dataset` instead. DEIMv2's
`workspace.create()` then forwarded `collate_fn` as a CocoDetection
constructor kwarg, which has no such parameter, → TypeError.

The fix is one digit (`s/4 spaces/2 spaces/`). The deeper lesson:
**bash heredoc + indented-YAML composition is fragile**. Going
forward, any v4 config generation we extend should validate the
output by parsing it through PyYAML before invoking the trainer.
This is now a benchmark candidate
(`dev/benchmark-candidates/pipeline-bootstrap-questions.md` §1).

### Validation status

After all 13 fixes:

* `tests/mobile_app_training_v4/` — 31 tests pass on torch 2.4 (VM)
  and torch 2.11 (host); the ONNX-roundtrip test self-skips when
  onnxscript is missing.
* `bash 00_setup.sh` — installs the full dep set on a clean env in
  ~minutes; on an env that already has them, ~seconds.
* `kwcoco subset --gids ... + tile_kwcoco.py + simplify_kwcoco
  + v4_mock train + export + ORT inference` — runs end-to-end on
  CPU in ~10 s on the smoke fixture.
* DEIMv2-N first sweep cell trainer — survives past config build,
  past data load, into the actual training loop on the host. Long
  GPU runs not yet confirmed at the time of writing.

---

## 2026-05-12: mobile_app_training_v4 — first long-sweep findings (pico topk + 320 eval bbox)

### Trigger

The first long Pareto sweep
(`$V4_ROOT/sweeps/20260511T134137Z`) ran for ~26 h on the host and
produced the first full per-cell signal. `08_status.sh` (added in
`ba5a5a3`) summarised the state. Two distinct failure modes the
upfront audit + 31-test pytest suite did *not* catch.

### Symptoms (in failure order)

| # | cell | stage | error |
|---|------|-------|-------|
| 1 | `deimv2_pico_*_fixed_{320,416,512}` (3 cells) | first val pass after epoch 0 | `RuntimeError: selected index k out of range` from `torch.topk(scores.flatten(1), num_top_queries, dim=-1)` in `tpl/DEIMv2/engine/deim/postprocessor.py:59` |
| 2 | `deimv2_n_tile_g2_fixed_320x320` | eval | `KeyError: 'bbox'` from `kwcoco.coco_evaluator._coerce_dets` on the *predicted* dataset (118/118 images predicted cleanly, then evaluator coercion died) |

The healthy outcomes (n@416 AP=0.4056 @20.3 ms, n@512 AP=0.4765
@31.5 ms, n@640 still in flight) confirm the rest of the pipeline
works end-to-end — these are two specific bugs, not a meltdown.

### Root cause #1 — `num_top_queries > num_queries × num_classes`

The DEIMv2 postprocessor's first op is
`torch.topk(scores.flatten(1), num_top_queries, dim=-1)`. The flatten
yields `B × (num_queries × num_classes)` elements.

* Upstream default (`tpl/DEIMv2/configs/base/dfine_hgnetv2.yml:76`):
  `num_top_queries: 300`.
* Pico variant
  (`tpl/DEIMv2/configs/deimv2/deimv2_hgnetv2_pico_coco.yml:44`):
  `num_queries: 200`.
* v4 generated train.yml hard-codes `num_classes: 1` (shitspotter is
  a single-class detector).

→ flatten size = 200 × 1 = **200**; k = 300; topk dies.

n variant uses the default 300 queries — 300 × 1 = 300, k = 300, so
it scrapes by. Same trap waits for `femto` (150 q) and `atto`
(100 q) if anyone enables them.

Generic invariant: `num_top_queries ≤ num_queries × num_classes`.
Upstream's COCO configs (80 classes) make this trivially satisfied;
single-class detectors break it.

### Root cause #2 — kwcoco coco_eval requires `bbox` on every coerced annotation

`CocoEvaluator._coerce_dets` does
`kwimage.Boxes([a['bbox'] for a in anns], 'xywh')` on both the true
and the predicted dataset. Any annotation without a `bbox` key
raises `KeyError`.

The cli_predict_boxes path for the 320×320 model evidently emitted
at least one annotation without a `bbox`. The 416 and 512 cells did
not — same predictor, same code, only the model output distribution
differs. Best guess: at 320 input the model produced at least one
prediction that degenerated to zero-area after rescale, and the
downstream box-list build dropped the `bbox` field instead of
filtering the annotation. Needs a one-line probe (`kwcoco stats
pred_boxes.kwcoco.zip` + `[a for a in d.anns.values() if 'bbox' not
in a]`) to confirm.

This is a duplicate of a class of bug the 04 script already
*anticipated* on the **true-side** (the simplified test GT had the
same issue earlier — `04_eval_on_test.sh:50` calls it out for the
mock dispatcher). The predicted side was not similarly guarded.

### Fixes (proposed; not yet landed at the time of writing)

1. In `_train_deimv2_variant.sh` generated train.yml, add
   `postprocessor.num_top_queries: <clamp>` where
   `clamp = min(upstream_num_top_queries, num_queries * num_classes)`.
   For shitspotter (num_classes=1), that's just
   `min(300, num_queries)`. Mirror the same override block we already
   use for `eval_spatial_size`.
2. Either (a) filter prediction anns without `bbox` in
   `cli_predict_boxes` before writing the kwcoco file, or (b) tolerate
   the missing field in the evaluator probe by adding
   `--skip_invalid_anns` if `kwcoco coco_eval` grows one. (a) is the
   cleaner fix — the box list is the deliverable; if there's no box,
   there's no detection.
3. Append both gotchas to `_train_deimv2_variant.sh`'s error-helper
   "Common causes & fixes" block so the next sweep run prints the
   fix hint inline with the traceback.

### Durable lesson

**The matrix-row failures the audit was supposed to prevent shift
when the matrix shape shifts.** The earlier audit fronted dep
problems, YAML composition, and the HGNetv2 fixed-size constraint —
all *cross-variant* issues. These two new bugs are *intra-variant*:
they only surface on a specific combination of (variant, dataset
shape, evaluator backend). The matrix didn't include them because we
had only run deimv2_n at one input size before the big sweep.

Two follow-ups in `dev/benchmark-candidates/pipeline-bootstrap-questions.md`
(Q5, Q6) capture the invariants in question form.

### Validation status

* `08_status.sh` (added `ba5a5a3`) — verifies on disk against the
  real sweep dir.
* The proposed fixes are *not* committed. The sweep is currently
  on cell `deimv2_n@640 fixed` (epoch 17, COCO AP@.50 = 0.554) and
  letting it finish before re-running with the fixes is cheaper than
  killing it.

---

## 2026-05-11: mobile_app_training_v4 — four cross-library API failures the agent could not have caught

### Trigger

Scaffolded `experiments/mobile_app_training_v4/` from a VM with no GPU and
no shitspotter Python env on PATH. All design + syntax + import-check
work passed in the VM. The user then ran `run_all.sh` on the host and
hit four runtime failures in sequence — each one a wrapper-API misuse
that compiled fine and parsed fine but exploded at first invocation.

### Symptoms

1. `gdown --fuzzy` →
   `__main__.py: error: unrecognized arguments: --fuzzy`
2. `kwimage.imresize(image, (W, H), interpolation='area')` →
   `cv2.error: Failed to allocate 44947419955200 bytes` on the first
   image. The (W, H) tuple was interpreted as the *scale* parameter,
   not `dsize`.
3. `kwimage.imwrite(fpath, image, imwrite_params=[('JPEG_QUALITY', 90)])` →
   `cv2.error: imwrite() got an unexpected keyword argument 'imwrite_params'`.
   kwimage forwards `**kwargs` straight to `cv2.imwrite`, which expects
   a flat `params=[INT_FLAG, INT_VALUE, ...]` list, not a name=value
   kwarg, and definitely not whatever the kwimage-specific name was.
4. `gdown` failure-mode caching: when Drive returned an HTML "quota
   exceeded" stub, the saved `.pth` was a few KB. The next run's
   `if [ -f "$dst" ]` short-circuit happily reused the bogus file as
   "downloaded".

### Root cause

All four are the same shape: **shelling out to or wrapping a
third-party API without verifying the call signature in the version
actually installed.** The agent had no way to test against the real
runtime — a hosted VM with no GPU and no shitspotter env can't
exercise gdown, kwimage, or DEIMv2 — so every API call became a
bet against an assumed signature.

The specific assumptions that broke:

* `gdown --fuzzy` was canonical in 4.x/5.x; gdown 6.0.0 (released
  2026) dropped it because the URL parser now handles every form
  natively. Old example commands transplanted into v4 docs without
  re-checking the upstream changelog.
* `kwimage.imresize`'s second positional is `scale`, not `dsize`.
  The call read like English ("imresize this image to (1280, 960)")
  and the misuse wasn't a type error — it was a value semantics error
  that compiled fine and only surfaced as a multi-TB allocation.
* `kwimage.imwrite` forwards `**kwargs` to `cv2.imwrite`. The
  documentation for the friendly knob (JPEG quality) lives in cv2's
  C++ docs. The pythonic-looking `imwrite_params=[('NAME', value)]`
  form was hallucination, neither kwimage's API nor cv2's.
* `gdown` writes whatever Drive returns to the destination path,
  including HTML error pages, with exit code 0. The "did the file
  download?" question is *not* answered by `[ -f "$dst" ]`.

### Fix

* `00_setup.sh`: drop `--fuzzy`, pass the bare file ID (works in
  every gdown version), and require the saved file to be ≥ 1 MiB
  before treating it as cached. SHA `8e905fb`.
* `tile_kwcoco.py:_resize_with_long_side`: pass `dsize=` explicitly
  to `kwimage.imresize`. SHA `985b5d6`.
* `tile_kwcoco.py:_imwrite`: drop the spurious `imwrite_params`
  kwarg; build the cv2-style flat `params=[cv2.IMWRITE_JPEG_QUALITY,
  q]` list when JPEG, fall through to defaults otherwise. SHA below.

### Durable lesson

When writing wrapper code from inside an environment that cannot
run the wrapped library, **the call site is at risk for every
positional, every kwarg name, every CLI flag**. Mitigations, in
descending order of value:

1. **Prefer keyword arguments at every wrapper call site, even when
   the positional form looks self-explanatory.** `kwimage.imresize(
   img, dsize=(W, H))` makes the next bug-of-its-kind a `TypeError`
   instead of a 45 TB allocation.
2. **Treat external-CLI defaults as version-bound.** Pin the version
   in setup, or re-read the upstream `--help` before quoting flags
   from memory. `--fuzzy` was load-bearing for two years; that's how
   long the agent's mental model can be wrong.
3. **Never trust file-presence as success for a downloaded artifact.**
   Min-size guard (this fix), or a hash check (better), or both.
4. **The first run on the real host is the test suite.** Plan for it.
   The four bugs above are exactly what a 5-minute smoke run would
   have caught — the v4 `V4_RUN_ALL_SMOKE=1` mode now exists for
   that reason. Use it before the full sweep.

### Candidate follow-up

* If a similar wrapper cluster appears again, promote
  "wrapper API misuse" to a benchmark candidate in
  `dev/benchmark-candidates/`. For now the four examples above are
  variations on the same theme; one entry covers it.
* The kwimage-imresize footgun in particular is generic — any agent
  writing wrappers around kwimage should default to `dsize=` /
  `scale=` keyword form rather than the positional shortcut.

### Validation status

After all four fixes, `run_all.sh` re-run by the user.
Step 0 reused the (real, > 1 MiB) DEIMv2 weights and step 1
proceeded past the first image without crashing. Subsequent steps
were not yet exercised at the time of writing this entry.

---

## 2026-05-10: Phone-app v2 scaffold — five hard invariants in one run

### Trigger

User-launched autonomous run with a 2-hour budget to scaffold a
KMP+Compose replacement for the .NET MAUI app at `tpl/poopdetector/`.
GOAL.md target: Milestone 0 (decision) minimum, Milestone 1 (skeleton)
soft success, Milestone 2 (real model backend) 90%+ true success.

### Symptoms

Five categories of subtle failure that surfaced during the run:

1. **ONNX shape mismatch** — `ai.onnxruntime.OrtException: Got
   invalid dimensions for input: images, index: 2 Got: 640 Expected:
   416` from deep inside `session.run`, not at construction time.
2. **Overlay misalignment** — Detection boxes rendered in a different
   region of the screen than the live camera image.
3. **Rotation drift** — On Pixel 5 portrait-locked, the camera buffer
   arrives at `rotationDegrees=90`, but the model produces detections
   in buffer (landscape) coordinates while the preview is rotated to
   portrait.
4. **YOLOX-nano OOD over-fire** — Stock dog photo produces a 0.757
   confidence "poop" detection.
5. **Bloated APK** — First `assembleDebug` produced 82 MB.

### Root cause

1. ORT defers tensor-shape validation until first `run()`. Our
   backend was constructed with a wrong-sized `ModelSpec` and the
   error only surfaced on the first frame.
2. CameraX `PreviewView` defaults to `FILL_CENTER`; our shared
   `DetectionOverlay` defaulted to a min-scale letterbox
   (effectively `FIT_CENTER`). Two coordinate spaces, same `Box`.
3. CameraX delivers buffers in native sensor orientation and
   reports `rotationDegrees` separately. PreviewView handles the
   display rotation internally; the overlay didn't.
4. The shipped YOLOX-nano was trained on pre-cropped patches; full-
   image OOD inputs produce high-confidence garbage.
5. AGP defaults include `armeabi-v7a`, `arm64-v8a`, `x86`, `x86_64`
   for every native `.so`. ONNX Runtime ships ~20 MB per ABI.

### Fix

1. Added `validateInputShape()` to both ONNX backends that reads
   `session.inputInfo[name].info` and asserts H/W match the spec,
   with a Kotlin-side error referencing the `modelId`.
   **Fix SHA:** `21a3b43` (pre-error at `92edfe4`).
2. Added `OverlayScaleMode` enum + `CameraSurface.overlayScaleMode`
   property. Android surface declares `FILL_CENTER`, desktop still-
   image declares `FIT_CENTER`.
   **Fix SHA:** `1eba4b8` (pre-error at `40197ac`).
3. Added pure-Kotlin `BoundingBox.rotated(degrees, w, h)` and rotated
   every detection plus the frame W/H before pushing into AppState.
   **Fix SHA:** `feca293` (pre-error at `e82ccc3`).
4. Documented the OOD behaviour in `ModelSpec.notes` and in
   `docs/004_kotlin_python_parity.md`. Did not change the detection
   pipeline.
   **Fix SHAs:** `fe1b1c8` (docs) + `eb85008` (ModelSpec.notes)
   (pre-error at `92edfe4`).
5. Set `ndk.abiFilters = setOf("arm64-v8a")` for Pixel 5. APK shrank
   82 MB → 29 MB.
   **Fix SHA:** `8e4f892` (pre-error at `e82ccc3`).

### Durable lesson

Every cross-stack ML deployment task touches at least *three* fragile
boundary layers that don't show up as compile errors:

- **Shape boundary** between ModelSpec and the real ONNX file (fails
  late inside native code).
- **Coordinate boundary** between the buffer the detector sees and
  the display the user sees (manifests as silent visual offsets, not
  as exceptions).
- **Training-data boundary** between what the model was trained on and
  what the camera ingests (silent over- or under-firing).

Make the failure mode loud at the earliest possible point in each
boundary, and document the OOD-behaviour story so the next agent
doesn't try to "fix" it as a regression.

A fourth boundary — **packaging** — is also worth front-loading.
AGP's "ship every ABI" default is almost always wrong for a known
target.

### Candidate follow-up

Each of the five symptoms is a benchmark candidate in
`dev/benchmark-candidates/app-deployment-questions.md`.

Cross-references to consider for further candidates:

- **TFLite parity** — when LiteRT gets wired up, the same five
  boundary layers will produce *different* failure modes. Re-run
  the candidate template against LiteRT specifically.
- **Front camera vs back camera** — the v2 app has a `useFrontCamera`
  toggle. Front cameras on most phones report a *mirrored* preview
  in PreviewView but unmirrored buffers in ImageAnalysis. The
  overlay alignment story is incomplete until that case is
  validated on a real phone.
- **Quantised models** — the registered `*-float16.onnx` variant has
  a different output dtype and currently isn't loaded. When it is,
  the YOLOX postprocess will need a dtype-aware path.

### Validation status

What ran on the VM (Linux desktop, JVM CPU):

- 122 unit tests across 28 files, all green.
- ONNX smoke test against the real
  `tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx`.
- `scripts/python_reference_compare.py` against the same model +
  `dog.jpg`; top-3 detections within ~0.01 of Kotlin output.
- `compare` CLI against three registered models on the same image.
- Both `assembleDebug` (29 MB) and `assembleRelease` (20 MB).

What did **not** run on the VM (requires phone):

- Live FPS measurement, NNAPI delegate verification, preview
  smoothness, thermal/battery behaviour, real-world over-fire
  observation, front-camera mirroring, on-device failure-case
  capture.

Each is documented as such in
`dev/journals/2026-05-10_phone_app_kmp_scaffold.md` and
`tpl/shitspotter-phone-app/docs/001_build_run_validate.md`.

---

## 2026-05-15: Compose gesture pass-through — two independent bugs that look identical

### Trigger

Adding pinch-to-zoom and per-photo swipe navigation to the ShitSpotter
phone app photo viewer (`PhotoViewer` in `AppScreen.kt`). User reported
"pinch to zoom and swipe to move to the next photo do not work."

### Symptoms

After adding `PhotoDetectionOverlay` (sibling `Box` with `fillMaxSize` +
`detectTapGestures`) and `Modifier.transformable` on the pager page
content Box, both single-finger swipe and two-finger pinch stopped
working completely. Detection box tapping and FN box drawing still worked.

### Root cause

Two independent bugs with identical surface symptoms:

**Bug 1 — `detectTapGestures` always consumes pointer-down.**
`detectTapGestures` internally calls `awaitFirstDown().also { it.consume() }`.
Because the overlay is higher z-order and processes events first, every
single-finger down was consumed before reaching the pager's scroll
handler or the page content's gesture handler.

**Bug 2 — `Modifier.transformable` always consumes single-finger drag.**
`transformable` wraps `detectTransformGestures`, which calls
`event.changes.forEach { it.consume() }` once movement exceeds
`touchSlop` — even for single-finger horizontal drags at `zoomScale==1f`.
So even after fixing Bug 1, the pager's `scrollable` still never saw a
swipe because the child page Box's `transformable` consumed it first
(Main pass: child before parent).

The initial wrong hypothesis was that the overlay was the sole cause.
Fixing Bug 1 alone was insufficient; Bug 2 independently caused the same
failure for different reasons.

### Fix

Bug 1: Replace `detectTapGestures` with a custom `awaitEachGesture` loop
using `awaitFirstDown(requireUnconsumed = false)`. Only consume the *up*
event when a tap actually lands on a detection or FN box.

Bug 2: Replace `rememberTransformableState + Modifier.transformable` with
a custom `pointerInput` that consumes only when appropriate:
- 2-finger touch → consume (pinch zoom)
- 1-finger touch + `zoomScale > 1.05f` → consume (pan)
- 1-finger touch + `zoomScale == 1f` → no consume (let pager swipe)

Additional compile note: `PointerInputChange.positionChanged()` is
absent in Compose Multiplatform 1.7.0 `commonMain`; use
`(p.position - p.previousPosition).getDistance() > 0f` instead.

### Durable lesson

Any `pointerInput` modifier on a composable that overlaps a scrollable
container (pager, LazyColumn, ScrollView) must be written with explicit
consumption logic. The "safe" Compose gesture APIs (`detectTapGestures`,
`transformable`) eagerly consume events and cannot safely coexist with
ancestor or sibling scrollables without modification.

Rule of thumb: if a composable's gesture handler should "pass through"
some events, write a custom `awaitEachGesture` loop and only call
`change.consume()` when you have confirmed you are handling that gesture.

### Candidate follow-up

Two benchmark candidates added to
`dev/benchmark-candidates/app-deployment-questions.md`:
- "Compose `Modifier.transformable` silently blocks `HorizontalPager` swipe" (Level B)
- "`detectTapGestures` in a full-screen overlay consumes pointer-down, breaking all gesture pass-through" (Level A)
