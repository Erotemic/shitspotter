# Benchmark candidates — pipeline bootstrap

Hard-problem invariants discovered during the v4 mobile-app training
pipeline build (2026-05-11). Per [`../AGENT_BENCHMARK_DISCIPLINE.md`](../AGENT_BENCHMARK_DISCIPLINE.md),
each candidate is the distillation of a *real* failure into a
question a future agent could plausibly mis-answer.

The thread tying these together: a v4 sweep cell costs minutes of
GPU time *just to surface the next missing dep*. Front-loading the
discovery into a 30-second `00_setup.sh` pre-flight and a
CPU-only smoke test changes the iteration cost from "next sweep
restart" to "next probe call."

---

## Q1 — YAML composition by bash heredoc + indent arithmetic

Status: draft
Level: B
Tags: config-generation, yaml, parser-divergence, train-eval-export-parity
Requires full dataset: no
Requires trained weights: no
Pre-error SHA: `03708dc` (the v4_mock dispatcher already worked, so we
                          thought the YAML composition was fine)
Fix SHA:       `f5d96aa` (dedent MULTISCALE_BLOCK from 4 to 2 spaces)

### Source context

`experiments/mobile_app_training_v4/_train_deimv2_variant.sh` builds
the per-cell DEIMv2 train.yml by interpolating `$MULTISCALE_BLOCK`
into a larger heredoc. The block had 4-space indent intended to land
the `collate_fn:` key as a sibling of `dataset:` under
`train_dataloader:`. With 4 spaces of indent in the inner heredoc and
2 spaces of indent on its surroundings, YAML actually parsed
`collate_fn` as a *child* of `dataset` — and DEIMv2's
`workspace.create()` then forwarded `collate_fn` as a kwarg to
`CocoDetection.__init__`, which has no such parameter:

```text
TypeError: CocoDetection.__init__() got an unexpected keyword argument 'collate_fn'
```

The trainer survived `bash -n`, survived `yaml.safe_load` (the
document was *valid* YAML — just wrong), and only failed deep inside
`engine/core/workspace.py` at module-build time.

### The hard question

> When generating structured config files (YAML, JSON, TOML) by string
> interpolation, what is the cheapest way to verify the *parsed
> structure* matches the intent — not just that the file parses?

### Invariant to preserve

For `train_dataloader` in v4-generated DEIMv2 configs:

```python
yaml.safe_load(open('train.yml'))['train_dataloader'].keys()
# must include {'total_batch_size', 'num_workers', 'dataset', 'collate_fn'}
# and dataset.keys() must NOT include 'collate_fn'
```

### Acceptance criteria

A pytest fixture that:

1. Invokes `_train_deimv2_variant.sh` end-to-end (or its config-build
   stage in isolation) with a full sweep of `(variant, train_policy,
   input_size)` combinations.
2. `yaml.safe_load`s the resulting `train.yml`.
3. Asserts the structural invariant above.
4. Plus: invokes DEIMv2's `engine.core.YAMLConfig` to load the same
   file and reach the model-build step without raising.

The test should fail clearly when an interpolation indent is wrong,
*before* the trainer is ever launched on a GPU.

### Why this is a benchmark question

A capable agent could:

* Get the YAML wrong by `s/2/4/g` somewhere.
* Get it wrong by writing a sibling key inside `dataset:` but visually
  thinking it's under `train_dataloader:`.
* Add a new heredoc-interpolated block in a future patch and reproduce
  the same indent confusion.

Catching it requires either (a) testing the parsed structure, not just
the byte stream, or (b) running through the framework that consumes it.
Both are easy; neither is reflex behaviour.

---

## Q2 — Cross-library transitive runtime deps

Status: draft
Level: A
Tags: env-bootstrap, undeclared-deps, fail-fast, pipeline-orchestration
Requires full dataset: no
Requires trained weights: no
Pre-error SHA: `03708dc` (sweep would crash on first cell with
                          ModuleNotFoundError 30 s in)
Fix SHA:       `f5d96aa` (00_setup.sh installs all of DEIMv2's
                          requirements.txt + onnx trio + gdown)

### Source context

The v4 pipeline composes 6+ third-party libraries (DEIMv2, kwimage,
kwcoco, geowatch, torch, gdown). Each has its own undeclared / lazy
runtime deps:

* `gdown 6.x` dropped `--fuzzy` (CLI surface change)
* `torch.onnx.export` on torch ≥ 2.5 imports `onnxscript` at
  function-call time, even when `dynamo=False`
* `geowatch.__init__` hard-imports `osgeo` (GDAL python bindings)
  at package init, blocking lazy users
* `shitspotter.cli.simplify_kwcoco` hard-imports
  `geowatch.utils.util_kwimage` at top of `main()`
* `kwimage.imresize(interp='area')` only works with cv2; skimage
  fallback raises NotImplementedError
* DEIMv2's `tpl/DEIMv2/requirements.txt` (faster_coco_eval,
  calflops, transformers, tensorboard, scipy) is not declared
  by the wrapping shitspotter package

Each missing piece killed a sweep cell ~30 s in, after model
construction but before any meaningful work. Each fix surfaced the
next.

### The hard question

> When orchestrating a pipeline that spans N third-party packages,
> what is the right place + shape for a pre-flight environment audit
> that catches all O(N) categories of missing-dep bugs *before* the
> first GPU minute is spent?

### Invariant to preserve

For any clean host that has the v4 sweep's *direct* deps installed,
running `bash 00_setup.sh && bash 02_sweep.sh` to completion should
not surface any `ModuleNotFoundError` from any *transitive* library
the pipeline composes.

### Acceptance criteria

A pre-flight check that, given a fresh host with only the
shitspotter package + DEIMv2 submodule installed:

1. Probes for each of: gdown, onnxscript, onnx, onnxruntime, every
   line of `tpl/DEIMv2/requirements.txt`.
2. Triggers the following imports inside a subprocess and reports
   each as "OK" or "missing X":
   - `from shitspotter.cli.simplify_kwcoco import *`
   - `from torch.onnx import export`
   - `import faster_coco_eval`
   - `from osgeo import gdal`
3. Loops with `pip install` for each missing module that has a
   declared install path.
4. Exits non-zero if any required dep can't be acquired automatically,
   and prints the manual install command.

A pytest equivalent:

```python
def test_v4_sweep_pre_flight():
    assert importlib.util.find_spec('gdown') is not None
    assert importlib.util.find_spec('onnxscript') is not None
    assert importlib.util.find_spec('faster_coco_eval') is not None
    # ... etc — fail loud if any are missing
```

### Why this is a benchmark question

Adding a new cell to the v4 sweep, or a new third-party submodule,
trivially regresses this. A capable agent will land the new code,
ship a setup script that installs *its own* deps, and not realise the
new module's runtime imports pull in packages the rest of the
pipeline didn't already need. The fix is mechanical (`pip install`),
the discovery is wasteful (one whole sweep iteration per missing
dep). Front-loading it into the audit is the lesson.

---

## Q3 — Upstream architectural constraints leak through config

Status: draft
Level: B
Tags: model-architecture, config-defaults, train-eval-export-parity
Requires full dataset: no
Requires trained weights: yes (DEIMv2 N pretrained ckpt + tile bundle)
Pre-error SHA: `f5d96aa` (variant-keyed defaults but multiscale by default)
Fix SHA:       (to be assigned after this commit)

### Source context

v4's per-cell config generator defaults `V4_TRAIN_POLICY=multiscale`
across all variants. For the DINOv3-backed family (deimv2_s/m/l/x)
this is fine — the encoder dynamically interpolates positional
embeddings per batch. For the HGNetv2 hybrid encoder (Atto/Femto/
Pico/N) it's a hard architectural mismatch: pos_embed is pre-baked
at `eval_spatial_size` and doesn't interpolate, so a multi-scale
collate produces:

```text
RuntimeError: The size of tensor a (121) must match the size of
tensor b (100) at non-singleton dimension 1
```

mid-encoder, at first batch.

Upstream DEIMv2 knows this — every HGNetv2 variant ships with
`base_size_repeat: ~` in its config. Multi-scale is opt-in only for
DINOv3-backed variants (`base_size_repeat: 20`). I overrode this
without checking why.

### The hard question

> When `__include__`-ing or programmatically extending an upstream
> framework config, which fields encode *architectural* constraints
> (must not be changed without code work) vs *training hyper-
> parameters* (free to tune)?

### Invariant to preserve

For any v4 sweep cell using an HGNetv2-backed variant
(`deimv2_atto`, `_femto`, `_pico`, `_n`), the generated train.yml
must set `base_size_repeat: ~` (i.e. `V4_TRAIN_POLICY=fixed`). For
DINOv3-backed variants (`deimv2_s`, `_m`, `_l`, `_x`), multi-scale
is permitted.

### Acceptance criteria

A pytest fixture that:

1. Generates a per-cell config for every (variant, train_policy)
   pair in `02_sweep.sh`'s default matrix.
2. Asserts `base_size_repeat == None` for HGNetv2 variants regardless
   of requested train policy.
3. Asserts the train policy override is honoured for DINOv3-backed
   variants.

Bonus: launch each generated config through `engine.core.YAMLConfig`
+ `cfg.model.deploy()` on CPU with a tiny dummy batch to confirm
the encoder accepts the configured shape. This catches the same
class of bug for *any* future architectural mismatch, not just this
specific one.

### Why this is a benchmark question

Adding a new variant family (e.g. a future `deimv2_v3_x` with a
totally different encoder) trivially reintroduces this. The lesson
is *always check upstream configs for which fields are non-default
in the upstream's own per-variant overrides* — those are the fields
that encode architectural constraints.

---

## Q4 — RLIMIT_NOFILE for torch dataloader IPC

Status: draft
Level: B
Tags: env-bootstrap, torch-multiprocessing, fail-late, pipeline-orchestration
Requires full dataset: no
Requires trained weights: yes (any trained DEIMv2 cell + tile bundle)
Pre-error SHA: `819a27a` (the day's bootstrap fixes done; trainer
                          starts but dies a few iterations in)
Fix SHA:       `aec1b82` (raise RLIMIT_NOFILE in sweep + trainer
                          wrappers; optional file_system sharing
                          strategy as a fallback)

### Source context

torch dataloader workers pass tensors back to the main process via
`torch.multiprocessing.reduce_storage`, which opens a unix-domain
socket per shared tensor (the `file_descriptor` sharing strategy).
With 4 workers × batch=128 × O(many) shared tensors per batch, FD
usage climbs past the default 1024 soft limit a few iterations into
training:

```text
OSError: [Errno 24] Too many open files
  File ".../torch/multiprocessing/reductions.py", line 616, in reduce_storage
  File ".../multiprocessing/reduction.py", line 198, in DupFd
```

The crash happens *deep inside multiprocessing*, not where the
trainer lives. Stack trace points to `DupFd` rather than to anything
the user's code touches, so the root cause is non-obvious.

### The hard question

> Which torch-runtime-level resource limits (FDs, shared-memory
> bytes, NCCL timeouts, CUDA allocator behaviour) are the trainer's
> wrapper script responsible for setting before launching the
> framework? Which are the framework's job?

### Invariant to preserve

Any v4 sweep cell launched on a fresh shell must not crash with
`OSError: [Errno 24]` from inside torch.multiprocessing. The wrapper
script must raise `RLIMIT_NOFILE` to a value safe for the
configured (workers × batch × tensor count) at the upper end of the
sweep matrix.

### Acceptance criteria

A pytest fixture that:

1. Captures the soft FD limit before invoking
   `_train_deimv2_variant.sh` in a subshell.
2. Confirms that the script raises the soft limit to ≥ V4_FD_LIMIT
   before the trainer subprocess is spawned.
3. Does so silently when the limit is already ≥ V4_FD_LIMIT.
4. Falls back gracefully (with a clear warning, no abort) when the
   shell hard limit is below V4_FD_LIMIT.

Bonus: smoke-train a v4_mock cell with `num_workers=8, batch=64` to
verify no FD storm under the configured limit.

### Why this is a benchmark question

A future agent could:

* Add a new sweep cell with bigger batch / more workers and not
  re-check the FD math.
* Refactor the wrapper scripts and forget to re-raise the limit.
* Move the trainer launch into a helper that doesn't inherit the
  ulimit -n call.

Each regression silently produces a "trains for a few iterations
then dies in DupFd" failure mode that's expensive to debug from the
stack trace alone. The lesson: **treat RLIMIT_NOFILE as a contract
between the wrapper script and the framework**, declared once near
the launcher and never assumed to come from the user's shell.

The fallback knob (`V4_TORCH_MP_SHARING=file_system`) is a useful
escape valve for genuinely huge batch × worker configs where even
65536 isn't enough — but it costs throughput, so default off.

---

## Q5 — Postprocessor `num_top_queries` vs `num_queries × num_classes`

Status: draft
Level: B
Tags: config-generation, upstream-architectural-constraint, single-class-detector, train-eval-export-parity
Requires full dataset: no
Requires trained weights: no (the topk runs in the very first val pass)
Pre-error SHA: `cbb2d9f` (per-(variant,input_size) batch defaults — sweep
                          ran into this on first long sweep)
Fix SHA:       _none yet_ (proposed in
                          `dev/journals/lessons_learned.md`
                          §2026-05-12)

### Source context

DEIMv2's postprocessor entry op is:

```python
# tpl/DEIMv2/engine/deim/postprocessor.py:59
scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
```

`scores.flatten(1)` has shape `(B, num_queries * num_classes)`. The
default `num_top_queries=300` (from `base/dfine_hgnetv2.yml:76`) was
calibrated for the upstream COCO experiments where
`num_classes=80`, so `num_queries × 80` is always well above 300
even for the smaller HGNetv2 variants:

| variant | num_queries | × 80 (COCO) | × 1 (shitspotter) |
|---------|-------------|-------------|--------------------|
| atto    | 100         | 8000        | **100** |
| femto   | 150         | 12000       | **150** |
| pico    | 200         | 16000       | **200** |
| n       | 300         | 24000       | 300    |
| s/m/l/x | 300         | 24000       | 300    |

For a **single-class** detector (shitspotter has `num_classes: 1`),
any HGNetv2 variant smaller than `n` flips the inequality. The
v4 generated train.yml hard-codes `num_classes: 1` but inherits
`num_top_queries: 300` from upstream — and the first val pass of
the first epoch dies in `torch.topk`.

### The hard question

> When a v4 generated config alters one upstream invariant
> (`num_classes`, `eval_spatial_size`, etc.), which other upstream
> defaults silently couple to it — and how do you catch the coupling
> before the first GPU minute?

The same shape of question covers `eval_spatial_size` ↔ encoder
`pos_embed` (already documented as Q3), and now
`num_classes` ↔ `num_top_queries`. The general invariant is:

```python
num_top_queries <= num_queries * num_classes
```

Failing this manifests as `RuntimeError: selected index k out of
range` *deep inside* `torch.topk` — not as a config-build error.

### Invariant to preserve

For every v4 generated train.yml, after the postprocessor block is
resolved:

```python
cfg['DEIMPostProcessor']['num_top_queries'] <= (
    cfg.get('num_queries', 300) * cfg.get('num_classes', 80)
)
```

The clamp is correct: emit
`num_top_queries = min(upstream_value, num_queries * num_classes)`
in the generated config when the inequality would otherwise be
violated.

### Acceptance criteria

A pytest fixture that:

1. For each `(variant, num_classes)` pair the sweep can emit, calls
   the v4 trainer's config-build step (no GPU needed).
2. Asserts the inequality above against the resolved YAML.
3. Drives a CPU smoke run that exercises the postprocessor on a
   synthetic batch with the resolved `num_queries × num_classes`
   shape and confirms it completes without raising.

Bonus: the v4 trainer's error-helper block (printed on non-zero
trainer exit) explicitly names the `RuntimeError: selected index k
out of range` symptom and points at the clamp fix.

### Why this is a benchmark question

A capable agent could:

* Switch detector architectures without re-deriving the invariant
  (the COCO defaults are always satisfied at 80 classes).
* Generate the train.yml from a template that overrides only the
  obvious knobs (`num_classes`, `output_dir`, `eval_spatial_size`)
  and inherit the silently-coupled ones.
* Read the postprocessor source, see `num_top_queries=300` as the
  default, and forget that the input it operates on has shrunk.

The lesson chains directly with Q3: **upstream model configs encode
multi-knob invariants**, and changing any one knob without
re-checking its partners produces a deep-stack error that costs a
full sweep cell to surface.

---

## Q6 — kwcoco evaluator requires `bbox` on every coerced annotation (true + predicted)

Status: draft
Level: A
Tags: kwcoco-eval, predict-eval-parity, single-class-detector, dataset-hygiene
Requires full dataset: no
Requires trained weights: a real detector run — the bug needs at least
                          one zero-area / degenerate prediction
Pre-error SHA: `cbb2d9f` (per-(variant,input_size) batch defaults — sweep
                          surfaced this on the n@320 cell first)
Fix SHA:       _none yet_ (proposed in
                          `dev/journals/lessons_learned.md`
                          §2026-05-12)

### Source context

`kwcoco.coco_evaluator.CocoEvaluator._coerce_dets` runs the same
coercion against both the true GT and the predicted dataset:

```python
# kwcoco/coco_evaluator.py:510
boxes=kwimage.Boxes([a['bbox'] for a in anns], 'xywh'),
```

It calls itself recursively, first on the predicted dataset, then on
the true. The first invocation died for the v4 sweep's
`deimv2_n_tile_g2_fixed_320x320` cell:

```text
KeyError: 'bbox'
  -> cell deimv2_n_tile_g2_fixed_320x320 FAILED at: eval
```

after `cli_predict_boxes` cleanly wrote 118/118 prediction images.
The same predictor at 416 and 512 produced eval AP fine
(0.4056, 0.4765); only 320 tripped it. Strongest hypothesis: at
320×320 input the model emits at least one prediction that
degenerates to zero-area after rescale, and the kwcoco-side box
serialisation stores the ann without a `bbox` key instead of
filtering it.

### The hard question

> When a downstream tool (here `kwcoco coco_eval`) refuses
> annotations with missing optional fields, where does the filter
> belong — at the predictor (every emitted annotation has a `bbox`),
> at the evaluator (skip-and-warn), or at the canonical
> "predict-then-eval" wrapper in between?

### Invariant to preserve

For every kwcoco file written by `cli_predict_boxes` and consumed by
`04_eval_on_test.sh`:

```python
import kwcoco
d = kwcoco.CocoDataset(pred_boxes_fpath)
missing = [a for a in d.anns.values() if 'bbox' not in a]
assert not missing, f"{len(missing)} predicted anns are missing 'bbox'"
```

The predictor is the right place to enforce it: an ann with no box
isn't a detection. Either drop the ann at emission time, or emit an
explicit `bbox=[x,y,0,0]` and let the evaluator's IoU filter handle
it.

### Acceptance criteria

A pytest fixture that:

1. Runs `cli_predict_boxes` against a small kwcoco fixture using a
   model whose first-cell predictions are *known* to include
   degenerate boxes (e.g. a smoke-trained mock detector at very low
   resolution).
2. Loads the resulting pred kwcoco file with `kwcoco.CocoDataset`
   and asserts every ann has a `bbox` field.
3. Pipes the same file into `kwcoco coco_eval --true_dataset
   <test> --pred_dataset <pred>` and asserts the call exits zero.

Bonus: the v4 trainer's error-helper block extends to the eval
script, so the next sweep run prints the fix hint inline.

### Why this is a benchmark question

A capable agent could:

* Trust that "a kwcoco file written by our predictor" satisfies all
  downstream tools' assumptions.
* Filter degenerate boxes in the COCO `score` path but not in the
  ann-write path.
* Add a guard in the wrong layer (e.g. catching the `KeyError` in
  the v4 eval script instead of fixing the predictor) and ship a
  silent under-counting bug to the next agent who reads the AP
  number.

This chains with Q5: small variants at small input sizes are more
likely to produce degenerate predictions, *and* are the same
variants that trip Q5's topk inequality. Both bugs cluster on
"smallest model + smallest input" — exactly the cell the sweep is
ordered to run *first*.

---

## Composition note

Q1, Q2, Q3 chain: an agent who skips the pre-flight (Q2) won't
discover the YAML structural bug (Q1) until the trainer dies inside
the framework — at which point they may *also* trip the architectural
constraint (Q3) because they're rapidly iterating on the wrong
hypothesis. The cheapest defense is to run all three checks before
the first GPU minute.

Q5, Q6 chain with Q3 along a different axis: **single-knob overrides
on an upstream config silently break partner knobs**. Q3 is the
spatial version (`eval_spatial_size` ↔ encoder `pos_embed`); Q5 is
the head version (`num_classes` ↔ `num_top_queries`); Q6 is the
output-contract version (predictor ↔ evaluator agreement on
required ann fields). All three only surface mid-sweep on the
specific (variant, input_size) cells that violate the partner
invariant — never on the smoke test.

The v4 audit pass put a pytest suite in place
(`tests/mobile_app_training_v4/`, 31 tests) that covers Q2's probes
and Q1's structural assertions. Q3, Q5, and Q6 are not yet covered
by tests — adding the three together is the natural next step,
since the failure modes share a generative structure
("upstream-knob coupling we forgot to mirror").
