# Small Data Tuning

This directory is a reproducible playground for fast-turnaround model
comparisons on ShitSpotter. The goal is not to replace the larger historical
experiments under [`experiments/`](../), but to make it cheap to answer focused
questions like:

* Does a detector family improve faster than another as we add more training
  images?
* Are we still seeing the same model ordering on a small but fixed benchmark
  cohort?
* Is a new optimization or preprocessing idea promising enough to justify a
  full-data rerun?

The design is intentionally manifest-first. Every run in this directory should
be traceable back to:

1. the exact source datasets,
2. the exact subset-selection recipe,
3. the exact subset image ids,
4. the exact model family wrapper and hyperparameters used for training,
5. the exact checkpoint-selection and evaluation logic used afterward.

## Scope

This scaffold seeds support for the model families that have mattered most in
recent ShitSpotter work:

* YOLO
* Mask R-CNN
* GroundingDINO / DINOv2-era detector tuning
* DEIMv2 / DINOv3 detector tuning
* DEIMv2 + SAM2 detector-plus-segmenter evaluation

The old experiment directories are still the source of truth for historical
results. This directory standardizes a smaller, faster experimental loop.

## Directory layout

```text
small-data-tuning/
    README.md
    common.sh
    choose_small_subsets.sh
    select_subsets.py
    run_all_small_experiments.sh
    run_yolo_small_experiment.sh
    run_maskrcnn_small_experiment.sh
    run_grounding_dino_small_experiment.sh
    run_deimv2_small_experiment.sh
    run_deimv2_sam2_small_experiment.sh
```

At runtime, artifacts are meant to live under the ShitSpotter experiment DVC
area, not in git:

```text
$DVC_EXPT_DPATH/small_data_tuning/
    cohorts/
        random_seed0_train0128_vali0064_test0064/
            cohort_manifest.json
            train.kwcoco.zip
            train.mscoco.json
            vali.kwcoco.zip
            vali.mscoco.json
            test.kwcoco.zip
            test.mscoco.json
    runs/
        <model_family>/
            <cohort_name>/
                <run_name>/
                    run_manifest.json
                    ...
```

That split is deliberate:

* `cohorts/` defines the benchmark data.
* `runs/` defines model- and hyperparameter-specific work.

If two agents disagree on training details but use the same cohort manifest,
their runs are still directly comparable.

## Experimental plan

### Phase 1: lock down subset selection

Subset selection must happen before any model tuning.

The default selector is intentionally simple and auditable:

* method: `random`
* seed: explicit and recorded
* split policy: sample each split independently
* stratification: positive-image vs negative-image stratified random sampling

That last point matters because pure random sampling can accidentally produce a
small cohort with too few annotated images. The current selector keeps the
selection rule simple while reducing that risk.

Future selectors should plug into the same manifest schema. Good candidates are:

* clustering and representative selection,
* active-learning style uncertainty picks,
* metadata-aware coverage balancing,
* hard-example mining from prior runs.

The rule is: a new selector is only acceptable if it still emits a complete
audit trail.

### Phase 2: benchmark small-data training curves

We want a family of cohorts, not only one tiny split. A useful default ladder is:

* train: `64`, `128`, `256`, `512`
* validation: fixed small validation cohort
* test: fixed small test cohort

This lets us compare:

* quality vs training-set size,
* runtime vs training-set size,
* sensitivity of each model family to limited data.

The validation and test cohorts should stay fixed while training size changes.
That keeps the performance curves interpretable.

### Phase 3: promote only promising ideas

A small-data win is not automatically a production win. This directory is meant
to act as a triage system:

* if a change loses here, it probably should not consume a full-data run yet;
* if a change wins here, it earns a larger-scale follow-up.

## Reproducibility rules

Every new wrapper or experiment in this directory should follow these rules:

* Always write a `run_manifest.json` before or alongside training.
* Always record the `cohort_manifest.json` path that defined the data.
* Avoid hidden shell-state dependencies.
* Prefer explicit environment variables with defaults over magic constants.
* Prefer append-only output directories over in-place mutation.
* If a stage can fail partway through, either:
  * write a canonical final artifact only on success, or
  * write explicit run-state metadata so partial artifacts are recognizable.

## Suggested workflow

### 1. Materialize the benchmark cohorts

```bash
PYTHON_BIN=/home/joncrall/.local/uv/envs/uvpy3.13.2/bin/python \
bash /home/joncrall/code/shitspotter/experiments/small-data-tuning/choose_small_subsets.sh
```

That wrapper calls [`select_subsets.py`](./select_subsets.py) with a deterministic
random-selection recipe and writes auditable cohort manifests under
`$DVC_EXPT_DPATH/small_data_tuning/cohorts/`.

### 2. Launch one model family on one cohort

Examples:

```bash
bash /home/joncrall/code/shitspotter/experiments/small-data-tuning/run_yolo_small_experiment.sh \
    --cohort_name random_seed0_train0128_vali0064_test0064

bash /home/joncrall/code/shitspotter/experiments/small-data-tuning/run_deimv2_small_experiment.sh \
    --cohort_name random_seed0_train0128_vali0064_test0064
```

### 3. Compare results across families

The wrappers are intentionally aligned around the same cohort structure so that
aggregation can be added later without rewriting every experiment family again.

## Model-family notes

### YOLO

The small-data wrapper mirrors the older YOLO-v9 experiment shape but writes its
dataset YAML and run metadata into a standardized small-data run directory.

### Mask R-CNN

The small-data wrapper uses `python -m shitspotter.detectron2.fit` with a
generated config file so that subset paths and solver settings are always
explicitly recorded.

### GroundingDINO / DINOv2-era detector

The wrapper keeps the same broad preparation pattern as the historical tuning
script:

* reroot to absolute paths,
* rewrite categories into a training-friendly layout,
* export the ODVG metadata expected by the external training code.

This is the most likely family to need environment-specific care, so the wrapper
is intentionally verbose.

### DEIMv2 / DINOv3 detector

The small-data wrapper reuses the more recent `foundation_detseg_v3` training
path, which already knows how to resize offline, export detector COCO, and cope
with brittle multi-GPU training by falling back to a single GPU when needed.

### DEIMv2 + SAM2

The detector+segmenter wrapper is intentionally scoped as a fast detector-first
experiment. By default it reuses the tuned `v5` SAM2 checkpoint so detector
changes remain the primary variable. That keeps the iteration loop short. If we
later want a small-data SAM2 tuning mode, it should be added as an explicit
option rather than silently changing the default comparison.

## Future work

This directory is meant to grow in a structured way. Good follow-on work:

* add a tabular aggregator over `run_manifest.json` files,
* add Dockerfiles or container recipes per model family,
* add richer subset selectors,
* add runtime / memory tracking to the standardized manifests,
* add checkpoint-selection helpers for all families, not only DEIMv2.

The important thing is to preserve the invariant that a future researcher can
open this directory with no chat context and reconstruct both the plan and the
artifact layout quickly.
