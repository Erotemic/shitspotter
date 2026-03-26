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

## Current priority

The current high-priority benchmark is narrower than the original scaffold:

* compare DINOv2 via OpenGroundingDINO against DINOv3 via DEIMv2,
* detector-only first, no SAM in the comparison,
* train on positives-only subsets of size `128`, `256`, and `512`,
* evaluate on prepared derivatives of the canonical full validation and test
  splits,
* apply the same preprocessing recipe to both families:
  * poop-only category view,
  * offline size reduction,
  * box simplification,
  * explicit manifests recording that these happened.

The benchmark is now config-tag aware:

* OpenGroundingDINO can be rerun under tagged configs, though the default is a
  single `baseline` tag so it remains a stable reference line.
* DEIMv2 is intended to carry most of the tuning work. Each DEIMv2 recipe gets
  a `config_tag`, writes into its own run subtree, and appears as its own curve
  in the aggregate plot.
* The current default DEIMv2 tags are:
  * `baseline`
  * `small_batch16`
  * `small_batch16_low_lr_all_0p8`
  * `small_batch16_low_lr_all_0p6`
  * `small_batch16_backbone_0p5`
  * `small_batch16_backbone_0p25`

That means new DEIMv2 tuning ideas can be added without overwriting earlier
results, and the analysis layer can compare `model_family + config_tag` against
training size.

That benchmark lives under the shared benchmark root:

```text
$DVC_EXPT_DPATH/small_data_tuning/dino_detector_benchmark/
```

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

### Recommended detector benchmark workflow

This is the end-to-end path to answer the current DINOv2-vs-DINOv3 question.

#### 1. Prepare the shared benchmark data once

```bash
PYTHON_BIN=/home/joncrall/.local/uv/envs/uvpy3.13.2/bin/python \
bash /home/joncrall/code/shitspotter/experiments/small-data-tuning/prepare_dino_detector_benchmark.sh
```

This writes:

* a benchmark-wide [`benchmark_manifest.json`](./prepare_dino_detector_benchmark.py),
* positives-only prepared train subsets for `128`, `256`, `512`,
* prepared validation and test sets derived from the canonical full splits,
* per-split metadata describing source statistics and prepared statistics.

#### 2. Run the OpenGroundingDINO line

```bash
PYTHON_BIN=/home/joncrall/.local/uv/envs/uvpy3.13.2/bin/python \
bash /home/joncrall/code/shitspotter/experiments/small-data-tuning/run_opengroundingdino_dino_detector_benchmark.sh
```

#### 3. Run the DEIMv2 line

```bash
PYTHON_BIN=/home/joncrall/.local/uv/envs/uvpy3.13.2/bin/python \
bash /home/joncrall/code/shitspotter/experiments/small-data-tuning/run_deimv2_dino_detector_benchmark.sh
```

#### 4. Aggregate and draw the train-size curves

```bash
PYTHON_BIN=/home/joncrall/.local/uv/envs/uvpy3.13.2/bin/python \
python /home/joncrall/code/shitspotter/experiments/small-data-tuning/analyze_dino_detector_benchmark.py \
    --benchmark_root /data/joncrall/dvc-repos/shitspotter_expt_dvc/small_data_tuning/dino_detector_benchmark \
    --out_dpath /data/joncrall/dvc-repos/shitspotter_expt_dvc/small_data_tuning/dino_detector_benchmark/analysis
```

Or run the whole chain:

```bash
PYTHON_BIN=/home/joncrall/.local/uv/envs/uvpy3.13.2/bin/python \
bash /home/joncrall/code/shitspotter/experiments/small-data-tuning/run_dino_detector_benchmark.sh
```

The analysis step writes:

* `benchmark_summary.tsv`
* `train_size_curve.png`
* `analysis_manifest.json`
* `run_links/` symlinks into the per-run artifact trees for inspection

The curves plot Box AP on the y-axis, training size on the x-axis, and
`model_family:config_tag` as the comparison hue.

### Extending DEIMv2 tuning

The DEIMv2 runner accepts a space-separated list of tagged config specs through
`DEIMV2_CONFIG_SPECS`. Each spec is encoded as:

```text
config_tag|main_lr_scale|backbone_lr_scale|train_batch_size|use_amp
```

Example:

```bash
DEIMV2_CONFIG_SPECS="
baseline|1.0|1.0|24|True
small_batch16|1.0|1.0|16|True
small_batch16_low_lr|0.8|0.8|16|True
small_batch16_backbone_tiny|1.0|0.25|16|True
" \
bash /home/joncrall/code/shitspotter/experiments/small-data-tuning/run_deimv2_dino_detector_benchmark.sh
```

This keeps the shared benchmark data fixed while letting the DEIMv2 detector
recipe evolve in a comparable, append-only way.

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
