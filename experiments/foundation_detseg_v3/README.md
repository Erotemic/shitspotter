# Foundation Det+Seg V3

This experiment family adds a new detector-plus-segmenter bootstrap path for
ShitSpotter polygon annotation work.

Primary backend:

- `deimv2_sam2` = DEIMv2-M detector + SAM2.1 box-prompted refinement

Baseline backend:

- `maskdino` = MaskDINO-R50 direct instance segmentation

Expected environment variables:

- `SHITSPOTTER_DEIMV2_REPO_DPATH`
- `SHITSPOTTER_SAM2_REPO_DPATH`
- `SHITSPOTTER_MASKDINO_REPO_DPATH`
- `DVC_DATA_DPATH`
- `DVC_EXPT_DPATH`

Setup:

```bash
bash experiments/foundation_detseg_v3/setup_environment.sh
```

CPU smoke tests:

```bash
bash experiments/foundation_detseg_v3/cpu_smoke_tests.sh
```

Train DEIMv2 detector:

```bash
bash experiments/foundation_detseg_v3/train_deimv2_detector.sh
```

Run DEIMv2 + SAM2 on validation:

```bash
bash experiments/foundation_detseg_v3/run_deimv2_sam2_on_vali.sh
```

Run DEIMv2 + SAM2 on test:

```bash
bash experiments/foundation_detseg_v3/run_deimv2_sam2_on_test.sh
```

Train MaskDINO baseline:

```bash
bash experiments/foundation_detseg_v3/train_maskdino_baseline.sh
```

Run MaskDINO on validation:

```bash
bash experiments/foundation_detseg_v3/run_maskdino_on_vali.sh
```

Run MaskDINO on test:

```bash
bash experiments/foundation_detseg_v3/run_maskdino_on_test.sh
```

Bootstrap a new cohort of phone images:

```bash
COHORT_DPATH=/path/to/new/cohort \
PACKAGE_FPATH=/path/to/foundation_package.yaml \
bash experiments/foundation_detseg_v3/run_bootstrap_new_cohort.sh
```

Aggregate evaluation results:

```bash
TARGET_DPATH="$DVC_EXPT_DPATH/_foundation_detseg_v3" \
bash experiments/foundation_detseg_v3/aggregate_foundation_results.sh
```

Template packages live in `experiments/foundation_detseg_v3/packages/`. Copy or
rebuild them with `python -m shitspotter.algo_foundation_v3.cli_package build ...`
before running GPU evaluation.
