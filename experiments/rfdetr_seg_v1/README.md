# ShitSpotter RF-DETR Seg 2XL campaign

This directory contains only ShitSpotter policy and orchestration. Generic
tiling, cache, RF-DETR, prediction, and evaluation code lives in the sibling
`kwcoco_detector_kit` repository.

The workflow deliberately has two validation products:

- `train_validation_tiles.kwcoco.zip` is the fixed cached tile pool consumed
  cheaply at every RF-DETR validation epoch.
- canonical validation runs the selected checkpoint over the original 1,258
  validation images using KDK tiled native-mask inference, reconstructs masks
  in source coordinates, merges overlap duplicates, and then computes source
  image box/mask AP and zero-poop-image false-positive statistics.

The test split is frozen in `config.yaml` but must not be evaluated while
choosing data policy, checkpoints, thresholds, or mining rounds.

Run the local preflight first. It resolves split filenames relative to
`paths.data_root`, checks every referenced asset, samples real decodes, builds
an exhaustive real-data smoke corpus, and simulates the complete tile policy
without encoding the full candidate universe:

```bash
python experiments/rfdetr_seg_v1/driver.py verify-inputs
python experiments/rfdetr_seg_v1/driver.py census
python experiments/rfdetr_seg_v1/driver.py prepare-smoke
python experiments/rfdetr_seg_v1/driver.py simulate-policy
python experiments/rfdetr_seg_v1/driver.py status
```

Then build the deterministic bounded pools and generate—but do not execute—the
production command:

```bash
python experiments/rfdetr_seg_v1/driver.py build-pools
python experiments/rfdetr_seg_v1/driver.py prepare
python experiments/rfdetr_seg_v1/driver.py status
```

`census --hash-assets` additionally records every source-image SHA-256. Tile
generation always hashes each used source asset before accepting a cache hit.
Stage status is derived from validated artifacts; there is no separate durable
workflow-state database.

Legal negatives are deterministically sampled before JPEG encoding. The
configured train/validation fractions were selected from the full policy
simulation to retain enough examples for the 3:1 round-0 mixture without
materializing the roughly 1.16 million-negative universe. All positives and
all retained negatives are ordinary cached JPEGs during training; the sampling
step is not a lazy GPU data path.

For a writable local smoke area without editing production policy:

```bash
export SHITSPOTTER_RFDETR_ROOT=/tmp/shitspotter-rfdetr-smoke
python experiments/rfdetr_seg_v1/driver.py verify-inputs
python experiments/rfdetr_seg_v1/driver.py census
python experiments/rfdetr_seg_v1/driver.py prepare-smoke
python experiments/rfdetr_seg_v1/driver.py simulate-policy
```

Override the dataset checkout when it is mounted elsewhere with
`SHITSPOTTER_DATA_DPATH`. Override the shared cache independently with
`SHITSPOTTER_RFDETR_CACHE`.
