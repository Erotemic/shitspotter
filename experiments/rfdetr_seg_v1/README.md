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

Run the metadata census, build deterministic cached pools, and generate—but do
not execute—the production command:

```bash
python experiments/rfdetr_seg_v1/driver.py census
python experiments/rfdetr_seg_v1/driver.py build-pools
python experiments/rfdetr_seg_v1/driver.py prepare
python experiments/rfdetr_seg_v1/driver.py status
```

`census --hash-assets` additionally records every source-image SHA-256. Tile
generation always hashes each used source asset before accepting a cache hit.
Stage status is derived from validated artifacts; there is no separate durable
workflow-state database.

For a writable local smoke area without editing production policy:

```bash
export SHITSPOTTER_RFDETR_ROOT=/tmp/shitspotter-rfdetr-smoke
python experiments/rfdetr_seg_v1/driver.py census
```
