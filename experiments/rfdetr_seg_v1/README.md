# ShitSpotter RF-DETR Seg 2XL campaign

This directory contains only ShitSpotter policy and orchestration. Generic
tiling, cache, virtual-candidate selection, RF-DETR, prediction, and evaluation
code lives in the sibling `kwcoco_detector_kit` repository.

The data plane and control plane are deliberately separate:

- every safe negative window is represented in the persistent virtual candidate
  index under `candidates/`;
- positives are materialized once into the shared content-addressed tile cache;
- round 0 and fixed validation select deterministic source/scale-stratified
  negative quotas from the virtual indexes and materialize only those selected
  windows;
- later hard-negative mining scores the same virtual train universe and
  materializes only globally admitted hard negatives.

The workflow deliberately has two validation products:

- `train_validation_tiles.kwcoco.zip` is the fixed cached tile pool consumed
  cheaply at every RF-DETR validation epoch.
- canonical validation runs the selected checkpoint over the original 1,258
  validation images using KDK tiled native-mask inference, reconstructs masks
  in source coordinates, merges overlap duplicates, and then computes source
  image box/mask AP and zero-poop-image false-positive statistics.

The test split is frozen in `config.yaml` but must not be evaluated while
choosing data policy, checkpoints, thresholds, or mining rounds.

Run the local preflight first:

```bash
python experiments/rfdetr_seg_v1/driver.py verify-inputs
python experiments/rfdetr_seg_v1/driver.py census
python experiments/rfdetr_seg_v1/driver.py prepare-smoke
python experiments/rfdetr_seg_v1/driver.py simulate-policy
python experiments/rfdetr_seg_v1/driver.py build-candidates
python experiments/rfdetr_seg_v1/driver.py status
```

`build-candidates` is the complete safe-negative control plane and is reusable
when only the round mixture changes. `simulate-policy` reports the complete
positive/negative/ignore universe separately from the bounded materialization
plan.

Then build the pools. The positive pass uses only target-positive source images;
negative windows come from the already-built candidate indexes. The merge step
does not subsample a second time because selection has already happened:

```bash
python experiments/rfdetr_seg_v1/driver.py build-pools
python experiments/rfdetr_seg_v1/driver.py prepare
python experiments/rfdetr_seg_v1/driver.py status
```

After round 0 has produced its checkpoint, generate the four-GPU mining script.
The script scores deterministic source/scale-local shards concurrently, waits
for every rank, then runs the single global top-K finalizer and materializes
only the admitted hard negatives:

```bash
python experiments/rfdetr_seg_v1/driver.py prepare-mining --round-index=0
bash /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/rfdetr_seg_v1/rounds/round0/mining/RUN_MINING.sh
```

`census --hash-assets` additionally records every source-image SHA-256. Tile
generation always hashes each used source asset before accepting a cache hit.
Stage status is derived from validated artifacts; there is no separate durable
workflow-state database.

For a writable local smoke area without editing production policy:

```bash
export SHITSPOTTER_RFDETR_ROOT=/tmp/shitspotter-rfdetr-smoke
python experiments/rfdetr_seg_v1/driver.py verify-inputs
python experiments/rfdetr_seg_v1/driver.py census
python experiments/rfdetr_seg_v1/driver.py prepare-smoke
python experiments/rfdetr_seg_v1/driver.py simulate-policy
python experiments/rfdetr_seg_v1/driver.py build-candidates
```

Override the dataset checkout when it is mounted elsewhere with
`SHITSPOTTER_DATA_DPATH`. Override the shared cache independently with
`SHITSPOTTER_RFDETR_CACHE`.

## Round-0 v2 fine-tuning policy

The active RF-DETR run is `rfdetr.run_name: v2`. It reuses the exact same
materialized round-0 train/validation manifests as v1, but starts again from
the upstream pretrained RF-DETR Seg 2XLarge weights with a deliberately more
conservative optimization policy. V1 peaked early and then regressed while
using a constant 1e-4 model LR and 1.5e-4 encoder LR over a 60-epoch horizon.
V2 therefore uses 15 epochs, 5e-5 model LR, 1e-5 encoder LR, one epoch of
linear warmup, cosine decay to 5% of the base LR, and EMA mAP early stopping
with patience 4. The v1 workdir is not overwritten; v2 writes beneath
`rounds/round0/runs/v2/`.

After applying the matching KDK trainer overlay, regenerate the active run:

```bash
python experiments/rfdetr_seg_v1/driver.py prepare
cat "$SHITSPOTTER_RFDETR_ROOT/rounds/round0/runs/v2/ROUND0_COMMAND.txt"
```

Do not resume v2 from the regressing v1 checkpoint. The purpose of this run is
to test the optimization policy independently of the already-frozen 2:1 pool
composition.
