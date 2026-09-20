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

`build-pools` is the expensive validation boundary. It fully validates the
generated KWCoco artifacts before recording a durable pool receipt. Normal
`status` and `prepare` calls trust that immutable receipt instead of repeatedly
deserializing large manifests and checking every referenced tile. Use
`--details` when an explicit full audit is desired:

```bash
python experiments/rfdetr_seg_v1/driver.py status --details
python experiments/rfdetr_seg_v1/driver.py prepare --details
```

Schema-2 pool manifests produced before the receipt optimization are accepted
as already-validated stage-boundary artifacts. New pool builds write schema 3
with cheap file size/mtime signatures so accidental mutation can be detected
without reopening the KWCoco payload.

After round 0 has produced its checkpoint, generate the four-GPU mining script.
The script scores deterministic source/scale-local shards concurrently, waits
for every rank, then runs the single global top-K finalizer and materializes
only the admitted hard negatives:

```bash
python experiments/rfdetr_seg_v1/driver.py prepare-mining --round-index=0
bash /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/rfdetr_seg_v1/rounds/round0/mining/RUN_MINING.sh
```

Hard-negative mining is followed by a **truth-review gate** before mined tiles are
trusted as negatives for the next round. The detector can be correct while the
truth is wrong: a very hard "negative" may actually be an unannotated poop. Build
the ranked review queue after `RUN_MINING.sh` completes:

```bash
python experiments/rfdetr_seg_v1/review_hard_negatives.py --round-index=0
```

This writes `rounds/round0/mining/review/` with:

- `index.html`: hardest-first static previews; red is the mined prediction, blue
  is the mined tile, green is existing poop truth;
- `review_queue.tsv`: editable review status / notes plus source image, source gid,
  score, tile identity, and prediction box in source coordinates;
- `annotation_targets.tsv`: the canonical source image and adjacent LabelMe JSON
  sidecar path to edit/create when the mined "negative" is actually positive;
- `review.kwcoco.zip`: diagnostic source-image subset containing existing truth
  plus a dedicated `__hard_negative_review__` proposal category. **Do not edit
  this review KWCoco as truth.**

If manual review changes source annotations, regenerate the source KWCoco and
rebuild every artifact whose truth fingerprint is now stale, especially the
train candidate index and derived positive/negative pools. A candidate that was
legal under old truth must not silently survive as a training negative.

While a run is active, use the cheap phase/status helper rather than inferring
training state from GPU utilization alone:

```bash
python experiments/rfdetr_seg_v1/run_status.py
python experiments/rfdetr_seg_v1/run_status.py --watch --interval=10
```

It reads only the small pool receipt, `metrics.csv`, and (when available) the
matching Docker log. It reports approximate train steps/epoch, validation batch
count, the last completed validation metrics, and a phase guess such as
`training` or `validation_or_metric_finalize`.

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

## Round-0 v3 fine-tuning policy

The active RF-DETR recipe is `rfdetr.run_name: v3`. It reuses the exact same
materialized round-0 train/validation manifests and the conservative optimizer
policy introduced for v2. The only intended training change is a modest
physical-batch increase from 4 to 8 images per GPU on four GPUs:

- train batch: 8/GPU, global batch 32
- validation batch: 8/GPU
- gradient accumulation: 1
- epochs: 15
- model LR: `5e-5`
- encoder LR: `1e-5`
- one epoch linear warmup
- cosine decay to 5% of the base LR
- EMA mAP early stopping, patience 4

The learning rates are deliberately unchanged. V3 is still a conservative
fine-tune of the pretrained RF-DETR Seg 2XLarge model, not a large-batch
re-tuning experiment. The global batch doubles from 16 to 32, so v3 performs
about half as many optimizer updates per epoch as v2; compare quality by epoch
and wall time with that difference in mind.

A prior low-utilization observation did **not** reproduce after rebuilding the
container image and restarting the unchanged v2 configuration. The rebuilt v2
run again showed good GPU utilization. Therefore this campaign does not carry
forward any speculative launcher/DDP/device-binding changes from that
investigation. The batch increase is intentionally the only runtime-policy
change in v3.

The v1/v2 workdirs are not overwritten; v3 writes beneath
`rounds/round0/runs/v3/`. Regenerate the active run with:

```bash
python experiments/rfdetr_seg_v1/driver.py prepare
cat "$SHITSPOTTER_RFDETR_ROOT/rounds/round0/runs/v3/ROUND0_COMMAND.txt"
```

Do not resume v3 from v1/v2. Start from the same upstream pretrained RF-DETR
Seg 2XLarge weights so this remains an interpretable batch-size experiment.

See `JOURNAL.md` for the campaign handoff log and rationale behind the current
data architecture and current training policy.
