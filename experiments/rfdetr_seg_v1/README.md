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

## Round-0 v4 large-batch / low-LR policy

The active RF-DETR recipe is now `rfdetr.run_name: v4`. It intentionally reuses
the exact same materialized round-0 train/validation manifests as v3. No
hard-negative mining result or source-truth update is included in this run.

V4 is a narrow optimizer experiment against the completed v3 baseline:

- train batch: 16/GPU on four GPUs, global batch 64
- validation batch: 8/GPU (unchanged from v3)
- gradient accumulation: 1
- epoch ceiling: 10
- model LR: `2.5e-5` (half of v3)
- encoder/backbone LR: `5e-6` (half of v3)
- one epoch linear warmup (unchanged)
- cosine decay to 5% of the base LR (unchanged)
- EMA mAP early stopping, patience 4 (unchanged)

The experiment hypothesis is that the larger physical batch reduces gradient
noise while the lower learning rates slow parameter movement enough to improve
the early validation optimum and/or reduce the post-optimum AP decline seen in
v3. This is deliberately *not* linear LR scaling with batch size.

The v3 acceptance baseline is fixed:

```text
v3 best displayed validation: 3/15
box mAP50:95:                0.6975
segm mAP50:95:               0.6705
canonical artifact:          checkpoint_best_total.pth
```

The primary v4 success criterion is EMA segmentation mAP50:95 greater than
`0.6705`; box mAP50:95 greater than `0.6975` is a useful secondary result. Keep
checkpoint selection on segmentation AP even if threshold-specific F1 continues
to improve later than AP.

The ten-epoch ceiling is intentionally shorter than v3's old 15-epoch ceiling.
V3 peaked at displayed epoch 3 and stopped after displayed epoch 7 with patience
4, so ten epochs leaves room for a lower-LR optimum to shift later without
spending another unnecessary 15-epoch run. Early stopping remains the real stop
condition.

V1/v2/v3 workdirs are not overwritten; v4 writes beneath
`rounds/round0/runs/v4/`. Regenerate the active run with:

```bash
python experiments/rfdetr_seg_v1/driver.py prepare
cat "$SHITSPOTTER_RFDETR_ROOT/rounds/round0/runs/v4/ROUND0_COMMAND.txt"
```

Start v4 fresh from the same upstream pretrained RF-DETR Seg 2XLarge weights as
the prior runs. Do not resume from the v3 checkpoint; that would answer a
different question.

See `JOURNAL.md` for the completed v3 trajectory and the rationale for this v4
control.

## Binary truth semantics and round-1 invalidation

The RF-DETR campaign is a **binary poop detector** even though the source
KWCoco currently contains 68 declared categories. `config.yaml` now makes the
supervision contract explicit:

- `poop` is the only positive detector target;
- confidently named non-target annotations such as leaf, rock, pinecone, stick,
  and bark remain background/distractor evidence rather than becoming detector
  classes;
- `unknown`, `ignore`, and the observed typo-like `unkown` are treated as
  uncertain regions and block training-negative windows;
- annotations with no usable category identity fail closed as uncertain until
  they are audited.

This is not the same as KDK's historical `tile_role="ignore"`, which means a
window intersected target geometry but could not safely retain it. Source
uncertainty is now represented by `truth_semantics` and such windows are omitted
entirely when the trainer has no region-ignore mechanism.

The already-materialized v3 validation pool predates this policy. In that
frozen pool, uncertainty annotations were dropped from detector truth, so a
prediction landing only on an `unknown`/`ignore` region can be counted as a
false positive. The magnitude of that effect is not yet measured; do not
mutate the running v3 validation set to correct it mid-experiment.

The candidate universe is truth-policy-dependent. Candidate indexes created
before this change are intentionally rejected by `driver.py` because their
manifest policy does not contain the current truth semantics. **Do not rebuild
or disturb the active v3 run.** Rebuild train/validation candidate indexes and
derived pools only after the local truth-review pass and any manual annotation
corrections, before constructing round 1.

The reference census captured at this handoff is
`dataset_category_census.json`: 11,247 train images, 8,960 annotations, 8,176
`poop` annotations, 784 non-poop annotations, and 68 declared categories. Run
this against the current canonical source after annotation edits to regenerate
an exact census and enumerate semantic edge cases:

```bash
python experiments/rfdetr_seg_v1/audit_truth.py \
    --src "$SHITSPOTTER_DATA_DPATH/train.kwcoco.zip" \
    --dst /tmp/shitspotter_truth_census.json
```

The source documentation explicitly defines `unknown` / `ignore` as uncertain
poop-vs-background regions and describes named clutter labels as sparse false-
positive annotations. `residue`, `residual`, uncategorized annotations, and the
single observed `unkown` annotation still require source/LabelMe inspection;
the campaign does not silently infer their meaning. `unkown` is conservatively
blocked for now.

## Stable model snapshot -> local package -> source-space review

Keep packaging and annotation-QA work off the active four-GPU training job. In
the pinned RF-DETR source, `checkpoint_best_ema.pth` is refreshed whenever EMA
validation improves, while `checkpoint_best_total.pth` is promoted by
`BestModelCallback.on_fit_end`. Therefore a still-running v3 run should snapshot
`checkpoint_best_ema.pth` immutably before transfer.

The validation table is one-based while the RF-DETR callback log reports the
zero-based internal epoch index. For example, the current best printed under
`Val (ema) (Epoch 3/15)` and was logged as `Best EMA metric improved ... (epoch
2)`. Snapshot names should avoid ambiguous bare `epoch2` terminology. The
current recommended name is:

```text
v3_best_ema_val3_map6705_20260920
```

On `aiq-gpu`, create the immutable snapshot:

```bash
cd ~/code/shitspotter
export SHITSPOTTER_RFDETR_ROOT=/data/users/jon.crall/shitspotter_rfdetr_v1
python experiments/rfdetr_seg_v1/run_status.py
python experiments/rfdetr_seg_v1/snapshot_model.py \
    --name=v3_best_ema_val3_map6705_20260920 \
    --checkpoint=checkpoint_best_ema.pth
```

The helper hashes the live checkpoint before and after copying, verifies the
copy, snapshots the small training metadata, and atomically publishes only a
stable result.

V3 has now completed by early stopping. Its displayed validation epoch 3 was
the final best (`segm mAP50:95=0.6705`, `box mAP50:95=0.6975`), followed by four
non-improving validations through displayed epoch 7. RF-DETR then promoted the
best EMA state into `checkpoint_best_total.pth` and explicitly logged `Best
total checkpoint saved from EMA`. For any new frozen v3 package,
`checkpoint_best_total.pth` is therefore the canonical stable artifact; the
mutable-EMA snapshot instructions above only describe the earlier while-running
state. The snapshot helper now defaults to `--checkpoint=auto`, which selects
`checkpoint_best_total.pth` when present and otherwise falls back to the live
EMA best. A completed-run snapshot can therefore simply be created with:

```bash
python experiments/rfdetr_seg_v1/snapshot_model.py \
    --name=v3_best_total_val3_map6705_20260920
```

On `toothbrush`, one snapshot name is enough to derive both transfer paths:

```bash
cd ~/code/shitspotter
python experiments/rfdetr_seg_v1/sync_model.py \
    --snapshot-name=v3_best_ema_val3_map6705_20260920
```

Local RF-DETR package/prediction work runs in KDK's RF-DETR Docker image rather
than requiring `rfdetr` in the host Python environment. KDK's generic builder
selects stable PyTorch `cu130` for the RTX 3090 (compute capability 8.6) even
when the host driver advertises CUDA 13.2; the historical `cu132` profile remains
available for Blackwell production hosts. Physical host GPU 0 is the default for
local review.

Inspect the resolved paths and GPU runtime:

```bash
SNAPSHOT=v3_best_ema_val3_map6705_20260920
python experiments/rfdetr_seg_v1/local_review.py status \
    --snapshot-name="$SNAPSHOT"
python experiments/rfdetr_seg_v1/local_review.py build-image
~/code/kwcoco_detector_kit/docker/rfdetr/kcd-rfdetr image-info
```

Then the normal local annotation-QA workflow is a single command:

```bash
python experiments/rfdetr_seg_v1/local_review.py all \
    --snapshot-name="$SNAPSHOT"
```

`all` performs, in order:

1. verify the immutable snapshot checksum manifest and the frozen train KWCoco
   SHA-256;
2. build a self-describing RF-DETR model package in
   `~/data/shitspotter_models/`;
3. create and predict a two-positive/two-negative smoke subset first;
4. run no-cache 768-window source-space prediction over the original train
   KWCoco on physical GPU 0, using `--windowed=true` and the PyTorch backend;
5. build the truth-aware review under
   `~/data/shitspotter_review/<snapshot-name>/review`.

The current defaults are batch 16, overlap 0.25, prediction score floor 0.01,
and review score floor 0.50. Prediction is also pipelined by default: two source
workers decode future images, two future window batches are realized ahead, the
main thread exclusively owns CUDA inference, and one CPU worker merges/NMSes and
polygonizes the previous source with at most two source results in flight. The
queues are bounded and review output remains deterministic.

The default pipeline knobs are:

```text
pipeline=true
source_workers=2
source_prefetch=2
window_prefetch=2
postprocess_workers=1
postprocess_inflight=2
```

Override them explicitly when profiling rather than editing campaign code:

```bash
python experiments/rfdetr_seg_v1/local_review.py predict \
    --snapshot-name="$SNAPSHOT" \
    --batch-size=24 \
    --source-workers=4 \
    --source-prefetch=3 \
    --window-prefetch=3 \
    --postprocess-workers=2 \
    --postprocess-inflight=3
```

Use `--pipeline=false` for a serial comparison run. The generated prediction
profile reports source/window work and wait time separately from GPU inference
and CPU postprocessing, so tuning should target the stage that is actually
starving or backpressuring the GPU.

Individual stages remain available for diagnosis or resumability:

```bash
python experiments/rfdetr_seg_v1/local_review.py package --snapshot-name="$SNAPSHOT"
python experiments/rfdetr_seg_v1/local_review.py smoke   --snapshot-name="$SNAPSHOT"
python experiments/rfdetr_seg_v1/local_review.py predict --snapshot-name="$SNAPSHOT"
python experiments/rfdetr_seg_v1/local_review.py review  --snapshot-name="$SNAPSHOT"
```

The review queue classifies each source-coordinate prediction as
`matched_target`, `known_distractor`, `uncertain_region`, or
`unexplained_prediction`. The last category is the primary missing-truth review
queue; `known_distractor` exposes systematic confusion with named clutter, and
`uncertain_region` must not be treated as an ordinary false positive.

The local workflow defaults to the proven PyTorch RF-DETR runtime. ONNX remains
a separate KDK package/export experiment until a pinned GPU ONNX Runtime
container profile is available; it should not replace the PyTorch prediction
backend until complete-loop parity and throughput are measured.

After manual truth corrections: regenerate canonical KWCoco, rerun
`audit_truth.py`, then rebuild the truth-dependent candidate indexes/pools before
hard-negative scoring and round 1. The held-out test split remains out of
checkpoint selection, threshold tuning, mining policy, and annotation-QA
selection.

## Coarse prediction space for annotation QA

Full native-resolution review is too expensive as the default discovery pass.
KDK now treats detector prediction resolution as a first-class coordinate space:
I/O and tiling may run against a scaled delayed-image view, while every emitted
box/polygon is transformed back into the source KWCoco's native image space.
This means review tooling and LabelMe sidecars continue to use canonical source
coordinates regardless of detector resolution.

The campaign's coarse discovery command is:

```bash
python experiments/rfdetr_seg_v1/local_review.py coarse \
    --snapshot-name=v3_best_ema_20260920
```

Its current defaults are:

```text
prediction_scale = 0.40
overlap          = 0.10
window           = 768
batch_size       = 16
```

The coarse outputs are kept separate from native-resolution products:

```text
train_predictions.scale0p4.kwcoco.zip
review.scale0p4/
smoke4.scale0p4.pred.kwcoco.zip
```

For TIFF/COG-like sources the scale is part of the delayed-image graph before
window crops are finalized, so GDAL-backed delayed-image optimization can use
source overviews where available. JPEG-like images still follow the decode-once
path and are resized once before their windows are sliced in memory.

The coarse pass is a discovery stage, not a change to canonical truth. Future
native-resolution refinement should consume interesting coarse proposals and
re-run only those regions at `prediction_scale=1.0`. A sampled native-resolution
audit of coarse-negative images should be used to estimate what the coarse pass
misses before it becomes a trusted hard-negative admission screen.

Long coarse scans are resumable. KDK checkpoints committed source-image
predictions every 250 images or five minutes by default and forces a checkpoint
on exceptions / Ctrl-C. Re-running the same `local_review.py predict` or
`coarse` command resumes matching partial output automatically; changed
prediction settings fail closed instead of mixing coordinate spaces or model
outputs.
