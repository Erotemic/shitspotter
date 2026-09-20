# RF-DETR Seg 2XL campaign journal

This is the durable handoff log for `experiments/rfdetr_seg_v1`. Keep it short,
current, and factual. It should capture decisions and observations that a new
agent would otherwise have to reconstruct from terminal logs or chat history.

## 2026-09-20 — Round-0 pool architecture and v1/v2/v3 training

### Frozen source manifests

The campaign is using the larger train/validation/test split manifests:

- train: `train_imgs11247_7f54a82d.kwcoco.zip`
- validation: `vali_imgs1258_9cdcc10a.kwcoco.zip`
- test: `test_imgs121_80c4a26b.kwcoco.zip`

The test split remains held out from policy/checkpoint/threshold selection.

### Negative control plane and materialized pools

The original pool design eagerly retained a random fraction of legal negatives.
That was replaced with a virtual control-plane architecture:

- every safe negative window is represented by the persistent candidate index;
- positive tiles are materialized once into the shared cache;
- exact deterministic source/scale-stratified negative quotas are selected from
  the virtual universe;
- only admitted negatives are materialized;
- round manifests reference the shared cache and do not recopy image data.

Existing candidate indexes are durable and should not be rebuilt merely because
round mixture or trainer hyperparameters change.

Observed candidate universes:

- train safe negatives: 1,161,819
- validation safe negatives: 129,672

Current round-0 / fixed-validation materialization policy is 2 negatives per
positive:

- train positive tiles: 62,417
- train negative tiles: 124,834
- round-0 train total: 187,251
- validation positive tiles: 7,455
- validation negative tiles: 14,910
- fixed tiled validation total: 22,365

`pool_manifest.json` schema 2 records this completed pool. New builds after the
status-performance cleanup write schema 3 receipts with cheap immutable-file
signatures. Full KWCoco validation belongs at `build-pools`; normal `status` and
`prepare` should not repeatedly deserialize and validate these unchanged files.
Use `status --details` or `prepare --details` for an explicit full audit.

### GDAL / static TLS import-order issue

On `aiq-gpu`, GDAL was installed successfully but could fail to import with:

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6:
cannot allocate memory in static TLS block
```

This is an import-order / ELF static-TLS issue, not a missing GDAL wheel. The
immediate reliable workaround for host-side pool materialization was:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 \
python experiments/rfdetr_seg_v1/driver.py build-pools
```

A delayed-image fix was prepared separately to pre-import optional GDAL early on
Linux and to distinguish “GDAL absent” from “GDAL installed but native import
failed.” Do not misdiagnose this specific failure as an ordinary missing-GDAL
installation.

### First production launch and GPU OOM

The first four-GPU launch reached DDP initialization but failed on the first
training forward because GPU 0 was already occupied by another workload. The
training rank itself was using only a small fraction of the 95 GiB device while
almost no free VRAM remained. Releasing that external GPU workload resolved the
OOM; batch size 4/GPU was not implicated by that failure.

PyTorch then emitted DDP/autograd performance warnings about AccumulateGrad
stream mismatch and a singleton-dimension gradient-stride mismatch. These were
performance warnings, not correctness failures. Investigate only if profiling
shows meaningful steady-state cost.

### V1 optimization behavior

V1 used approximately:

- 60 epochs
- model LR `1e-4`
- encoder/backbone LR `1.5e-4`
- no useful LR decay inside the 60-epoch horizon under the inherited default
  step schedule

Validation EMA metrics peaked early and then declined nearly monotonically:

| epoch | box mAP50:95 | box mAP75 | mask mAP50:95 |
| ---: | ---: | ---: | ---: |
| 2 | 0.6665 | 0.7543 | 0.6448 |
| 3 | 0.6447 | 0.7201 | 0.6236 |
| 4 | 0.6398 | 0.7124 | 0.6193 |
| 5 | 0.6339 | 0.7012 | 0.6144 |
| 6 | 0.6328 | 0.7059 | 0.6123 |

Precision remained high while recall declined somewhat, and mAP75 degraded more
than mAP50. The leading hypothesis is overly aggressive / overly long
fine-tuning of a strong pretrained representation. The 2:1 broad-negative pool
is a secondary hypothesis, but it is intentionally held fixed for v2 so the
optimization change is isolated.

### Active V2 policy

The active run is `rfdetr.run_name: v2`; it starts again from the original
pretrained RF-DETR Seg 2XLarge weights and writes under
`rounds/round0/runs/v2/`, preserving v1 checkpoints/logs.

V2 policy:

- epochs: 15
- model LR: `5e-5`
- encoder/backbone LR: `1e-5`
- scheduler: cosine
- cosine minimum factor: 0.05
- warmup: 1 epoch
- EMA mAP early stopping: enabled
- patience: 4
- min delta: 0.001
- batch size: 4/GPU on 4 GPUs
- pool composition: unchanged 2:1 negatives:positives

A transient low-utilization episode was investigated after one v2 restart. The
process was mostly idle during a sample taken at validation startup, and ranks
1-3 also showed small CUDA contexts on GPU 0. That initially suggested a
launcher/device-placement issue. However, after rebuilding the container image
and restarting the **unchanged v2 configuration**, good GPU utilization returned.
The suspected utilization regression therefore did not reproduce. Do not treat
the GPU-0 context observation or AccumulateGrad stream warning as a proven
throughput root cause without a fresh controlled profile.

### V3 batch-size experiment

V1/v2 training used batch 4/GPU on four GPUs (global batch 16) and consumed only
about 15 GiB of each ~96 GiB GPU. Even when utilization is healthy, that leaves
substantial memory headroom. V3 changes only the physical batch geometry:

- train batch size: 8/GPU on 4 GPUs (global batch 32)
- validation batch size: 8/GPU
- gradient accumulation: 1
- epochs: 15
- model LR: `5e-5`
- encoder/backbone LR: `1e-5`
- scheduler/warmup/early-stopping policy: unchanged from v2
- pool composition: unchanged 2:1 negatives:positives

The batch increase is deliberately modest. It doubles the global batch rather
than jumping directly to 64+, and no speculative KDK launcher, DDP, CUDA-device,
or auto-batch changes are part of this experiment. Learning rates are also left
unchanged. V3 has about half as many optimizer updates per epoch as v2, so the
two recipes are not update-count-equivalent.

If v3 still develops the same high-precision / falling-recall pattern under the
conservative schedule, the next controlled experiment should change pool
composition (for example 1:1 negatives) without rematerializing the negative
cache.

### Operational performance observations

Large KWCoco manifest construction/serialization and repeated validation feel
slower than expected. A separate recon should measure object construction,
validation, JSON serialization, ZIP compression, filesystem writes, and reload
costs before choosing an optimization. Do not assume JSON encoding is the only
bottleneck.

The immediate orchestration fix is to avoid redundant validations of immutable
pool artifacts. Before this cleanup, `status` could validate the same positive
pool twice and `prepare` would validate the large train/validation manifests yet
again. Normal status/prepare now use the pool receipt; `--details` opts into the
expensive checks.

### V3 live run observations and validation-phase diagnosis

A fresh rebuilt RF-DETR image restored healthy utilization with the unchanged v2
recipe, so the earlier low-utilization episode is treated as transient/environmental.
The active experiment was then moved to v3 with only the batch change documented
above.

Observed v3 training behavior on `aiq-gpu` (4 x ~95 GiB GPUs):

- batch 4/GPU had previously used roughly 15 GiB/GPU and sustained about 70% GPU
  utilization when training was healthy;
- v3 batch 8/GPU uses roughly 26 GiB/GPU and sustained about 80% utilization in
  the training loop;
- this leaves substantial memory headroom. A future controlled run can reasonably
  test batch 16/GPU (global batch 64), but **do not mutate the currently running v3
  recipe merely for throughput**. Preserve its result first.

The apparent recurring "mid-run slowdown" was identified rather than guessed. On
v3, `metrics.csv` stopped with sparse train logging at epoch 0 / step 5849, while
the pool contains 187,251 train tiles. At global batch 32 the expected epoch size
is approximately `187251 / 32 = 5851.6` optimizer steps, so the slowdown occurred
exactly at the epoch boundary. The Docker log then emitted at 2026-09-20
16:44:36 UTC:

```
Skipping validation-loss computation: compute_val_loss='auto' found no scheduler
or callback monitoring 'val/loss' ...
```

That message is emitted at RF-DETR validation-epoch start. Therefore the observed
GPU-utilization collapse is validation / metric-finalization work, **not test
loss, not hidden training loss, and not evidence that the training dataloader
suddenly degraded**. `val/loss` is explicitly skipped. The fixed validation pool
contains 22,365 tiles; at 8/GPU x 4 GPUs this is about 699 distributed validation
batches, followed by distributed prediction/target merging, bbox COCO metrics,
segmentation COCO metrics, F1/precision/recall work, early-stopping/best-model
callbacks, and checkpoint writes. Those CPU/synchronization-heavy phases can make
`nvtop` look nearly idle before the next training epoch resumes.

Use the new cheap status helper during future runs:

```bash
python experiments/rfdetr_seg_v1/run_status.py
python experiments/rfdetr_seg_v1/run_status.py --watch --interval=10
```

It estimates train steps and validation batches from the pool receipt and active
batch geometry, reads the small `metrics.csv`, and, when exactly one matching
container is running, scans recent Docker phase markers. This should be the first
thing a fresh agent uses when utilization changes.

Potential future throughput experiments, in order, after v3 quality is known:

1. increase **validation** batch independently (16/GPU, then possibly 32/GPU),
   because validation has no backward-activation storage;
2. consider `eval_interval: 2` if per-epoch validation is a material wall-clock
   cost, while accounting for early-stopping semantics;
3. test train batch 16/GPU (global batch 64) with the same optimizer policy before
   considering anything larger;
4. profile the existing AccumulateGrad stream / gradient-stride warnings only if
   training-phase utilization remains unexpectedly poor. They remain warnings,
   not a proven root cause.

### Hard-negative truth-QA gate

Hard-negative mining is now explicitly treated as both a training-data mechanism
and an annotation-audit mechanism. A model prediction on a supposedly safe
negative can be a genuine model false positive **or a false negative in the truth
annotations**. The hardest mined negatives should therefore be manually reviewed
before they are automatically trusted as round-(N+1) background.

KDK now owns the generic `mine-review` workflow. It joins completed mining ledgers
back to the virtual candidate index and canonical source KWCoco, maps each mined
prediction from tile coordinates into source-image coordinates, and writes:

- `review_queue.json` / `review_queue.tsv` with rank, score, tile identity,
  source gid/path, source-coordinate prediction box, current truth counts, and
  blank human review fields;
- `review.kwcoco.zip`, a **diagnostic** source-image subset preserving current
  truth plus `__hard_negative_review__` proposal annotations;
- `previews/*.jpg` and `index.html`, hardest first, with red prediction boxes,
  blue tile extents, and green existing target-truth boxes;
- adjacent `.json` sidecar discovery when one exists beside the source image.

ShitSpotter wraps that generic primitive with:

```bash
python experiments/rfdetr_seg_v1/review_hard_negatives.py --round-index=0
```

which additionally writes `annotation_targets.tsv`. ShitSpotter's canonical manual
truth is the LabelMe JSON sidecar next to the source image, so this table gives the
exact image and JSON path to edit/create. The review KWCoco is never the source of
truth.

The default review policy is top 200 globally selected hard negatives, at most 3
per source image, score >= 0.30. These are UX defaults, not mining-selection
semantics, and can be overridden at review time without rescoring.

**Truth correction invalidation rule:** if review finds an unannotated poop and
canonical truth changes, regenerate the source KWCoco and rebuild truth-dependent
artifacts before round 1. In particular, do not reuse the old train negative
candidate index: a window classified as safe under old truth may now contain a
positive. Rebuild the candidate index and derived positive/negative pools (and any
round manifest built from them) against the corrected manifest fingerprint.

### Current aiq-gpu path / provenance anchors

The live campaign uses environment overrides rather than the historical defaults
written in `config.yaml`:

- `SHITSPOTTER_DATA_DPATH=/data/users/jon.crall/shitspotter_dvc`
- `SHITSPOTTER_RFDETR_ROOT=/data/users/jon.crall/shitspotter_rfdetr_v1`
- active v3 workdir:
  `/data/users/jon.crall/shitspotter_rfdetr_v1/rounds/round0/runs/v3`
- train candidate index:
  `/data/users/jon.crall/shitspotter_rfdetr_v1/candidates/train_negative_candidates`
- validation candidate index:
  `/data/users/jon.crall/shitspotter_rfdetr_v1/candidates/validation_negative_candidates`
- shared tile cache:
  `/data/users/jon.crall/shitspotter_rfdetr_v1/tile_cache`
- RF-DETR image: `kwcoco-detector-kit:rfdetr-cu132-aiq`

Frozen source-manifest SHA-256 values recorded for this campaign are:

- train: `b3b6246531e525493e653917609cf0194d5c5eca2881314aa2c53c88160650a4`
- validation: `72de53533abf3fa9d9db19f24b4d02dbd56a19cadf0a86107dda0d28aa4d00a5`
- test: `403cbf79cf91711c886f30b13f66a3076118fa3a9571d87a6d9f3509b7ee68ed`

These hashes are the provenance anchors for the currently materialized pools. If
manual hard-negative review changes truth, the regenerated source manifest hash is
expected to change; that is an intentional signal to invalidate old truth-derived
artifacts, not something to work around.

### Fresh-agent handoff / next gates

Current state at the time of this journal update:

1. The v3 round-0 RF-DETR Seg 2XLarge run is intentionally left running.
   Recipe: batch 8/GPU x4, global 32, 15 epochs, `5e-5` model LR, `1e-5`
   encoder LR, cosine + one-epoch warmup, EMA-mAP early stopping, unchanged 2:1
   negative:positive pools.
2. Training-phase throughput is healthy (~80% average GPU utilization, ~26 GiB
   VRAM/GPU). Periodic low utilization at epoch boundaries is currently explained
   by validation/metric work.
3. Do not touch the held-out test split while choosing checkpoint, pool policy,
   thresholds, or mining rounds.
4. When v3 produces useful validation metrics, compare its early curve to v1's
   epoch-2 peak rather than blindly waiting for all 15 epochs; early stopping is
   enabled.
5. After choosing the round-0 checkpoint, run `driver.py prepare-mining`, execute
   `RUN_MINING.sh`, then run `review_hard_negatives.py` **before** admitting mined
   negatives to round 1.
6. Manually inspect the hardest review items. If they expose truth omissions,
   correct LabelMe/source truth and rebuild stale truth-dependent artifacts before
   continuing. If they are genuine false positives, the selected hard negatives
   are appropriate training background.
7. Only after v3 quality is understood should throughput policy move to batch
   16/GPU/global 64 or less-frequent/larger-batch validation. Keep those changes
   as separate interpretable experiments.

The important architecture remains: complete safe-negative universe virtual;
materialize positives/fixed validation/admitted negatives into the shared cache;
mining ledgers are durable/resumable score evidence; finalization is a cheap
re-tunable global selection step; manual truth review is now a gate between mining
and treating the selected examples as trusted negative supervision.

### 2026-09-20 — source-space annotation-QA and truth-semantics handoff

A correctness audit found that the round-0 tiler and virtual candidate builder
were filtering source annotations to `category_names: [poop]` before deciding
whether a window was legal background. That was correct for deciding which
categories become positive detector classes, but incorrect for uncertainty:
source annotations named `unknown` or `ignore` became invisible and their pixels
could be learned as ordinary background. A positive poop tile could likewise
contain an uncertain region that was implicitly background.

The repository history supports a three-way semantic split. The WACV FAQ says
`unknown` / `ignore` are used when the annotator cannot confidently decide
whether a region is poop and explicitly says those regions are not forced into
positive or background. The older annotation journal says false-positive
structures were deliberately retained under names such as leaf, stick, grass,
and shadow so they can serve as explicit hard-negative evidence. Therefore the
campaign remains binary: `poop` is the only positive model class; named nuisance
classes are background/distractor evidence; uncertainty blocks supervision.

KDK now owns the generic `TruthSemantics` abstraction, and both eager tiling and
the virtual negative universe apply it. ShitSpotter configures:

```yaml
truth_semantics:
  target_categories: [poop]
  ignore_categories: [ignore, unknown, unkown]
  uncategorized_annotation_policy: ignore
  default_non_target_policy: background
  unclassified_category_policy: ignore
```

`unkown` is blocked conservatively because the current train census contains one
such annotation, but the code snapshot does not contain the corresponding
canonical source asset/LabelMe record needed to prove it is a typo. Likewise,
`residue`, `residual`, and the 32 uncategorized annotations are not assigned new
semantics from their spelling. `audit_truth.py` now writes exact source/LabelMe
paths for those cases when run against the DVC data so they can be inspected.

The existing candidate manifests are now intentionally stale for round 1. Their
policy fingerprints predate truth semantics. This does **not** require touching
the currently running v3 job: its already-materialized round-0 pools remain its
frozen experimental input. The corrected candidate indexes and pools should be
regenerated only after the source-space review and any truth changes, before
round-1 admission.

A second correctness problem was found in model packaging. Generic package code
renamed the selected checkpoint to `weights/checkpoint.pth`; when unpacked it
became `checkpoint.pth`, but `RFDETRTrainer.find_checkpoint()` recognizes
`checkpoint_best_total.pth`, `checkpoint_best_ema.pth`, or `last.ckpt`. Thus an
apparently self-contained RF-DETR package could fail to reconstruct a usable
workdir. Package building now delegates selection to the trainer and preserves
the canonical checkpoint basename and SHA-256. It also includes RF-DETR's JSON
generated config, exact label order, inference/capability metadata, runtime
framework versions, training-manifest identities when supplied, and ONNX parity
metadata.

The generic tiled inference implementation that already reconstructed boxes and
native masks into source coordinates has been promoted into KDK's predictor
layer and reused by ordinary `kwcoco-detector-kit predict`. A no-cache source
window reader chooses decode-once for JPEG-like assets and delayed regional reads
for TIFF-like assets, with a correctness fallback to decode-once if crop
realization is unsupported. No persistent tile corpus is prepared for this
annotation-QA pass.

The vendored RF-DETR 1.10.1 source includes a segmentation ONNX export contract
with raw `dets`, `labels`, and `masks`. KDK now has a matching ONNX predictor and
postprocessed parity check. This establishes that native-mask export is supported
by the bundled upstream code; it does **not** establish performance on the RTX
3090. ONNX preference is gated on successful real-window parity, and the local
acceptance run must still benchmark complete source-space throughput and choose a
24-GB-safe batch size. Start with batch 16, then compare 8/16/24/32 as memory
allows rather than copying the 96-GB Blackwell training batch assumptions.

The active v3 fixed tiled validation was materialized before this uncertainty
policy existed, so `unknown` / `ignore` pixels in those frozen tiles were not
represented as ignore regions. Because the exported detector truth contains only
`poop`, a prediction landing only on one of those uncertain regions can be
counted as a false positive by the current validation metrics. The size of
that effect has not been measured. Do not mutate that running experiment to repair
its validation data mid-run. Corrected validation pools for future rounds will
omit windows intersecting uncertainty under the new semantics. Canonical
source-space annotation review also explicitly classifies such detections as
`uncertain_region` rather than `false_positive`.

The intended next sequence is now:

1. leave v3 running and snapshot the current `checkpoint_best_ema.pth` immutably (the pinned RF-DETR only promotes `checkpoint_best_total.pth` on fit end);
2. rsync the snapshot to `toothbrush` and verify hashes;
3. build a self-describing package locally, exporting ONNX and measuring real-window parity (failed parity leaves PyTorch preferred);
4. predict the original training KWCoco in source coordinates with no tile cache;
5. inspect high-confidence unexplained/distractor/uncertain predictions and edit only canonical LabelMe truth by hand;
6. regenerate canonical KWCoco and audit its hashes/census;
7. rebuild candidate indexes/pools under the corrected truth semantics;
8. only then score/review/materialize legal hard negatives for round 1.

The test split remains excluded from every decision in that loop.

### 2026-09-20 — v3 new best at displayed validation 3/15; local Docker review workflow

V3 continued to improve without changing the running recipe. The previous best
validation was displayed as `Val (ema) (Epoch 2/15)`:

| metric | value |
| --- | ---: |
| box mAP50:95 | 0.6908 |
| box mAP50 | 0.8780 |
| box mAP75 | 0.7880 |
| mAR@500 | 0.8865 |
| F1 | 0.8217 |
| precision | 0.8584 |
| recall | 0.7880 |
| segm mAP50:95 | 0.6627 |
| segm mAP50 | 0.8813 |

RF-DETR then printed `Best EMA metric improved to 0.6627 (epoch 1)`. The next
validation, displayed as `Val (ema) (Epoch 3/15)`, improved again:

| metric | value |
| --- | ---: |
| box mAP50:95 | 0.6975 |
| box mAP50 | 0.8853 |
| box mAP75 | 0.7983 |
| mAR@500 | 0.8952 |
| F1 | 0.8309 |
| precision | 0.8602 |
| recall | 0.8035 |
| segm mAP50:95 | **0.6705** |
| segm mAP50 | 0.8897 |

The effective early-stopping metric improved by about 0.008 and RF-DETR logged
`Best EMA metric improved to 0.6705 (epoch 2)`. The table heading is one-based
(`Epoch 3/15`) while the callback message reports the zero-based internal epoch
index (`epoch 2`). Future snapshot names should therefore encode the displayed
validation number and/or metric instead of using ambiguous names such as
`epoch2_best`. The recommended immutable name for this checkpoint is
`v3_best_ema_val3_map6705_20260920`.

This new best is above the recorded v1 peak: v1's best known
segmentation mAP50:95 was 0.6448, versus 0.6705 here, and box mAP50:95 improved
from 0.6665 to 0.6975. No optimizer, batch, data, or validation-policy change was
made between these v3 validations. Leave the active v3 run unchanged and allow
early stopping / the 15-epoch ceiling to decide when it finishes.

The existing local snapshot named `v3_best_ema_20260920` should not be assumed
to be either the 0.6627 or 0.6705 checkpoint from its name alone; the RF-DETR
log timestamps and local shell timestamps may use different time zones. Inspect
its snapshotted `metrics.csv` / checksum provenance before deciding what it
contains. Do not overwrite it. When taking another snapshot, first confirm the
current live best and use a new immutable name that identifies the displayed
validation / metric.

The local annotation-QA execution path was also hardened. RF-DETR should no
longer be installed ad hoc into the host Python environment merely to package or
predict a checkpoint. KDK now provides:

```text
docker/rfdetr/build_auto.sh
docker/rfdetr/build_stable_cuda130.sh
docker/rfdetr/kcd-rfdetr
```

`build_auto.sh` selects by GPU architecture as well as driver capability. On the
RTX 3090 (compute capability 8.6) it intentionally chooses stable PyTorch cu130
even though the installed driver reports CUDA 13.2. On a Blackwell-class host
(compute capability >= 12.0) with a CUDA-13.2-capable driver it selects the
existing cu132 nightly profile. RF-DETR itself has no custom CUDA extension in
this image, so the old `blackwell` builder name represented a deployment profile,
not a Blackwell-only model implementation.

`kcd-rfdetr` is now the normal container execution surface. It defaults to
physical GPU 0, mounts `$HOME` at the same path in the container, bind-mounts the
current KDK checkout over the baked editable-install path, runs with the host
uid/gid, and safely converts command argv into the historical RF-DETR image's
`bash -lc` entrypoint. This preserves the existing aiq multi-GPU launch behavior
while eliminating hand-written local `docker run` functions.

ShitSpotter now wraps that generic runtime with
`experiments/rfdetr_seg_v1/local_review.py`. For an immutable snapshot, the
normal local sequence is:

```bash
SNAPSHOT=v3_best_ema_val3_map6705_20260920
python experiments/rfdetr_seg_v1/local_review.py all --snapshot-name="$SNAPSHOT"
```

The command verifies the snapshot checksum manifest and frozen train SHA,
packages the model, runs a four-image smoke prediction first, predicts the full
original train KWCoco in source coordinates with no persistent tile cache, and
builds the truth-aware review. PyTorch is the default backend for this pass.
ONNX remains an explicit experiment until a pinned GPU ONNX Runtime container
profile has measured native-mask parity and complete-loop throughput on the
3090.
