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
