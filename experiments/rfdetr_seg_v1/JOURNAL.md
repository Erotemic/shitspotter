# RF-DETR Seg 2XL campaign journal

This is the durable handoff log for `experiments/rfdetr_seg_v1`. Keep it short,
current, and factual. It should capture decisions and observations that a new
agent would otherwise have to reconstruct from terminal logs or chat history.

## 2026-09-20 — Round-0 pool architecture and v1/v2 training

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

If v2 still develops the same high-precision / falling-recall pattern under the
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
