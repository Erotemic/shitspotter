# RF-DETR Seg 2XL four-GPU training campaign

Last updated: 2026-09-19

Status: local preflight is implemented and validated on real data. Core KDK
geometry/cache/adapter work and the ShitSpotter experiment surface exist; no
production training has started. The next hard boundary is the aiq-gpu image,
loader-throughput, and four-rank smoke gate.

This document is the durable handoff and progress tracker for preparing the
next ShitSpotter training campaign. The target is a strong native instance
segmentation model trained with RF-DETR Seg 2XL on `aiq-gpu` (4 x 96 GB
Blackwell GPUs), while retaining a plausible inference path on `toothbrush`
(RTX 3090).

The immediate objective is **ready to train**, not completion of the long
round-0 run. The last preparation step must print one exact command that starts
round 0, but must not execute that command automatically.

## How to use and maintain this document

Update this file whenever a gate changes state. A completed checkbox must link
or point to durable evidence: a test name, output manifest, log, image digest,
or measured report. Do not mark a stage complete because code exists; mark it
complete only when its acceptance evidence exists.

When resuming after lost context:

1. Read this document and the newest entry in `dev/journals/codex.md`.
2. Inspect `git status --short` in both repositories. Existing unrelated dirty
   files belong to the maintainer and must not be disturbed.
3. Verify the KDK RF-DETR submodule pin with
   `git -C /home/joncrall/code/kwcoco_detector_kit submodule status tpl/rf-detr`.
4. Resolve the dataset revisions recorded below; do not silently follow a
   changed `train.kwcoco.zip`, `vali.kwcoco.zip`, or `test.kwcoco.zip` link.
5. Read the latest stage-state/provenance JSON in the experiment output before
   rerunning an expensive action.

## Scope and repository boundary

Generic machinery belongs in `/home/joncrall/code/kwcoco_detector_kit`:

- segmentation-preserving tiling;
- positive/negative/ignore classification and invariants;
- a virtual negative-candidate representation if the design spike validates it;
- deterministic negative selection and cumulative hard-negative replay;
- batched and deterministically sharded mining;
- the RF-DETR trainer and predictor adapters;
- native-mask prediction/evaluation support;
- the dedicated RF-DETR Docker image and generic runtime support;
- provenance and generic tests.

Project-specific policy belongs in this repository, under the eventual
`experiments/rfdetr_seg_v1/` directory:

- concrete ShitSpotter manifests and category `poop`;
- tile sizes/scales, safety margins, negative ratios, and round definitions;
- image tag and build invocation;
- `aiq-gpu` mounts, paths, output directories, and launch commands;
- census, smoke-dataset construction, evaluation commands, and the staged
  driver.

Do not copy RF-DETR into ShitSpotter, revive the GeoWATCH trainer, or put
ShitSpotter-specific assumptions in KDK.

## Confirmed facts

- KDK pins RF-DETR tag `1.10.1` at commit
  `e3fc28795f2a4303069c6b72e80431e5ed716030`.
- In pinned RF-DETR 1.10.1, `RFDETRSeg2XLarge` is a built-in Apache-2.0
  variant with default weight name `rf-detr-seg-xxlarge.pt`; it does not use
  the optional `rfdetr_plus` package. The earlier Plus/licensing claim was
  stale and was corrected against the pinned source and model table.
- RF-DETR requires Transformers `>=5.1,<6`; the existing OpenGroundingDINO
  image pins Transformers `<4.47`. A separate training image is mandatory.
- The pinned RF-DETR Roboflow-style COCO loader accepts absolute `file_name`
  values. Its required small layout is `train/_annotations.coco.json` and
  `valid/_annotations.coco.json`; image assets do not need to be copied merely
  to sit beside those files.
- Its canonical selected checkpoint is `checkpoint_best_total.pth`.
- Upstream's supported multi-GPU route is `torchrun` plus an explicit Lightning
  `devices` setting. Launching four processes without that setting can silently
  fail to use the intended four-device configuration.
- KDK multiscale tiling now transforms both polygon and RLE segmentations and
  enforces positive/negative/ignore safety invariants.
- KDK now supports deterministic pre-encode negative sampling, batched/sharded
  mining, and cumulative hard-negative replay.
- KDK's current TileStore backends are materialized JPEG/kwcoco and
  WebDataset. There is no source-window/virtual-crop TileStore today.
- The installed `kwcoco` is version 0.8.9. Its Python API supplies the dataset
  algebra needed here: `subset`, `union`, `reroot`, validation, category
  operations, and serialization.
- Every current target mask in the identified train/validation/test revisions
  is a kwcoco polygon dictionary (`exterior`/`interiors`), not COCO RLE. The
  earlier preliminary RLE classification was wrong because it treated every
  segmentation dictionary as RLE. Polygon geometry is the production priority;
  exact RLE behavior remains covered as a required generic invariant.
- `aiq-gpu` does not resolve from the environment used for this audit. Remote
  build/smoke work must run from a host with the correct DNS/SSH setup.

## Dataset identity and preliminary census

The canonical symlinks currently declare the following targets, but are broken
in this environment because they point through `/home/joncrall/...`. Matching
files exist under `/data/joncrall/dvc-repos/shitspotter_dvc/`.

| Split | Declared revision | Images | Poop-annotated images | Zero-poop images | Poop annotations | Poop masks |
|---|---|---:|---:|---:|---:|---:|
| train | `train_imgs11247_7f54a82d.kwcoco.zip` | 11,247 | 3,897 | 7,350 | 8,176 | 8,176 polygon |
| validation | `vali_imgs1258_9cdcc10a.kwcoco.zip` | 1,258 | 482 | 776 | 1,019 | 1,019 polygon |
| test | `test_imgs121_80c4a26b.kwcoco.zip` | 121 | 116 | 5 | 2,180 | 2,180 polygon |

These are preliminary metadata counts, not the completed census. The test
revision is recorded only to freeze split identity. It must not be scored while
choosing scales, negative ratios, epochs, thresholds, or mining rounds.

Maintainer/remote confirmation still needed:

- [ ] Confirm these are the intended canonical split revisions.
- [x] Resolve the split manifests from an explicit configurable `/data/...`
  root without changing split membership.
- [x] Confirm Seg 2XL is built into the pinned Apache-2.0 RF-DETR package and
  does not require `rfdetr_plus`.
- [ ] Identify the shell/host from which `aiq-gpu` and Docker are available.

## Dataset implementation rules

Use the Python `kwcoco.CocoDataset` API inside KDK and the ShitSpotter staged
driver. The CLI is for human inspection and reproducible one-liners, not the
primary implementation mechanism.

Routine operations:

- `CocoDataset.subset()` after KDK has selected stable tile identities;
- `CocoDataset.union()` only to materialize an already-decided composition;
- `CocoDataset.reroot(absolute=True)` for RF-DETR annotation layouts;
- `CocoDataset.validate()` after every important generated dataset;
- ordinary kwcoco serialization to preserve image and annotation metadata.

Use `conform` conservatively. Tile generation must explicitly produce correct
width, height, bbox, area, segmentation, categories, and image metadata.
Conformance is allowed only for a specific known downstream requirement; it is
not a repair pass on every intermediate artifact.

`kwcoco.union()` is not the replay-bank policy. KDK must first select and
deduplicate stable tile identities, enforce hard/random quotas and source/scale
caps, and only then use kwcoco to construct the output dataset.

After every subset/union/composition, assert that these first-class image fields
survived:

- `tile_identity`
- `tile_role`
- `tile_source_gid`
- `tile_scale_name` and scale factor
- crop extent in source coordinates
- `negative_origin` where applicable
- source dataset fingerprint and cohort/context metadata where available

## Intended data flow

```text
canonical ShitSpotter kwcoco
        |
        +-- positive windows
        |       +-- transform/crop masks
        |       +-- materialize once in a reusable fast-storage tile cache
        |
        +-- fixed validation windows (positive + negative)
        |       +-- materialize once in the same reusable cache
        |
        +-- safe negative-window index
                +-- source image + scale + crop metadata only
                +-- batched/sharded on-demand mining
                +-- deterministic hard bank + broad/random selection
                        +-- materialize/cache every admitted negative
                                +-- compose cache references with positives
                                        +-- RF-DETR annotation-only layout
```

Unsafe/ignored windows should be counted and recorded in the build report or a
lightweight index. Do not encode their JPEGs unless a diagnostic visualization
sample is explicitly requested.

## Storage architecture and I/O rule

Lazy windows are useful for defining a very large candidate universe and for
sampling it without paying the permanent disk cost up front. They are **not**
the intended training data path. Repeated source-image decode, scale, and crop
work can starve the GPUs before GPU memory or compute is saturated. Every tile
admitted to a training or repeatedly consumed validation pool must therefore be
materialized on fast storage before training begins.

Use two distinct layers:

1. **Candidate/control plane:** lightweight source-window records used for
   census, role classification, sampling, and (if benchmarks support it)
   one-pass mining.
2. **Training/data plane:** immutable encoded tile assets plus ordinary kwcoco
   annotations, optimized for repeated dataloader access across many epochs.

The materialized data plane uses a deterministic recipe-addressed cache keyed by
the materialization identity defined below:

- positive tiles are encoded once and reused by every round;
- the fixed validation pool is encoded once and reused by every in-training
  validation pass and exploratory checkpoint evaluation;
- a negative tile is encoded the first time it is admitted by broad sampling
  or the hard bank, then reused in later rounds;
- round datasets contain small manifests with absolute references to the cache,
  not copied or re-encoded per-round assets;
- smoke datasets and round datasets are immutable selection manifests; the
  cache is a separate persistent artifact that grows monotonically as useful
  tiles are admitted;
- writes use same-directory temporary files, validation, and concurrency-safe
  atomic publication, and a cache hit is accepted only after geometry/config
  metadata and image readability validate;
- cache layout must avoid an unbounded number of files in one directory;
- the cache must live on, or be staged to, storage fast enough for the actual
  four-GPU dataloader. Network/archive storage remains provenance/source
  storage, not automatically the training hot path.

Extend KDK's existing JPEG/KWCoco tile machinery with deterministic cache paths,
validation, and reuse semantics where practical. Do not introduce a parallel
general storage framework or heavyweight cache database without benchmark
evidence that the simple design is insufficient. Hash-prefix directories plus
small provenance sidecars are the initial design.

RF-DETR currently consumes ordinary COCO image paths, so the initial data plane
should use encoded image files rather than requiring an upstream WebDataset
integration. A future shard cache is reasonable only if ordinary files cannot
sustain measured training throughput and it can be integrated without forking
RF-DETR's loader architecture.

Storage acceptance is based on throughput, not just successful reads or disk
size. Benchmark cold and warm behavior through RF-DETR's actual PyTorch loader
with the real container, worker count, prefetching, pinned memory, augmentation,
mount, batch size, and four-rank access pattern. Record examples/s, data-wait
time, step throughput, CPU utilization, GPU utilization, and cache size. The
selected layout must feed the measured training step rate with explicit
headroom.

## Stable tile identity

Tile identity is required before mining or replay work. It must not depend on
the round, score, role, output filename, or generated image ID.

The proposed identity is a versioned SHA-256 over canonical JSON containing:

- source dataset fingerprint;
- source gid and normalized source asset identity;
- requested and actual source scale;
- crop extent in scaled coordinates and source coordinates;
- tile-identity schema version and coordinate convention.

Changing a safety margin or quality threshold may change a tile's
classification without changing the underlying window identity. Store policy
fingerprints separately—including the target category set, intersection/safety
rules, and quality thresholds—so the same spatial candidate can be compared
across policy versions. Output size, padding, codec, and writer behavior belong
to the materialization identity below. Add collision and cross-run stability
tests.

## Materialization identity and cache publication

`tile_identity` identifies the underlying spatial sample. It deliberately does
not identify the encoded bytes. Introduce a separate versioned
`materialization_identity`, also a SHA-256 over canonical JSON, containing at
least:

- `tile_identity`;
- output width and height;
- scaling and interpolation policy;
- crop rounding and padding policy/value;
- EXIF orientation and colorspace/channel behavior;
- output codec and quality/compression settings;
- materialization implementation version.

The cache asset path uses this materialization identity, for example
`cache/12/12f083....jpg`, with an optional adjacent JSON provenance record.
Changing JPEG quality, resize semantics, orientation handling, or output shape
must produce a different cache key without changing the spatial tile identity.

Publishing must be idempotent and safe when several workers discover the same
tile:

1. Check an existing entry and validate its sidecar, dimensions, and decodability.
2. Encode to a uniquely named temporary file in the destination directory.
3. Validate the temporary image and provenance before publication.
4. Publish with a per-key lock or an atomic no-clobber filesystem operation;
   if another worker wins, validate and reuse the winner.
5. Never treat a partial file as a cache hit. Quarantine an invalid existing
   entry rather than silently overwriting unexplained bytes.

Tests must cover concurrent duplicate requests, interrupted temporary writes,
valid cache reuse, changed materialization configuration, and corrupt-entry
detection. A cache database is out of scope unless the filesystem design is
measured and found inadequate.

## Tile-role semantics

Classification is based on target-mask intersection, not merely retained bbox
or total area:

- **positive**: target mask content survives and every intersecting target is
  represented acceptably in the emitted supervision;
- **negative**: there is no target-mask intersection and no target within the
  configured conservative boundary margin;
- **ignore**: a target intersects, or is close enough to make background
  supervision unsafe, but the window cannot retain valid positive supervision.

If a crop has one acceptable object and one unusable clipped fragment, it must
not silently train the fragment as background. Unless RF-DETR is proven to
support an appropriate ignore-region representation, classify the whole window
as ignore.

Negative origins should distinguish at least:

- trusted zero-annotation source image;
- safe background window from an annotated source image.

## Candidate-index and mining-storage design spike

This investigation is a prerequisite only for choosing the candidate-universe
scan implementation. It cannot reverse the decided requirement that training
and repeated validation consume materialized cached assets.

Decided boundary: keep the negative universe as lightweight source-window
records, but materialize every negative admitted to a training pool. The open
question is how to make the one-pass mining scan efficient without eagerly
encoding the entire candidate universe.

The spike must answer:

1. Can candidate records be represented in a compact kwcoco-compatible index
   without pretending each crop is already a standalone image asset?
2. Can decoding be grouped by `(source_gid, scale)` so one source decode/resize
   serves many windows rather than reopening the image per candidate?
3. Can four deterministic rank shards cover the universe without overlap or
   omission and resume from atomic shard ledgers?
4. Does batch construction keep the GPU fed on `aiq-gpu` when source assets are
   on the actual mounted storage?
5. Can selected negative crops be materialized deterministically with their
   stable tile and materialization identities and provenance preserved?
6. Does RF-DETR need any change beyond receiving the materialized selected
   round corpus? The expected answer is no.
7. Is grouped lazy mining fast enough, or does mining require a bounded
   transient cache of decoded/scaled sources or candidate chunks?
8. Does the persistent materialized training cache sustain four-GPU training
   without input stalls under the intended worker count?

Compare at least:

- virtual on-demand windows grouped by source/scale;
- virtual windows backed by a bounded transient decoded-scale/chunk cache;
- a small representative eager candidate subset as a throughput baseline, not
  as a proposed full-universe architecture;
- direct kwcoco/ndsampler window loading where useful as existing prior art.

Do not choose WebDataset merely because it exists: KDK's own storage notes say
it is awkward for iterative round composition, and RF-DETR already consumes
ordinary COCO paths. The decision report must record index size, candidate
count, source decode/crop throughput, training-cache read throughput, GPU
utilization during mining and training, temporary/permanent disk use, cache-hit
reuse across rounds, and implementation complexity.

Decision gate:

- [ ] `virtual_negative_design.json` and a short narrative exist.
- [ ] A representative benchmark supports the selected representation.
- [ ] The selected representation can reproduce the same pixels and identity
  as deterministic materialization of a sampled candidate.
- [ ] The materialized training cache feeds the measured four-GPU step rate
  with documented headroom.

## Work plan and progress gates

### Stage 0 — freeze inputs and establish provenance

- [ ] Confirm and record exact ShitSpotter, KDK, and RF-DETR commits.
- [x] Record resolved dataset paths and manifest hashes in
  `input_verification.json`; full source-asset hashing remains optional because
  tile cache admission hashes every used source asset.
- [ ] Record host/storage mount expectations for `aiq-gpu`.
- [ ] Record the resolved `rf-detr-seg-xxlarge.pt` weight digest and download
  provenance after the first controlled model instantiation.
- [x] Derive durable stage status from required artifacts, matching
  fingerprints, and output validation. Transient markers may be added for
  expensive work, but there is no independent workflow-state database.

Gate evidence: `inputs.json`, `provenance.json`, and validated path report.

### Stage 1 — full census and tile-policy simulation

The census action must record:

- image, annotation, annotated-image, and zero-target-image counts;
- missing/invalid masks and segmentation representation distribution;
- bbox and mask-area distributions in pixels and normalized image area;
- apparent object side/diameter distribution;
- source image dimensions and orientation;
- cohort/context distribution;
- predicted positive, negative, and ignored window counts by source scale;
- predicted disk use for the persistent positive cache, fixed validation cache,
  round-0 admitted-negative cache, and incremental cache growth expected from
  each later mining round;
- apparent-size distributions induced by candidate scale policies.

Evaluate the historical `1.0,0.66,0.40,0.25` scales at 768, plus census-derived
alternatives. Do not adopt the old scales, a 3:1 negative ratio, or three rounds
without evidence.

- [x] Census implementation uses kwcoco Python APIs.
- [x] Metadata census JSON and a full geometry-only policy simulation are
  generated locally from the canonical manifests.
- [x] Proposed 768 scale policy and deterministic negative retention are
  quantified: train has 62,426 positive, 197,039 retained-negative, 964,764
  skipped-negative, and 35,219 ignored windows; validation has 7,450 positive,
  22,988 retained-negative, 106,686 skipped-negative, and 4,324 ignored
  windows. The predicted round-0 manifest references 280,142 materialized
  tiles (91.5 GB using the measured smoke JPEG mean). Runtime/quality acceptance
  still requires aiq profiling.

Gate evidence: `census.json`, `census.md`, and policy comparison tables.

### Stage 2 — segmentation-safe tiling and role invariants in KDK

Implement generic source scaling, crop/intersection, clipping, and translation
for segmentation geometry. Recompute bbox and area from the surviving mask.

Test priority:

1. COCO RLE -> scale -> crop -> RLE, compared pixel-exactly with an independently
   rasterized nearest-neighbor reference;
2. RLE boundary fragments and empty intersections;
3. polygon crossing a boundary with expected bbox and approximate area;
4. scaling plus polygon crop;
5. multipolygon preservation;
6. mixed acceptable/unacceptable intersecting targets;
7. safety margin behavior;
8. invariant that a negative has no retained annotation or known target-mask
   intersection.

Every positive, fixed-validation, and admitted training-negative tile must be
materialized before its training/evaluation stage begins. Unselected negative
candidates and ignored windows remain index records only.

- [x] Geometry tests exercise behavior absent from the old implementation.
- [x] RLE tests pass with exact raster equivalence.
- [x] Polygon/multipolygon tests pass.
- [x] Negative-safety invariants pass.
- [x] Merge/composition rejects a mislabeled negative input.
- [x] Tile versus materialization identity and concurrent cache-publication
  tests pass.

Gate evidence: named pytest results and a small visual geometry report.

### Stage 3 — native-mask and optional-batch predictor protocol

Preserve the minimal predictor protocol while allowing records to carry a
native `mask`. Add optional `predict_batch(images, orig_sizes)` with a generic
scalar fallback. Route native masks through KDK's existing mask-to-annotation
utilities instead of creating a parallel segmentation framework.

- [x] Existing box-only predictors remain compatible.
- [x] Batch/scalar prediction parity test passes.
- [x] Native mask becomes a valid KWCoco segmentation with matching bbox/area.
- [x] Packaged prediction routes native masks to segmentation annotations.

Gate evidence: unit/integration tests and a produced prediction kwcoco.

### Stage 4 — thin RF-DETR Seg adapter in KDK

Implement only the variants needed now, especially `rfdetr-seg-2xlarge`.

The adapter will:

- generate a serialized configuration and small Python launcher;
- create the upstream `train/valid` annotation layout with absolute paths and
  no bulk asset copy;
- instantiate `RFDETRSeg2XLarge` using COCO-pretrained weights;
- make resolution, augmentation backend, devices, precision, batch,
  accumulation, learning rates, epochs, seed, and validation behavior explicit;
- launch the pinned upstream-supported `torchrun` path;
- select `checkpoint_best_total.pth`;
- reload via RF-DETR's checkpoint API;
- emit boxes, scores, labels, and full-resolution masks.

Provisional baseline, subject to smoke evidence:

- resolution 768;
- four GPUs;
- BF16;
- physical batch 4/GPU;
- gradient accumulation 1;
- effective batch 16;
- fixed model input, with scale diversity supplied by the tile corpus.

Profile batch 8/GPU, but do not adopt it automatically because it changes the
effective batch to 32 and therefore optimization behavior.

- [x] Config-generation tests pass without importing RF-DETR.
- [x] Annotation-only layout generation resolves absolute image paths.
- [ ] Checkpoint selection/reload tests pass.
- [ ] Predictor emits native masks.

Gate evidence: generated config/layout fixtures and adapter test results.

### Stage 5 — dedicated RF-DETR container and basic GPU smoke

KDK now contains `docker/rfdetr/`, separate from the incompatible
OpenGroundingDINO/Transformers environment. It pins the RF-DETR submodule and
records the KDK, RF-DETR, and Dockerfile identities.

- [x] Dedicated Dockerfile and aiq CUDA 13.2/Blackwell build helper exist.
- [ ] Image builds on `aiq-gpu`.
- [ ] Container sees all four GPUs and reports its immutable image ID.
- [ ] KDK/RF-DETR imports and pretrained model instantiation succeed.
- [ ] Native forward/prediction produces a mask.

Gate evidence: build log, `/etc` provenance JSON, `docker inspect`, and GPU
smoke log. This gate precedes acceptance of the real mining/storage design.

### Stage 6 — scalable mining and cumulative replay

Mining requirements:

- use `predict_batch` when available;
- stable deterministic rank assignment from `tile_identity`;
- no overlap or omission across four rank shards;
- atomic shard score ledgers and deterministic merge;
- record a result for every attempted candidate, including failures;
- retain max score, top prediction bbox/mask location, class, source/scale/crop,
  checkpoint, rank, and enough metadata for visualization;
- resume only completed shards and never mistake partial output for success.

Replay-bank policy runs before kwcoco composition:

- union identity records from all completed mining rounds;
- deduplicate by `tile_identity`;
- retain cumulative valuable hard examples;
- impose configurable per-source/per-scale caps;
- add fresh stratified random coverage;
- use explicit hard/random quotas and seeds;
- materialize newly admitted negatives into the shared tile cache;
- reuse already-cached replay negatives without re-encoding or copying;
- assert role/provenance metadata survives final materialization.

- [x] Deterministic SHA-256 sharding and exact disjoint-coverage tests pass.
- [ ] Interrupted-shard recovery test passes.
- [x] Cumulative replay deduplication, quota, cap, and fresh-random selection
  implementation has focused tests.
- [ ] Cache idempotence and atomic-write recovery tests pass.
- [ ] Round 1 contains cumulative hard examples and fresh broad coverage.

Gate evidence: mining ledger schema, replay manifest, tests, and a visualization
sample.

### Stage 7 — validation evaluation readiness

There are two deliberately different validation products:

1. **Train-validation pool:** immutable materialized tiles consumed by RF-DETR
   every epoch for cheap checkpoint feedback.
2. **Canonical validation evaluation:** fixed windows over the original 1,258
   validation images, native masks transformed back into source coordinates,
   overlap duplicates merged, and metrics computed on source-image kwcoco.

Canonical validation must retain negative source images and must produce:

- COCO box AP;
- COCO instance-mask AP;
- count/fraction of original zero-poop images with a false positive at configured
  thresholds;
- predictions per negative image;
- maximum false-positive score distribution and quantiles.

Use validation for checkpoint and round decisions. Preserve the existing
ShitSpotter salient/pixel path as a later compatible evaluation if it can be
called without old trainer machinery. Do not score the held-out test split in
this preparation campaign.

- [ ] Box AP runs on validation.
- [ ] Instance-mask AP runs on validation.
- [x] Generic source-image FP report is integrated and unit-tested; real
  validation output awaits a checkpoint.
- [ ] Test-evaluation action is unavailable from the exploratory driver or
  guarded against accidental use.

Gate evidence: validation metrics JSON and negative-FP report.

### Stage 8 — compact ShitSpotter experiment surface

Create `experiments/rfdetr_seg_v1/` with:

- `README.md` explaining the campaign and exact operational commands;
- one central `config.yaml`;
- one staged `run.py` driver.

Expected actions:

- `status`
- `census`
- `design-negative-store`
- `build-image`
- `prepare-tiles`
- `materialize-pool`
- `prepare-smoke`
- `smoke-forward`
- `smoke-train4x`
- `prepare-round0`
- `train-round0` (must require an explicit invocation)
- `mine`
- `prepare-next-round`
- `evaluate-validation`

The driver calls KDK APIs and kwcoco Python APIs. It must not reimplement
tiling, COCO conversion, mining, replay policy, or the trainer.

- [x] Config and staged driver exist with artifact-derived status.
- [x] Real-data input verification, deterministic smoke construction, and the
  full metadata-only tile-policy simulator run locally. The simulator exactly
  reproduced smoke materialization role counts (train 57/803/36 and validation
  32/391/25 positive/negative/ignore).
- [ ] Every aiq build/mount/launch command is visible in the README.
- [ ] Expensive stages write atomic state and config fingerprints.
- [ ] `status` identifies stale or mismatched outputs instead of reusing them.

Gate evidence: experiment directory and driver tests.

### Stage 9 — genuine four-GPU smoke ladder

Construct a tiny real ShitSpotter-derived dataset with positive masks,
trusted zero-annotation images, and validation positives/negatives.

Run, in order:

1. CPU unit and integration suite.
2. Container import/environment tests.
3. Single-device model instantiation and mask prediction.
4. A short single-GPU train/validation/reload pass if useful for debugging.
5. `torchrun --nproc_per_node=4` with explicit Lightning devices on real data.

The four-GPU smoke must prove:

- all four ranks participate;
- segmentation loss executes;
- validation executes on positive and negative images;
- a checkpoint is written atomically;
- `checkpoint_best_total.pth` reloads;
- KDK writes a prediction kwcoco containing segmentation;
- peak memory and throughput are recorded for batch 4/GPU;
- batch 8/GPU is either measured or explicitly rejected with evidence;
- the materialized training pool, on its intended storage/mount, does not
  starve the training loop under the intended RF-DETR worker, prefetch,
  pinned-memory, augmentation, and four-rank configuration.

- [ ] Four-rank participation proven in logs.
- [ ] Segmentation train and validation paths complete.
- [ ] Checkpoint reload and mask prediction pass.
- [ ] GPU and dataloader memory/throughput report exists.

Gate evidence: smoke manifest, rank logs, checkpoint, prediction kwcoco,
validation metrics, and profile JSON.

### Stage 10 — freeze the round-0 launch

Prepare round 0 from all positives plus the census-selected stratified negative
coverage. Sampling must be stratified across source images and scales so large
images or dense grids cannot dominate.

Before exposing the command, `status` must verify:

- all prior gates are complete;
- dataset/config/provenance fingerprints match;
- every referenced positive and negative asset exists in the validated
  materialized tile cache;
- every repeatedly consumed validation asset exists in the validated cache;
- positive, selected-negative, and ignored counts are recorded;
- every training negative passes the safety invariant;
- validation retains negatives;
- four-GPU smoke evidence matches the intended image and code;
- no held-out test evaluation has occurred.

- [ ] Round-0 manifest and composition report validate.
- [ ] One exact production command is printed and documented.
- [ ] The production command has not been executed by preparation automation.

Gate evidence: `round0_manifest.json`, final readiness report, and the exact
launch command.

## Mining-round decision policy

Do not hard-code three rounds.

- Round 0: positives plus broad stratified negatives.
- Mine 0: score all practical candidates using four deterministic shards.
- Round 1: positives plus cumulative hard bank plus fresh broad/random
  coverage.
- Mine 1: refresh the candidate universe if validation shows useful remaining
  false positives.
- Round 2: prepare only when validation box/mask AP or negative-FP evidence
  justifies the cost.

The stop/continue report must show metric deltas, hard-bank composition,
candidate score shifts, and training cost. Test results cannot inform it.

## Ready-to-train acceptance matrix

| # | Criterion | Status | Evidence |
|---:|---|---|---|
| 1 | Segmentation-preserving tile geometry and safe negatives | Not started | — |
| 2 | Thin RF-DETR Seg trainer/predictor with native masks | Not started | — |
| 3 | Dedicated provenance-stamped RF-DETR image builds on aiq | Not started | — |
| 4 | Genuine four-GPU BF16 smoke completes | Not started | — |
| 5 | Smoke checkpoint reloads to segmented KWCoco prediction | Not started | — |
| 6 | Repository boundary and compact ShitSpotter driver respected | Not started | — |
| 7 | Full census and exact tile-role counts recorded | Not started | — |
| 8 | Round 0 includes substantial stratified negative coverage | Not started | — |
| 9 | Batched/sharded mining and cumulative replay are reproducible | Not started | — |
| 10 | Exact round-0 command exists but has not been launched | Not started | — |

## Decisions and open questions

| Date | Decision or question | State | Rationale/evidence |
|---|---|---|---|
| 2026-09-18 | Use RF-DETR `1.10.1` at `e3fc2879...` | Decided | Pinned KDK submodule, not upstream `develop` |
| 2026-09-18 | Use a separate RF-DETR container | Decided | Incompatible Transformers ranges |
| 2026-09-18 | Use kwcoco Python API for dataset algebra | Decided | Avoid duplicate conversion code and subprocess orchestration |
| 2026-09-18 | Run `validate` routinely; use `conform` only for explicit requirements | Decided | Generated data must be correct by construction |
| 2026-09-18 | Prioritize production polygon geometry plus exact generic RLE tests | Corrected | Current target masks are kwcoco polygon dictionaries; the initial RLE claim was a classifier bug |
| 2026-09-18 | Do not materialize ignored windows by default | Decided | They are not training inputs; retain counts/index metadata |
| 2026-09-18 | Materialize every tile admitted to a training pool | Decided | Repeated lazy crop/decode work bottlenecks I/O before GPU compute |
| 2026-09-18 | Materialize the fixed validation pool | Decided | Validation is repeatedly consumed during training and can cause the same starvation |
| 2026-09-18 | Reuse a deterministic recipe-addressed materialization cache across rounds | Decided | Fast repeated training without per-round copies; encoded SHA-256 provides byte integrity |
| 2026-09-18 | Keep the unselected negative universe virtual | Decided | Avoid permanent cost for candidates that never train |
| 2026-09-18 | Choose lazy/transient mining representation from benchmark | Open gate | Full-universe mining can itself become source-I/O bound |
| 2026-09-18 | Separate tile identity from materialization identity | Decided | Byte identity also depends on resize, padding, orientation, codec, and implementation policy |
| 2026-09-18 | Extend existing KDK tile machinery for the cache | Decided | Prefer deterministic paths/reuse over a parallel storage framework |
| 2026-09-18 | Initial scale policy and negative ratio | Open | Must follow the completed census |
| 2026-09-18 | Batch 4 vs 8/GPU | Open | Profile both; preserve effective batch semantics |
| 2026-09-18 | Whether to run round 2 | Open | Validation evidence only |

## Risks to keep visible

- Mask scaling can introduce off-by-one or interpolation errors. Current data
  uses polygon dictionaries; generic RLE transforms still require
  nearest-neighbor/reference-equivalence tests.
- Bbox-only intersection is insufficient to certify a negative.
- A positive tile with an unrepresented clipped fragment can still teach false
  background unless the whole window is ignored.
- Virtual candidates trade disk pressure for repeated decode/resize cost;
  restrict them to candidate selection/mining, group work by source and scale,
  and permit a bounded transient mining cache when measurements justify it.
- A materialized corpus on slow or contended storage can still starve four
  GPUs; benchmark the exact mount and dataloader rather than assuming JPEGs
  alone solve throughput.
- A deterministic materialization cache needs atomic writes and validation so an interrupted
  encoder cannot leave a plausible-looking corrupt cache hit.
- Keying cached bytes only by spatial tile identity can reuse stale assets after
  codec or resize-policy changes; use a separate materialization identity.
- `kwcoco.union()` may reassign gids; stable identity must live in explicit
  metadata, not image IDs.
- A four-process `torchrun` is not proof of four-GPU training without rank,
  device, throughput, and memory evidence.
- A successful container import is not proof that the Plus model weights,
  segmentation loss, validation, and checkpoint reload all work.
- Broken canonical symlinks can cause accidental use of stale revisions.
- The existing ShitSpotter worktree is dirty; unrelated files and submodule
  changes must remain untouched.

## Final handoff contents

At ready-to-train completion, report:

- changes in each repository;
- exact KDK and RF-DETR revisions;
- exact unit, integration, container, and GPU smokes run;
- canonical split census and measured positive/negative/ignored counts;
- chosen scale and negative policies with their evidence;
- candidate/mining-store benchmark, materialized training-cache design, and
  measured cache throughput;
- Docker tag, immutable image ID, and embedded provenance;
- four-GPU memory and throughput measurements;
- validation box AP, mask AP, and negative-image FP statistics from the smoke;
- remaining maintainer decisions;
- the single exact command that begins production round 0.
