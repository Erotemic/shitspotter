## 2026-09-22 18:30:00 -0400

Model: GPT-5.6 Sol.

The user asked for a durable record of the hard-negative / missing-truth review
workflow developed during the RF-DETR Seg 2XL campaign. The important conceptual
change is that source-space detector predictions are treated as durable evidence
while prediction-vs-truth classifications are intentionally ephemeral: every
manual LabelMe correction is followed by `gather`, `make_splits`, and a fresh
`prediction-review` against current truth without rerunning detector inference.

I documented the transactional LabelMe loop rather than extending package code.
The workflow stages zero-overlap unexplained proposals into an isolated LabelMe
workspace, copies existing canonical sidecars (preserving point metadata), seeds
model proposals with a reserved unresolved label, and requires the reviewer to
relabel, correct, or delete every proposal. Copy-back is dry-run first and checks
for concurrent canonical-sidecar changes. After apply, KWCoco truth is rebuilt
and the same prediction file is reclassified before the next batch.

The document also distinguishes annotation QA from training hard-negative
admission. Newly labeled nuisances such as leaf/trash/pinecone improve canonical
truth immediately, but old negative candidate indexes and derived pools are stale
because their safety decisions depended on the old truth fingerprint. In
contrast, the prediction KWCoco remains reusable as long as source pixels and
inference policy are unchanged.

Two operational edge cases are recorded because they affected this campaign:
LabelMe point-only metadata must not imply every annotation has a bbox, and some
historical `*.scrubbed.json` sidecars reference the unscrubbed image in `imagePath`.
The latter is described as a one-off conservative data-hygiene repair, not a new
permanent package feature.

## 2026-09-26 20:17:42 -0400

Model: GPT-5.6 Sol.

The corrected-truth campaign progressed through a full new RF-DETR Seg 2XLarge
training attempt and a second round of annotation QA. The important result from
the last training run is that correcting the training truth alone produced an
early validation peak slightly above the previous v3 best while the training
server was still using the old, not-yet-reviewed validation truth. The EMA
trajectory was:

```text
validation epoch    box mAP50:95    segm mAP50:95    F1
1                   0.6667          0.6396           0.8059
2                   0.6953          0.6652           0.8290
3                   0.7029          0.6743           0.8354
4                   0.6811          0.6573           0.8332
5                   0.6810          0.6562           0.8336
6                   0.6794          0.6566           0.8343
```

The previous v3 best was approximately box mAP50:95 = 0.6975 and segmentation
mAP50:95 = 0.6705. Thus the corrected-training run's epoch-3 checkpoint was
slightly better against the same old validation target, but the gain was small
and the validation truth itself was known to contain annotation defects. The
later epochs fell into a stable lower-AP plateau, so the practical conclusion is
that this optimizer recipe still peaks very early. The result is encouraging but
is not being treated as proof that +0.0038 segmentation AP is statistically
meaningful.

Rather than spend another run solely reproducing that comparison after replacing
the validation annotations, the next campaign will use the corrected validation
truth and change the training distribution in a way motivated directly by review
evidence. Validation review found real missing positives and named nuisance
objects. The v6 `reviewed_hardneg` policy therefore keeps the successful v3
optimizer settings but reserves 20% of the normal negative budget for legal
windows that substantially contain trusted named distractor annotations. The
remaining 80% uses the existing source/scale-stratified negative sampler. The
quota is a minimum explicit quota rather than a cap: normal sampling may select
additional nuisance-containing windows. Validation gets a fixed reviewed-
distractor quota as well for the requested campaign, but validation tiles remain
strictly separate from training.

Truth semantics were refined during this cleanup. `residual` and `residue` were
used for messy cleanup regions and therefore must be treated as ignore regions,
not trusted background and especially not explicit hard negatives. The
historical `unkown` typo should be normalized to `unknown`; v6 rejects the typo
rather than silently perpetuating it. Named nuisance objects such as leaf, rock,
pinecone, trash, etc. remain trusted background and can seed reviewed-hard-
negative windows. Only `poop` is a detector target.

The validation/test review also exposed duplicate RF-DETR mask predictions where
a smaller polygon was almost completely contained by another same-class polygon.
KDK gained configurable mask containment suppression using
`intersection / min(area_a, area_b)` (mask IoMin), with 0.85 as the current
ShitSpotter starting point. The same setting can be applied at prediction time or
later to an existing prediction KWCoco during `prediction-review`, so completed
inference does not need to be rerun merely to remove containment duplicates.

The current local truth contains 32 train annotations with `category_id=None`
and no bbox/segmentation/keypoints. Inspection showed that these are legitimate
caption/image-metadata records (e.g. `grass; downview`, `noban;positive-only`,
`BAN:negative,error`) rather than unlabeled object geometry; one even has a null
caption but remains nonspatial. The previous v6 preflight incorrectly rejected
all uncategorized records. The corrected invariant is now:

```text
category unresolved + localized geometry
    -> malformed spatial truth; fail closed before training

category_id=None + no bbox/segmentation/keypoints
    -> nonspatial metadata; preserve and exclude from detection-schema checks
```

`verify-inputs` now reports the nonspatial metadata count/examples while
validating the actual detection annotations separately. This keeps metadata
semantics intact without weakening the guard against uncategorized polygons,
boxes, or keypoints.

Operationally, the intended next sequence is: finish authoritative LabelMe
cleanup; normalize `unkown`; regenerate train/validation/test KWCoco; sync the
corrected manifests to the training host; rebuild truth-dependent candidates and
pools while reusing raster cache; inspect the annotated-distractor quota counts;
then start `v6_reviewed_hardneg` fresh from upstream RF-DETR Seg 2XLarge
pretrained weights. Long-running commands should stay visible in tmux and use
`tee` for persistent logs rather than nohup/background-only workflows.

## 2026-09-27 09:15:00 -0400

Model: GPT-5.6 Sol.

The v6 `reviewed_hardneg` campaign completed successfully on `aiq-gpu` after the
corrected train/validation truth, reviewed distractor policy, and split-generation
cleanup were in place. The run stopped by early stopping after validation epoch 7.
The EMA trajectory was:

```text
validation epoch    box mAP50:95    box mAP50    box mAP75    mAR@500    F1      precision    recall    segm mAP50:95    segm mAP50
1                   0.6721          0.8700       0.7705       0.8748     0.8085  0.8643       0.7594    0.6443           0.8713
2                   0.6947          0.8891       0.7946       0.8829     0.8349  0.8728       0.8001    0.6680           0.8939
3                   0.7007          0.8956       0.8003       0.8905     0.8389  0.8729       0.8075    0.6747           0.9002
4                   0.6951          0.8916       0.7892       0.8815     0.8425  0.8884       0.8011    0.6709           0.8988
5                   0.6930          0.8904       0.7841       0.8852     0.8443  0.8969       0.7976    0.6673           0.8952
6                   0.6817          0.8826       0.7729       0.8756     0.8446  0.8923       0.8017    0.6590           0.8920
7                   0.6883          0.8851       0.7818       0.8815     0.8509  0.8783       0.8252    0.6645           0.8949
```

The best checkpoint was again validation epoch 3, with EMA segmentation
mAP50:95 = 0.6747 and box mAP50:95 = 0.7007. The trainer reported no monitored
metric improvement in the following four records and stopped as intended. The
saved artifact is:

```text
/data/users/jon.crall/shitspotter_rfdetr_v6_reviewed_hardneg/
    rounds/round0/runs/v6_reviewed_hardneg/checkpoint_best_ema.pth
```

The campaign completed with:

```text
/data/users/jon.crall/shitspotter_rfdetr_v6_reviewed_hardneg/OVERNIGHT_COMPLETE.json
```

This is an important negative result rather than an inconclusive run. The
previous corrected-truth experiment peaked at segmentation mAP50:95 = 0.6743
and box mAP50:95 = 0.7029, also at epoch 3. V6, despite corrected validation
truth plus explicit reviewed-distractor coverage, reproduced essentially the
same aggregate AP ceiling: 0.6747 segmentation / 0.7007 box. The repeatability
makes it unlikely that another minor learning-rate or duration tweak to this
same RF-DETR Seg 2XLarge recipe will yield a large gain.

The reviewed-hard-negative policy may nevertheless be changing the operating
point. F1 continued improving after aggregate AP peaked, reaching 0.8509 at
validation epoch 7 with precision 0.8783 and recall 0.8252. The v6 result should
therefore not be summarized as "hard negatives did nothing"; rather, they did
not materially change COCO-style aggregate AP while later epochs produced a
better precision/recall balance.

The metric shape suggests the next investigation should focus on localization
and mask fidelity rather than simply extending training. At the best epoch,
segmentation AP50 was 0.9002 while segmentation AP50:95 was 0.6747. Detection
at loose overlap is already strong, but performance falls substantially as IoU
requirements tighten. Before choosing a v7 training recipe, inspect AP by IoU
threshold, object/mask size, and high-confidence matches that pass IoU 0.5 but
fail stricter mask-overlap thresholds. This should distinguish a mask-head /
resolution limitation from annotation-boundary ambiguity or small-object IoU
sensitivity.

The source split semantics also changed during this campaign. The historical
BA/BAN logic used annotation presence to infer acquisition groups and exclude
images that might contain unannotated poop. That heuristic is no longer sound:
review added nuisance/ignore annotations, after/negative captures can still
contain real poop, and historical capture sequences were not always exact pairs
or triples. The dataset has now undergone sufficiently broad review that the
legacy exclusion machinery was disabled (`LEGACY_SYSTEM = False`). All reviewed
images are eligible for train/validation, while current spatial truth determines
positive, negative, and ignored regions. This intentionally separates detector
truth from historical acquisition-role inference.

After disabling the legacy exclusion path, the regenerated source splits were:

```text
                  n_anns    n_imgs
train               9388     11362
validation           1211      1271
test                  246       121
```

Compared with the immediately prior split generation, train gained 105 images
and validation gained 10 images while their annotation counts remained exactly
unchanged. These newly admitted source images are therefore annotation-free
background candidates, not additional labeled positives. KDK still controls the
actual fixed positive/negative tile budget, so admitting these images primarily
broadens the legal negative-source universe rather than increasing the training
set without bound.

The test manifest was also rebuilt cleanly instead of incrementally appending
LabelMe annotations to an existing KWCoco. Its resulting 246 annotations (227
`poop`) confirm that the previous multi-thousand-annotation test manifest had
been suffering from repeated annotation accumulation during `gather`. Until an
incremental gather implementation explicitly replaces per-image annotations,
test gathering should continue to rebuild from source rather than append onto a
previous test KWCoco.

Operational logs for this campaign are expected primarily at:

```text
/data/users/jon.crall/shitspotter_rfdetr_v6_reviewed_hardneg/logs/overnight.log
```

with trainer/run artifacts under:

```text
/data/users/jon.crall/shitspotter_rfdetr_v6_reviewed_hardneg/
    rounds/round0/runs/v6_reviewed_hardneg/
```

These are the first places to inspect when doing the planned IoU-threshold,
object-size, and mask-error analysis for the next model decision.

## 2026-09-27 10:48:00 -0400

Model: GPT-5.6 Sol.

Follow-up inspection of the structured v6 RF-DETR run metadata refined the
interpretation above. The compact run artifacts are more useful than the
campaign console log for optimization diagnosis. In particular,
`metrics.csv`, `training_config.json`, `policy.json`, `ROUND0_COMMAND.txt`, and
the generated RF-DETR config record the exact trajectory and effective trainer
settings.

The strongest new observation is that the training objective continued to
improve monotonically after validation AP peaked. Epoch-end training values
were approximately:

```text
validation epoch    train loss    cls CE    bbox L1    GIoU     mask CE    mask Dice
1                   15.5075       0.8008    0.02788    0.18804  0.02086    0.11906
2                   11.7721       0.6368    0.01895    0.14880  0.01703    0.09628
3                   11.2830       0.5967    0.01834    0.14444  0.01618    0.09352
4                   10.8987       0.5629    0.01766    0.14005  0.01575    0.09232
5                   10.4546       0.5273    0.01717    0.13643  0.01551    0.08955
6                   10.2120       0.5098    0.01692    0.13373  0.01505    0.08741
7                    9.9484       0.4859    0.01652    0.13128  0.01464    0.08631
```

Thus the run is not failing to optimize the training objective. Generalization
AP peaks while all major training-loss components keep falling. Meanwhile the
reported F1 continues to improve through epoch 7. This makes the failure mode
more specific: later optimization improves the operating-point precision/recall
tradeoff and training fit while degrading COCO-style localization/ranking AP.

There is also an important interaction between early stopping and the cosine LR
schedule. The configured base LR is `5e-5`, cosine `min_factor=0.05`, over a
15-epoch horizon, but early stopping ends the run after validation epoch 7. The
observed maximum optimizer LR near the end of each human-numbered epoch was:

```text
epoch 1    4.966e-5
epoch 2    4.941e-5
epoch 3    4.766e-5   # best validation AP
epoch 4    4.482e-5
epoch 5    4.109e-5
epoch 6    3.658e-5
epoch 7    3.156e-5   # early stop
```

The intended cosine floor would be about `2.5e-6`, so early stopping terminates
training while the LR is still roughly 63% of its initial value. Therefore the
previous statement that another duration/LR change is unlikely to help should
be weakened: a *minor arbitrary* tweak is not motivated, but the current setup
has not actually tested a low-LR refinement phase. A controlled schedule
ablation is warranted before concluding that RF-DETR itself is at its ceiling.
A good diagnostic experiment would keep the data/model/pool fixed and ensure
that the cosine schedule reaches its low-LR region, e.g. by disabling early
stopping for a bounded run or by using a shorter annealing horizon. A more
compute-efficient variant is to initialize a fresh low-LR fine-tune from the
best EMA checkpoint if RF-DETR's checkpoint-loading semantics are verified.
Treat this as a hypothesis test, not an assumed fix.

The exact model configuration also sharpens the mask-fidelity hypothesis. The
run used RFDETRSeg2XLarge at 768x768 with `mask_downsample_ratio=4`,
`mask_point_sample_ratio=16`, and mask CE/Dice coefficients of 5.0 each.
RF-DETR internal `multi_scale` was disabled, although `scale_jitter` remained
enabled and the upstream KDK tile pool already includes multiple source scales.
At the best validation epoch, box mAP50:95 was 0.70073 and segmentation
mAP50:95 was 0.67465, while segmentation AP50 was 0.90019. Both box and mask AP
peak together, so the ceiling should not be attributed solely to the mask head;
there is nevertheless a persistent mask-localization gap worth decomposing by
IoU and object size.

The generated training config had `seed=null`. Exact stochastic replication was
therefore not guaranteed. Convergence to essentially the same epoch-3 ceiling
as the earlier corrected-truth run is consequently stronger evidence that the
observed plateau is systematic rather than a single unlucky seed.

Before selecting a substantially different architecture, the next evidence
should come from two complementary diagnostics:

1. evaluate the best v6 checkpoint by IoU threshold and object/mask size, and
   inspect high-confidence matches that pass IoU 0.5 but fail stricter overlap;
2. run one controlled LR-schedule/refinement ablation that actually enters the
   low-LR regime while keeping the model, data, and sampling policy unchanged.

If neither changes the high-IoU/localization behavior materially, then a new
segmentation architecture or head becomes a much stronger next move.

## 2026-09-27 11:00:00 -0400

Model: GPT-5.6 Sol.

V7 is defined as a controlled follow-up to the completed v6 run.  V6 showed a
repeatable epoch-3 EMA segmentation peak near 0.675, but structured metrics also
showed that early stopping terminated training while the cosine LR was still
about 63% of its starting value.  The next run should therefore force the full
15-epoch cosine trajectory instead of interpreting the early-stop result as a
completed low-LR refinement test.

The user also requested that RF-DETR's internal multiscale training be enabled
and that physical batch size be doubled because the four-GPU job has substantial
memory headroom.  The resulting campaign config is:

```text
experiments/rfdetr_seg_v1/config.v7_multiscale_batch16_fullcosine.yaml
```

Trainer-side differences from v6 are deliberately explicit:

```text
run_name                       v7_multiscale_batch16_fullcosine
RF-DETR internal multi_scale   true
train batch / GPU              16   (v6: 8)
validation batch / GPU         16   (v6: 8)
number of GPUs                 4
global physical train batch    64
grad accumulation              1
epoch ceiling                  15
model LR                       5e-5 (unchanged)
encoder LR                     1e-5 (unchanged)
cosine min_factor              0.05 (unchanged)
warmup                         1 epoch
early stopping                 disabled
EMA / best metric              unchanged, segmentation mAP
```

This is intentionally *not* linear LR scaling with the doubled batch.  The v4
large-batch experiment already changed batch and LR together and did not improve
the baseline.  V7 keeps the successful v3/v6 LR values while testing three
requested/diagnostic changes: larger physical batch, RF-DETR internal
multiscale augmentation, and actually reaching the cosine low-LR regime.  These
changes mean v7 is not a one-variable ablation, so any improvement must later be
decomposed if attribution matters.

The source truth, 2:1 negative budget, 20% reviewed-distractor quota, KDK source
scales `[1.0, 0.66, 0.40]`, and validation policy remain unchanged.  The config
uses a distinct `run_name` but intentionally shares the same campaign root as v6
so the existing validated candidate/pool artifacts can be reused.  RF-DETR
trainer outputs therefore land beside v6 under:

```text
rounds/round0/runs/v7_multiscale_batch16_fullcosine/
```

ShitSpotter previously hardcoded `train_policy="fixed"` when generating the KDK
RF-DETR config.  The driver now maps an explicit `rfdetr.multi_scale` boolean to
KDK's train-policy contract: false -> `fixed`, true -> `multiscale`.  This keeps
the historical default unchanged while allowing campaign configs to request the
upstream internal multiscale path without hand-editing generated trainer files.

## 2026-09-27 12:10:00 -0400

Model: GPT-5.6 Sol.

Before launching v7, the batch-size/LR interaction was revisited.  Doubling the
per-GPU batch from 8 to 16 halves the number of optimizer updates per epoch, so
keeping the main model LR at `5e-5` would make the optimization regime more
conservative than intended in addition to the requested multiscale change.

The v7 main-model LR is therefore increased modestly by about sqrt(2):

```text
main model LR       7e-5   (v6: 5e-5)
encoder/backbone LR 1e-5   (unchanged)
```

The encoder LR intentionally remains at `1e-5`.  The user explicitly prefers a
more conservative encoder update rate rather than scaling it with the larger
batch.  This also avoids unnecessarily disturbing pretrained backbone features
while allowing the detector/segmentation heads to take somewhat larger steps.

This supersedes the immediately preceding journal note saying that v7 keeps
both v6 learning rates unchanged.  All other v7 settings remain unchanged:
internal RF-DETR multiscale is enabled, train and validation batch sizes are 16
per GPU, the full 15-epoch cosine schedule is allowed to run without early
stopping, and the v6 truth/pool/sampling policy is reused.
