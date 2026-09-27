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
