# Hard-negative and missing-truth review workflow

This document records the current ShitSpotter annotation-QA workflow for using a
trained detector to find both:

- **missed positive truth** (`poop` that should have been annotated), and
- **useful labeled hard negatives** (for example `leaf`, `trash`, `pinecone`,
  `rock`, or another nuisance class that the detector confuses with `poop`).

The central rule is:

> **Predictions are reusable; review classifications are disposable.**

If LabelMe truth changes but the source image pixels do not change, do **not**
rerun RF-DETR inference merely to refresh annotation QA. Rebuild the current
KWCoco truth and rerun the cheap prediction-vs-truth review against the existing
prediction KWCoco.

The second central rule is:

> **Canonical LabelMe sidecars are the source of truth. Review artifacts are
> disposable staging products.**

Do not edit `review.kwcoco.zip` as truth and do not copy model proposals into the
canonical dataset without human adjudication.

## Current artifacts

A typical local campaign uses paths like:

```bash
DATA=~/code/shitspotter/shitspotter_dvc
TRUE="$DATA/train.kwcoco.zip"

PRED=~/data/shitspotter_review/v3_best_ema_20260920/train_predictions.scale0p4.kwcoco.zip
REVIEW=~/data/shitspotter_review/v3_best_ema_20260920/review.scale0p4
```

`PRED` is the expensive, reusable model output. `TRUE` and `REVIEW` should be
considered refreshable whenever manual annotation changes.

## End-to-end loop

The normal iterative loop is:

```text
existing source-space prediction KWCoco
        |
        v
canonical LabelMe edits
        |
        v
gather + make_splits
        |
        v
prediction-review against CURRENT truth
        |
        v
transactional LabelMe review workspace
        |
        v
human relabel / correct / delete proposals
        |
        v
dry-run apply
        |
        v
commit apply to canonical LabelMe
        |
        v
gather + make_splits
        |
        v
repeat review with the SAME prediction KWCoco
```

Once annotation QA is satisfactory, rebuild any truth-dependent negative
candidate indexes / pools before using corrected nuisance annotations as hard
negative supervision.

## 1. Generate source-space predictions once

This is the expensive detector pass. For the current campaign, the coarse
source-space prediction file already exists:

```bash
PRED=~/data/shitspotter_review/v3_best_ema_20260920/train_predictions.scale0p4.kwcoco.zip
```

Do not regenerate it merely because truth annotations changed. The predictions
are tied to the source image pixels, not to the LabelMe truth.

Rerun inference only when something that affects model output changes, such as:

- source image pixels;
- model checkpoint;
- prediction resolution / tiling policy;
- inference threshold / backend behavior when that changes which detections are
  written.

## 2. Always rebuild KWCoco truth after canonical LabelMe edits

If any canonical LabelMe sidecar was edited, first refresh the dataset manifests:

```bash
cd ~/code/shitspotter

python -m shitspotter.gather
python -m shitspotter.make_splits
```

This step is important even if only 10--20 images changed. Any previous
prediction-review result for those images may now be stale.

The current training truth can then be referenced as:

```bash
TRUE=~/code/shitspotter/shitspotter_dvc/train.kwcoco.zip
```

Use the actual current `train.kwcoco.zip` produced by the local dataset checkout
if that checkout is elsewhere.

## 3. Reclassify the existing predictions against current truth

Run KDK `prediction-review` every time canonical truth changes:

```bash
TRUE=~/code/shitspotter/shitspotter_dvc/train.kwcoco.zip
PRED=~/data/shitspotter_review/v3_best_ema_20260920/train_predictions.scale0p4.kwcoco.zip
REVIEW=~/data/shitspotter_review/v3_best_ema_20260920/review.scale0p4

~/code/kwcoco_detector_kit/docker/rfdetr/kcd-rfdetr prediction-review \
    --true="$TRUE" \
    --pred="$PRED" \
    --dst-dpath="$REVIEW" \
    --target-categories=poop \
    --ignore-categories=ignore,unknown,unkown \
    --uncategorized-annotation-policy=ignore \
    --default-non-target-policy=background \
    --unclassified-category-policy=ignore \
    --min-score=0.5 \
    --top-n=20000
```

The review classifications are approximately:

- `matched_target`: prediction adequately explained by existing `poop` truth;
- `known_distractor`: prediction overlaps an existing named nuisance annotation;
- `uncertain_region`: prediction overlaps ignore / unknown truth;
- `overlapping_target`: prediction touches existing target truth but does not meet
  the normal target-match threshold;
- `unexplained_prediction`: prediction has **no positive-area overlap with any
  localized truth annotation**.

For missing-truth discovery, `unexplained_prediction` is intentionally
conservative. If a model proposal touches any localized annotation, it is not
put into the zero-overlap adjudication queue.

Bbox-less point annotations used as metadata do not localize a region and are
not treated as geometric overlap for this purpose.

## 4. Optional: inspect confusion visualizations before editing

KWCoco evaluation / review artifacts can carry per-annotation colors, and
`kwcoco visualize` should honor them. The convention used during this campaign
is:

- red: false-positive / unexplained model proposal;
- blue: prediction true positive;
- green: matched truth;
- purple: false-negative truth;
- yellow: ignore / uncertain truth;
- gray: nuisance / context truth.

This is useful for a quick visual scan, but it is not the canonical review UI.
The transactional LabelMe workspace below is preferred once there are enough
missed annotations or useful nuisance classes to justify actual edits.

## 5. Prepare a transactional LabelMe workspace

Never seed the next batch directly into canonical dataset directories. Prepare a
fresh isolated workspace:

```bash
TRUE=~/code/shitspotter/shitspotter_dvc/train.kwcoco.zip
REVIEW=~/data/shitspotter_review/v3_best_ema_20260920/review.scale0p4
WORKSPACE=~/data/shitspotter_review/v3_best_ema_20260920/labelme-review-top200-v3

cd ~/code/shitspotter

python -m shitspotter.labelme_review prepare \
    --review_dpath="$REVIEW" \
    --true="$TRUE" \
    --workspace="$WORKSPACE" \
    --top_images=200 \
    --min_score=0.5
```

Use a **new workspace path for every batch**. A review workspace is a transaction,
not a mutable long-lived mirror of the dataset.

`prepare` does two levels of stale protection:

1. it starts from KDK items classified as zero-overlap
   `unexplained_prediction`;
2. `--true` rechecks each candidate geometrically against the current KWCoco
   truth immediately before staging.

The second check is deliberately redundant. It prevents an old `review_queue`
from reseeding an image that was manually corrected after that queue was made.
The authoritative semantic classification still comes from rerunning
`prediction-review` against current truth.

### What is staged

For every admitted image the workspace contains:

- a copied source image;
- a copied canonical LabelMe sidecar, if one exists;
- all existing LabelMe shapes, including point-only metadata;
- one or more model proposals transformed back into LabelMe / EXIF-oriented
  coordinates;
- a transaction manifest containing canonical-path and SHA information.

A missing canonical sidecar is represented by a newly seeded LabelMe document.

Each model proposal starts with the reserved label:

```text
__review_proposal__
```

That label is intentionally invalid as accepted truth. It is a marker that human
adjudication is still required.

## 6. Review in normal LabelMe

Open the staged directory directly:

```bash
labelme "$WORKSPACE/items"
```

For **every** `__review_proposal__`, perform exactly one action.

### Missed positive

If the detector found real poop that was absent from truth:

1. change the proposal label to `poop`;
2. correct the polygon as needed.

Do not accept a bad detector mask merely because the detection itself is real.
The human-edited polygon becomes canonical truth.

### Useful false positive / labeled hard negative

If the model confused another object with poop, relabel the proposal to the
actual class, for example:

```text
leaf
trash
pinecone
rock
stick
...
```

These nuisance annotations are valuable. Under the current binary detector
policy they are background for the `poop` detector, but unlike anonymous
background they preserve *what* the model confused with the target.

### Ordinary background false positive

If the prediction is simply meaningless background and no persistent annotation
is useful, delete the proposal shape.

Deletion is an explicit rejection. It does not add anything to canonical truth.

### Ambiguous region

Do not force an uncertain object into a known nuisance category merely to clear
the queue. Use the dataset's uncertainty semantics (`unknown`, `ignore`, etc.)
when appropriate, or leave the item unresolved until it can be decided.

### Existing truth corrections

Because the canonical sidecar was copied into the workspace, existing shapes can
also be fixed while reviewing. Those changes are included in the transaction.

## 7. Check workspace status

At any point:

```bash
python -m shitspotter.labelme_review status \
    --workspace="$WORKSPACE"
```

Before apply, there should be no remaining `__review_proposal__` shapes.

An untouched proposal is **not** silently accepted. The apply step fails closed
while any reserved proposal label remains.

## 8. Dry-run the transaction

Always run apply once without `--commit`:

```bash
python -m shitspotter.labelme_review apply \
    --workspace="$WORKSPACE"
```

This validates the entire workspace before writing canonical files and produces
an `apply_plan.json`.

Important checks include:

- no unresolved proposal labels;
- canonical LabelMe files have not changed since workspace preparation;
- staged JSON is readable;
- the transaction manifest still matches the files being applied.

If the canonical dataset changed concurrently, do not force the transaction.
Prepare a fresh workspace against current truth instead.

## 9. Commit reviewed LabelMe files back to canonical truth

After inspecting the dry-run plan:

```bash
python -m shitspotter.labelme_review apply \
    --workspace="$WORKSPACE" \
    --commit=1
```

The apply implementation uses same-directory temporary files plus `os.replace`
for individual canonical sidecars and writes an `apply_receipt.json` so a
partially interrupted copy-back can be resumed safely.

After commit:

- accepted `poop` proposals are canonical positive truth;
- accepted nuisance labels are canonical labeled hard-negative truth;
- deleted proposals remain rejected;
- edits to pre-existing LabelMe shapes are canonical;
- temporary proposal metadata is removed.

## 10. Immediately rebuild KWCoco truth again

After committing the transaction:

```bash
cd ~/code/shitspotter

python -m shitspotter.gather
python -m shitspotter.make_splits
```

The old `review_queue.json` is now stale by definition.

## 11. Reuse the same model predictions and repeat

Do **not** rerun RF-DETR inference. Rerun Step 3 against the refreshed truth, then
prepare a fresh workspace for the next batch:

```bash
NEXT_WORKSPACE=~/data/shitspotter_review/v3_best_ema_20260920/labelme-review-top200-v4

python -m shitspotter.labelme_review prepare \
    --review_dpath="$REVIEW" \
    --true="$TRUE" \
    --workspace="$NEXT_WORKSPACE" \
    --top_images=200 \
    --min_score=0.5
```

This iterative structure is important. Corrections made in batch N naturally
remove or reclassify old proposals before batch N+1 is prepared.

In particular, if a user already corrected 10--20 images directly in the official
data directory before starting this transaction workflow, simply:

1. rerun `gather` and `make_splits`;
2. rerun `prediction-review` using the existing `PRED`;
3. prepare a new workspace against the refreshed `TRUE`.

There is no need to manually maintain a list of previously reviewed images.
Current geometric truth performs that filtering.

## 12. When reviewed nuisance labels become training hard negatives

Annotation QA and hard-negative admission are related but not identical.

A detector proposal labeled `leaf` or `trash` during review is now trustworthy
**source truth** saying that region is a nuisance rather than poop. It can inform
the next negative candidate universe, but existing truth-derived candidate
artifacts are stale after such corrections.

Before constructing a new training round, rebuild truth-dependent artifacts,
including as applicable:

- train negative candidate index;
- positive/negative tile pools;
- mining candidate manifests;
- any round manifests derived from old truth fingerprints.

Do not reuse an old candidate index simply because the source images are the same.
The safety classification of windows changed when truth changed.

This differs from the source-space prediction file: detector predictions remain
reusable because they depend on image pixels, whereas candidate admissibility
depends directly on truth.

## 13. Relationship to mining-stage `review_hard_negatives.py`

The older command:

```bash
python experiments/rfdetr_seg_v1/review_hard_negatives.py --round-index=0
```

reviews candidates produced by a completed **mining round**. It remains useful as
a truth-QA gate before mined negatives are admitted to round N+1.

The workflow in this document is a more direct dataset-wide annotation-QA loop:

```text
whole source dataset
    -> reusable source-space predictions
    -> prediction-review against current truth
    -> transactional LabelMe correction
```

Prefer the source-space loop when the immediate goal is to improve annotations
and discover systematic detector confusions. Use mining-stage review when
validating the actual negative examples selected by the training-data mining
pipeline.

They share the same principle:

> A hard negative is a hypothesis until a human or trusted truth explains it.

## 14. LabelMe / EXIF coordinate contract

ShitSpotter's canonical LabelMe annotations may live in EXIF-oriented annotation
coordinates, while KWCoco normalizes source geometry into its image canvas.

Therefore model polygons must **not** be copied directly from prediction KWCoco
into LabelMe JSON.

`shitspotter.labelme_review prepare` performs the inverse transform used by the
ShitSpotter ingest path before seeding proposals. The round-trip is tested by
converting both expected and reconstructed polygons to Shapely geometry and
checking symmetric-difference area within tolerance.

This is why the transactional workspace should be used instead of ad-hoc KWCoco
polygon copy/paste.

## 15. Point-only LabelMe metadata

Some ShitSpotter LabelMe sidecars contain point annotations used as metadata.
These may legitimately have no bounding box after conversion.

The review workflow should preserve these annotations. They are not localized
regions for zero-overlap filtering and should not cause visualization or review
code to assume every annotation has `bbox` geometry.

## 16. Scrubbed-image LabelMe sidecar hygiene

A historical data issue exists where a sidecar named like:

```text
foo.scrubbed.json
```

may still contain:

```json
"imagePath": "foo.jpg"
```

instead of referencing the actual sibling scrubbed image, for example:

```json
"imagePath": "foo.scrubbed.jpg"
```

This may not affect the KWCoco gather path, but it can make LabelMe itself fail to
open the sidecar correctly.

Treat this as a one-off data hygiene repair, not part of the permanent review
package. A safe cleanup should:

1. scan `*.scrubbed.json`;
2. find exactly one sibling image whose stem matches the JSON stem;
3. report mismatches in dry-run mode;
4. rewrite only `imagePath` when the match is unambiguous;
5. report, rather than guess, ambiguous or missing siblings;
6. rerun `python -m shitspotter.gather` afterward.

Changing only `imagePath` should not alter annotation geometry.

## 17. Failure and recovery rules

### Review queue seems to contain already-fixed images

Do not hand-maintain exclusions. Refresh current truth and rerun
`prediction-review`. `labelme_review prepare --true=...` also supplies a final
stale-overlap guard.

### Workspace apply says canonical files changed

Someone or something edited canonical LabelMe after workspace preparation. Do
not overwrite it. Rebuild current truth and prepare a fresh transaction.

### Many proposals remain unresolved

Continue reviewing. Do not bypass the reserved-label check.

### Prediction geometry looks wrong in LabelMe

Stop before apply. This indicates a coordinate-transform problem. Do not repair it
by blindly copying KWCoco coordinates into LabelMe.

### Source image pixels changed

The prediction KWCoco may now be stale. Unlike a truth-only edit, this is a reason
to rerun detector inference for the affected image set.

## 18. Compact operator checklist

For an ordinary review batch:

```bash
cd ~/code/shitspotter

# 1. Incorporate any direct canonical LabelMe edits.
python -m shitspotter.gather
python -m shitspotter.make_splits

# 2. Reclassify existing detector predictions against current truth.
TRUE=~/code/shitspotter/shitspotter_dvc/train.kwcoco.zip
PRED=~/data/shitspotter_review/v3_best_ema_20260920/train_predictions.scale0p4.kwcoco.zip
REVIEW=~/data/shitspotter_review/v3_best_ema_20260920/review.scale0p4

~/code/kwcoco_detector_kit/docker/rfdetr/kcd-rfdetr prediction-review \
    --true="$TRUE" \
    --pred="$PRED" \
    --dst-dpath="$REVIEW" \
    --target-categories=poop \
    --ignore-categories=ignore,unknown,unkown \
    --uncategorized-annotation-policy=ignore \
    --default-non-target-policy=background \
    --unclassified-category-policy=ignore \
    --min-score=0.5 \
    --top-n=20000

# 3. Prepare a fresh transaction.
WORKSPACE=~/data/shitspotter_review/v3_best_ema_20260920/labelme-review-next
python -m shitspotter.labelme_review prepare \
    --review_dpath="$REVIEW" \
    --true="$TRUE" \
    --workspace="$WORKSPACE" \
    --top_images=200 \
    --min_score=0.5

# 4. Human review.
labelme "$WORKSPACE/items"

# 5. Check and dry-run.
python -m shitspotter.labelme_review status --workspace="$WORKSPACE"
python -m shitspotter.labelme_review apply --workspace="$WORKSPACE"

# 6. Commit reviewed sidecars.
python -m shitspotter.labelme_review apply \
    --workspace="$WORKSPACE" \
    --commit=1

# 7. Rebuild truth. Then go back to step 2 for another batch.
python -m shitspotter.gather
python -m shitspotter.make_splits
```

Before starting a new hard-negative training round, additionally rebuild all
truth-dependent candidate indexes and derived pools.
