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
