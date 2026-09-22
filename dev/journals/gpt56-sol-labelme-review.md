## 2026-09-22 14:57:00 -0400

Model: GPT-5.6 Sol; reasoning configuration: system-selected hidden reasoning.

The user reviewed the highest-confidence zero-overlap RF-DETR findings and found enough missing `poop` truth and semantically useful false positives (`leaf`, `trash`, and similar nuisance classes) that one-off navigation to canonical LabelMe sidecars is no longer efficient. The desired workflow is transactional: stage selected images and current LabelMe truth in an isolated directory, seed model proposals there, review with ordinary LabelMe, then explicitly validate and copy reviewed sidecars back to canonical source locations.

I kept this behavior in ShitSpotter rather than KDK. KDK owns ranked source-space prediction/truth comparison; ShitSpotter owns the canonical LabelMe sidecars, EXIF-oriented annotation coordinates, and dataset-specific copy-back semantics. Each model proposal is written with a reserved `__review_proposal__` label. Human acceptance must be explicit: relabel to a real class and optionally refine geometry; deleting the proposal rejects it. Apply refuses any unresolved proposal and detects concurrent canonical-sidecar edits using the SHA256 captured when the workspace was prepared.

A critical coordinate-space constraint is that `shitspotter.gather.load_labelme_anns()` maps LabelMe's EXIF-oriented polygons into raw-image KWCoco coordinates. Workspace preparation therefore applies the inverse of that exact transform to source-space detector proposals before writing them into copied LabelMe files. Existing sidecars are copied rather than reconstructed, preserving point-only metadata and other legacy LabelMe fields. The rank is embedded in staged filenames so LabelMe's file browser naturally walks the review set in model-confidence order.

The copy-back implementation validates the whole workspace before its first canonical write, preserves JSON file modes, writes each sidecar through a same-directory temporary file plus `os.replace`, and records an incremental receipt. Re-running after a successful or partially successful apply recognizes files already written by that same receipt, making copy-back idempotent/resumable while still refusing unrelated concurrent edits. Proposal identity is recognized from both the temporary group id and an embedded description marker; an untouched reserved label remains unresolved even if an editor drops metadata.

Validation in the stripped artifact container was necessarily split. `py_compile` passes for the implementation and tests. The container lacks `scriptconfig`, `kwcoco`, and `kwimage`, so the full pytest geometry fixture cannot run there. I independently exercised the dependency-free transaction engine with a stubbed CLI module: unresolved refusal, metadata-loss safety, accept/relabel cleanup, dry-run non-mutation, commit, idempotent second apply, deletion-as-rejection, and concurrent canonical edit refusal all passed. The EXIF transform was also compared directly against the current `gather.py` implementation and mirrors its orientation cases exactly. The focused repository test remains the acceptance check in a normal ShitSpotter development environment.

## Current-truth refresh rule

Prediction artifacts are intentionally reusable across truth edits. Review
classification is not: after canonical LabelMe edits, rebuild KWCoco truth and
rerun `prediction-review` against the existing prediction KWCoco. The LabelMe
workspace `prepare` command now also accepts `--true` and drops stale proposals
that have acquired any positive-area overlap with localized current truth. This
extra check ignores bbox-less point metadata and exists as a final safety net,
not as a replacement for KDK's full truth-semantics classification.

## 2026-09-22 16:22:00 -0400

Model: GPT-5.6 Sol; reasoning configuration: system-selected hidden reasoning.

The focused transactional LabelMe tests exposed five identical `KeyError: source_fpath` failures. This is a synthetic-fixture mismatch, not a production queue defect: KDK `prediction-review` writes `source_fpath`, `source_image`, `labelme_json`, `prediction_bbox_xyxy`, and `prediction_has_segmentation` for every ranked item, but the local `_write_review()` test fixture only populated a reduced subset. I updated the fixture to mirror the actual KDK queue contract rather than weakening current-truth revalidation, because path identity is the deliberate bridge when regenerated truth may not preserve image ids. The production behavior remains unchanged: predictions are reusable, review classification should be regenerated against current truth, and workspace staging additionally rejects proposals that now overlap localized truth.

Validation here is limited by the artifact container: syntax compilation succeeds; pytest cannot collect because the container lacks `pytest-xdoctest` and then `scriptconfig` when repository addopts are disabled. The user's normal development environment remains the acceptance environment for the focused test file.
