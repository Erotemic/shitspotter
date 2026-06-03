# Evaluation Roadmap & Test Registry

A living registry of **evaluations we have run** and **evaluations we still
want to run** for the mobile detector models. The goal is to never lose track
of an intended test, to evaluate models *holistically* (not just headline AP),
and to keep enough artifacts around that we can re-run analyses post hoc —
especially when the test set is later updated with more representative data.

This complements, but is distinct from:
- [RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md) — narrative log of what happened.
- Per-run `mobile_app_training_vN/EVAL.md` — the scorecard for a single run.
- This file — forward-looking backlog + cross-run test matrix + retention policy.

## Evaluation philosophy — beyond AP

AP@0.5 is the headline ship gate, but it hides where a model actually
succeeds or fails. For any model we care about shipping, we want:

- **Stratified by object size** — COCO small / medium / large area ranges.
  Hypothesis: aggressive input downscaling hurts *small* poops most.
- **Stratified by text label** — the human-attached descriptive labels
  (surface, lighting, occlusion, poop type, etc.) so we can see *which kinds*
  of scenes the model struggles on, not just an average.
- **Operating-point precision/recall**, not only area-under-curve — the phone
  app runs at a fixed score threshold, so the precision/recall *at that
  threshold* is what users feel.
- **Latency / FPS on the real device** (Pixel 5), not just desktop CPU.
- **Qualitative failure gallery** — saved hard cases for eyeballing.

Status legend: ✅ done · 🔄 in progress · ⬜ planned · ❄️ blocked (see note)

---

## Completed evaluations

| Date | Eval | Model(s) | Result | Artifacts |
|------|------|----------|--------|-----------|
| 2026-06-02 | v10 pico Pixel 5 on-device bench (NNAPI) | pico@416 (v10) | 7 FPS, ~141 ms/frame (110 inf + 30 pre + 0.7 post) → device-eligible, **ship-ready** | app `fpsHint`; v10 EVAL.md device table |
| 2026-06-01 | v10 training + desktop AP + desktop latency | pico@416, n@640 | pico 0.478 (+0.072 vs v4), n 0.511 (−0.009 vs v4); both desktop-eligible | `/media/joncrall/flash1/kcd-ssd/v10/manifest.{tsv,json}`, per-cell `runs/<cell>/` + `eval/<cell>/` |
| 2026-05-14 | v4 Pixel 5 on-device bench (NNAPI + CPU EP) | v4 pico/n cells | n@640 2.5 FPS, pico@416 7.1 FPS (see CHANGELOG) | v4 manifest `/data/joncrall/shitspotter_v4/manifest.tsv` |

> Note: v10's `test_ap` is AP@0.5 from kwcoco `detect_metrics`. The saved
> `eval/<cell>/detect_metrics.json` already contains per-area-range breakdowns,
> so some size-stratified analysis below is derivable *without re-inference*.

---

## Planned evaluations (backlog)

Each item: what · why · prerequisite · re-run cost.

### 1. ✅ Pixel 5 on-device bench of the v10 best **pico** — DONE 2026-06-02
- **Result:** 7 FPS NNAPI, ~141 ms/frame (110 ms inference + 30 ms preprocess
  + 0.7 ms postprocess) — 7× the 1 FPS floor, **device gate passed → pico@416
  ship-ready**. Preprocess (~21%) is the main remaining latency lever.
- Closed the `device_eligible: TODO` gate; device table in
  [mobile_app_training_v10/EVAL.md](mobile_app_training_v10/EVAL.md) and the
  app `fpsHint` updated to the measured value.
- **Still open:** n@640 device bench (not a ship candidate, lower priority).

### 2. 🔄 Size-stratified accuracy (small / medium / large) — tool ready
- **Why:** quantify the small-poop hypothesis; "good overall AP" can hide a
  collapse on small objects.
- **Status:** the kit eval only emits `area_range=all`, so `detect_metrics.json`
  has no per-size breakdown. `experiments/size_stratified_eval.py` re-scores the
  saved `*_bbox_only.kwcoco.zip` predictions with COCO small/med/large (no
  re-inference) — runs in the host kwcoco env. **Not yet run** on v10 vs v11.
- **Cost:** post-hoc on existing artifacts (cheap).

### 3. ⬜ Text-label-stratified accuracy
- **Why:** find *which scene types* fail (surface/lighting/occlusion/type),
  using the human-attached text labels.
- **Prereq:** the text labels must be joined into the test kwcoco as
  categories/attributes so eval can group by them. Confirm where the labels
  live and whether they're already on the test images.
- **Cost:** post-hoc re-scoring of saved predictions *if* labels are on the
  same test images; otherwise needs a join step (no re-inference).

### 4. 🔄 pico input-resolution sweep (416 vs 512 vs 640) — v11 scaffolded
- **Status:** [mobile_app_training_v11](mobile_app_training_v11/README.md) sets
  up pico@640 as two isolated arms — `baseline` (640, human GT) to measure the
  resolution gain over v10 pico@416, and `distill` (640 + OGDino teacher
  pseudo-GT) to measure distillation's marginal value. Size-stratified eval
  (this is test #2) is the primary readout. Not yet run.
- **Why:** pico's *native* DEIMv2 res is 640; v10 ran it downscaled to 416.
  Test whether higher resolution recovers small-poop AP. Big latency headroom
  (pico@416 = 14 ms vs 80 ms budget; est. ~21 ms @512, ~34 ms @640).
- **Note:** "pico" is an **architecture** size (HGNetv2 `Pico` + `LiteEncoder`),
  independent of input resolution — so raising resolution is a separate knob
  from going to the `n` architecture.
- **Cost:** retrain per resolution (or at least re-export/re-eval).

### 5. ⬜ Tiled inference vs higher native resolution — **key open axis**
- **Why:** two competing ways to see small poops on-device:
  - (a) run pico at a **higher single-pass resolution**, vs
  - (b) keep pico at its native res but **crop the image into tiles at
    inference time** and run per-tile, stitching detections.
  Unknown which wins on the accuracy/latency/battery tradeoff *on the phone*.
- **Track:** accuracy (incl. size-stratified) AND device latency/FPS for each
  option; tiling multiplies inference passes, so the FPS cost is real.
- **Prereq:** a tiled-inference path in the phone app + a desktop harness that
  mirrors it so results are comparable.
- **Cost:** new inference code path + device bench.

### 6. ⬜ Holistic metric panel
- **Why:** move past single-number AP. PR curves, precision/recall at the app's
  deploy threshold, score calibration, and a saved failure-case gallery.
- **Cost:** post-hoc on saved predictions.

### 7. ⬜ Re-evaluate frozen models on the updated test set
- **Why:** when the test set is refreshed with more representative data, every
  shipped/candidate model should be re-scored on it for an apples-to-apples
  comparison over time.
- **Prereq:** retained model artifacts (see retention policy) so we can
  re-predict on the new test images.
- **Cost:** re-inference per model on the new test set (needs ONNX/checkpoint).

---

## Artifact retention policy

We keep training/eval artifacts so the tests above can be run *later*, possibly
against test sets that don't exist yet. There are two distinct post-hoc modes,
and they need different artifacts:

| Post-hoc mode | What you need retained |
|---------------|------------------------|
| New **metric** on the **same** test set (size/label strat, PR curves, recalibration) | Saved predictions + ground truth: `eval/<cell>/pred_boxes.kwcoco.zip`, `true_bbox_only.kwcoco.zip` |
| Updated / **new test set** (re-score on more representative data) | The model itself: `runs/<cell>/export/*.onnx` + `*.modelspec.json` **and/or** the checkpoint `best_stg2.pth`, plus the generated config |

So: **keep both** the saved predictions *and* the model artifacts. Minimum
keep-set per run:

- `runs/<cell>/export/*.onnx` + `*.modelspec.json` (deployable model)
- `runs/<cell>/best_stg2.pth` (and `best_stg1.pth`) (re-export / re-predict)
- `runs/<cell>/generated_configs/` + `policy.json` (provenance: kit + DEIMv2
  git SHAs, all hyperparameters — needed to reproduce/interpret)
- `runs/<cell>/log.txt` + `summary/events.out.tfevents.*` (training curves)
- `eval/<cell>/*.kwcoco.zip` + `detect_metrics.json` (post-hoc metric re-derivation)
- `manifest.{tsv,json}` (the run's eligibility summary)

Rules:
- **Do not delete run directories** under `kcd_root` after a run completes.
- v10 artifacts currently live on the SSD at
  `/media/joncrall/flash1/kcd-ssd/v10/` — this is fast scratch, **not a backup**.
  Ship-candidate models should additionally be copied/registered somewhere
  durable before the SSD is reclaimed.
- Each run's `policy.json` records the exact `kit_sha` / `deimv2_sha`; keep it
  so a model can be reproduced even after the code moves on.

---

## Open questions

- **Resolution vs tiling** (backlog #4 vs #5): unresolved which is the better
  small-poop lever on-device. This is the headline experimental question.
- Does the v9 distillation init (still pending) beat `_coco.pth` as a pico
  starting point? (Not an eval per se, but the next training lever.)
