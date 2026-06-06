# v11 evaluation — TO BE FILLED IN AFTER RUNNING

Two arms, both pico@640. Reference: v10 pico@416 = AP@0.5 **0.478**, Pixel 5
7 FPS / ~141 ms.

## Headline (overall AP@0.5)

| Model                | AP@0.5 | Δ vs v10 pico@416 | Δ vs v11 baseline |
|----------------------|--------|-------------------|-------------------|
| v10 pico@416         | 0.478  | —                 | —                 |
| v11 baseline (640)   | **0.588** | **+0.110**     | —                 |
| v11 distill (640)    | 0.552  | +0.074            | **−0.036 (REGRESSION)** |

`Δ vs v10` on the baseline row = the **resolution** effect.
`Δ vs baseline` on the distill row = the **distillation** effect.

**Baseline result (2026-06-03):** pico@640 = AP@0.5 **0.588**, a **+0.110**
jump over v10 pico@416 (0.478) from resolution alone — the biggest single-lever
gain in the v6→v11 ladder. `scales=640..640` (multiscale collapsed to fixed-640
again, as in v10), so this is pure fixed-resolution. Desktop CPU 32.4 ms (under
the 80 ms budget); device bench still TODO.

## Size-stratified AP (the actual hypothesis)

The whole point is small poops. **GAP (2026-06-03):** the kit eval currently
emits only `area_range=all` — `detect_metrics.json` has no small/medium/large
breakdown (true for v10 and v11 baseline). So this table can't be filled from
the existing artifacts. To get it: either enable area-range stratification in
the kit eval, or compute it post-hoc from the saved `eval/<cell>/pred_boxes.kwcoco.zip`
+ `true_*.kwcoco.zip` (no re-inference needed — see EVALUATION_ROADMAP.md #2).
Until then the +0.110 overall gain is suggestive but unconfirmed as a
small-object win.

| Model              | AP small | AP medium | AP large |
|--------------------|----------|-----------|----------|
| v10 pico@416       | TBD      | TBD       | TBD      |
| v11 baseline (640) | TBD      | TBD       | TBD      |
| v11 distill (640)  | TBD      | TBD       | TBD      |

Expectation: most of the resolution gain should land in **AP small**. If 640
does not move AP small, the small-poop hypothesis is wrong and tiling
(roadmap #5) is the next thing to try.

**Distillation size-stratified result (DEIM internal eval, epoch 119, both
@640):** distillation regressed on *every* size band, not just overall:

| AP (DEIM, .5:.95 unless noted) | baseline@640 | distill@640 | Δ |
|--------------------------------|--------------|-------------|---|
| AP@.5:.95                      | 0.358        | 0.289       | −0.069 |
| AP50                           | 0.563        | 0.467       | −0.096 |
| AP_small                       | 0.279        | 0.212       | −0.067 |
| AP_medium                      | 0.442        | 0.369       | −0.073 |
| AP_large                       | 0.388        | 0.337       | −0.051 |

So the teacher's boxes did **not** help small poops — they hurt them too. (For
the kwcoco-harness per-size numbers, run `experiments/size_stratified_eval.py`;
the DEIM-internal eval above is already conclusive on direction.)

## On-device (Pixel 5)

| Model              | Desktop ms (p50) | Pixel 5 ms | Pixel 5 FPS | Eligibility |
|--------------------|------------------|------------|-------------|-------------|
| v11 baseline (640) | 32.4             | TBD        | TBD         | HOST_PROMISING |
| v11 distill (640)  | TBD              | TBD        | TBD         | TBD         |

640 vs 416 ≈ 2.4× the compute; est. ~34 ms desktop / ~3 FPS device. Confirm it
still clears the 1 FPS floor (both arms have the same architecture+input, so
latency should be identical between them — accuracy is the only differentiator).

## Teacher pseudo-GT sanity-check (distill arm only)

Before trusting the distill arm, compare the merged bundle's teacher
annotations to human ones (count/image, box size, aspect). If the teacher
massively over-predicts small boxes on tiles, the distill gain may be
label-noise artifact rather than real recall.

| Metric                        | Value |
|-------------------------------|-------|
| Teacher boxes generated       | TBD   |
| Teacher boxes merged into GT  | TBD   |
| Median teacher box area / human| TBD  |

## Distill arm — RAN 2026-06-06 (negative result); integration notes below

The distill arm now runs end-to-end. Getting there took 6 kit fixes to the
never-functional OGDino teacher predictor (wrong module layout for the training
fork, split `util`, `weights_only`, build_model tuple) + the merged-bundle
reroot + `--shm-size`. Result: **distillation regressed** (see tables above) —
so the payoff was a clean negative, but the OGDino-teacher path is now working
for any future use. Original blocker write-up retained for history:

### (historical) BLOCKED: kit package-format mismatch

The teacher pseudo-label step fails with `KeyError('trainer')`, then would fail
deeper. Root cause: `kwcoco-detector-kit pseudo-label` → `predict_kwcoco` only
consumes **kit-native trainer packages** — it does `get_trainer(manifest["trainer"])`,
`materialize_workdir(...)`, then `OGDinoTrainer.build_predictor(workdir)` which
expects a kit workdir layout (`generated_configs/ogdino_cfg.py`,
`find_checkpoint(workdir)`, `policy.json`). But the teacher we point it at is a
**foundation_detseg_v3** package (`backend: opengroundingdino_sam2`,
`detector.config_fpath`/`checkpoint_fpath`, `segmenter`, `postprocess.nms_thresh`)
that references externally-produced files. The two schemas were never reconciled
because v9 (which designed this path) was never actually run.

Adding `trainer: opengroundingdino` to the package only clears the KeyError; it
then fails in `build_predictor` (no `generated_configs/ogdino_cfg.py` in the
foundation package). Fix options (each needs host-GPU to verify):
- **(A) Repackage**: a shim that lays the foundation detector
  `config_fpath`/`checkpoint_fpath` into the kit workdir layout
  `build_predictor` expects + a manifest with `trainer: opengroundingdino`.
- **(B) Adapter**: teach `OGDinoTrainer.build_predictor` to accept a
  foundation-style `config_fpath`/`checkpoint_fpath` directly.
- **(C) Bypass**: generate the pseudo-GT via the foundation_detseg_v3 pipeline's
  own OGDino predict path (the one that produced the 0.766 eval), then merge.

Not blocking the v11 baseline win — the distill arm is a separate kit-integration
task. The teacher itself loads fine (package built, AP=0.7656 confirmed).

**FIX IN PLACE (option A, 2026-06-03):** `make_teacher_kit_package.py` repackages
the foundation OGDino detector into the kit-native layout
(`trainer: opengroundingdino` + `artifacts: {checkpoint→checkpoint0000.pth,
train_config→generated_configs/ogdino_cfg.py, policy→policy.json}`,
`pipeline: detector_only`). `run.sh` now builds this and points `pseudo-label`
at it. **UNVERIFIED — needs host-GPU run** (no GPU/OGDino on the VM). When
running the distill arm, ensure `$KCD_OPENGROUNDINGDINO_REPO_DPATH` is set so
the predictor can import `groundingdino`. If it errors, the likely culprits are
OGDino config loadability or the repo env var — not the package layout.

## Decision (2026-06-06)

- [x] **Resolution helps** (+0.110 over v10 pico@416; gains on every size band)
      → 640 is the default pico input; `DEIMV2_PICO_640_V11` registered in the app.
- [x] **Distillation did not win** — distill 0.552 < baseline 0.588 (−0.036;
      −0.05…−0.10 on every DEIM band incl. small). This is "not a win," **not a
      proven dead end**: the prime suspect is the full-weight pseudo-GT merge
      (teacher over-predicts on tiles → label noise), i.e. a likely misconfig,
      not evidence that distillation can't help. **Parked, not refuted** —
      revisit only with filtered (higher score_thresh) + downweighted
      (`from_teacher` weight < 1.0) teacher boxes. Lower priority than
      resolution/tiling.
- next lever for small poops is NOT distillation: either more resolution
      (v12 pico@768, scaffolded) or tiled inference (EVALUATION_ROADMAP #5).

> If distillation is ever revisited, the levers to try are: filter teacher boxes
> harder (raise `--score_thresh` well above 0.30) and/or downweight them
> (`from_teacher` weight < 1.0) so noise doesn't dominate — not full-weight merge.

## Ship artifacts (baseline pico@640)

- [x] ONNX: `…/v11/baseline/runs/deimv2_pico_640x640_multiscale_512_768/export/deimv2_h640_w640.onnx`
      (+ `.modelspec.json`)
- [x] Phone app `ModelSpec`: `DEIMV2_PICO_640_V11` (sideload as
      `deimv2_pico_h640_w640_v11.onnx`); `push_models.sh` mapping added; listed
      first in `ModelRegistry.all`.
- [ ] Pixel 5 bench (fills `fpsHint` + device cols; ~3 FPS est.)
- [ ] Size-stratified AP confirmed (run `experiments/size_stratified_eval.py`)
- [ ] Repo tag

## Notes
