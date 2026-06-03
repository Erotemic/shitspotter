# v11 evaluation — TO BE FILLED IN AFTER RUNNING

Two arms, both pico@640. Reference: v10 pico@416 = AP@0.5 **0.478**, Pixel 5
7 FPS / ~141 ms.

## Headline (overall AP@0.5)

| Model                | AP@0.5 | Δ vs v10 pico@416 | Δ vs v11 baseline |
|----------------------|--------|-------------------|-------------------|
| v10 pico@416         | 0.478  | —                 | —                 |
| v11 baseline (640)   | **0.588** | **+0.110**     | —                 |
| v11 distill (640)    | TBD    | TBD               | TBD (= distillation) |

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

## Decision

- [ ] Resolution helps (baseline > v10 pico@416, esp. AP small) → 640 becomes
      the default pico input; update the app `ModelSpec` to a 640 entry.
- [ ] Distillation helps (distill > baseline by a meaningful margin) → fold
      teacher pseudo-GT into future recipes.
- [ ] Distillation is a wash/regression → log it; pseudo-GT recall does not
      transfer into pico's capacity. Stop pursuing it for this cell.
- [ ] Neither beats v10 → resolution is not the lever; pivot to tiled
      inference (roadmap #5).

## Ship artifacts (fill if a v11 cell ships)

- [ ] ONNX + modelspec paths
- [ ] Phone app `ModelSpec` (640 entry) + `push_models.sh` mapping
- [ ] Pixel 5 bench
- [ ] Repo tag

## Notes
