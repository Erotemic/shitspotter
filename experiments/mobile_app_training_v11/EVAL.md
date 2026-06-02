# v11 evaluation — TO BE FILLED IN AFTER RUNNING

Two arms, both pico@640. Reference: v10 pico@416 = AP@0.5 **0.478**, Pixel 5
7 FPS / ~141 ms.

## Headline (overall AP@0.5)

| Model                | AP@0.5 | Δ vs v10 pico@416 | Δ vs v11 baseline |
|----------------------|--------|-------------------|-------------------|
| v10 pico@416         | 0.478  | —                 | —                 |
| v11 baseline (640)   | TBD    | TBD (= resolution)| —                 |
| v11 distill (640)    | TBD    | TBD               | TBD (= distillation) |

`Δ vs v10` on the baseline row = the **resolution** effect.
`Δ vs baseline` on the distill row = the **distillation** effect.

## Size-stratified AP (the actual hypothesis)

The whole point is small poops. Fill from `eval/<cell>/detect_metrics.json`
(`area_range=small|medium|large`).

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
| v11 baseline (640) | TBD              | TBD        | TBD         | TBD         |
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
