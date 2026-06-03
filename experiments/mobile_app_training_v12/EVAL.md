# v12 evaluation — TO BE FILLED IN AFTER RUNNING

Pure resolution push: pico@768 vs the resolution ladder.

## Headline (AP@0.5)

| Model               | AP@0.5 | Δ vs 640 | Desktop ms (p50) | Pixel 5 FPS | Eligibility |
|---------------------|--------|----------|------------------|-------------|-------------|
| v10 pico@416        | 0.478  | —        | 14.1             | 7 (NNAPI)   | shipped-prev |
| v11 baseline @640   | 0.588  | —        | 32.4             | ~3 (est)    | HOST_PROMISING |
| v12 pico@768        | TBD    | TBD      | TBD (~47 est)    | TBD (~2 est)| TBD         |

## Size-stratified AP (the actual hypothesis)

Run `python experiments/size_stratified_eval.py --candidate v11_640=/media/joncrall/flash1/kcd-ssd/v11/baseline/eval/deimv2_pico_640x640_multiscale_512_768 --candidate v12_768=/media/joncrall/flash1/kcd-ssd/v12/eval/deimv2_pico_768x768_multiscale_640_896`

| Model        | AP small | AP medium | AP large |
|--------------|----------|-----------|----------|
| v11 @640     | TBD      | TBD       | TBD      |
| v12 @768     | TBD      | TBD       | TBD      |

Expectation: if resolution is still the lever, **AP small** rises again. If it
plateaus, single-pass resolution has saturated for these tiles → pivot to
tiled inference (roadmap #5).

## Decision

- [ ] 768 > 640 (esp. AP small) and still under 80 ms desktop / above 1 FPS
      device → new ship candidate; register in app, Pixel 5 bench.
- [ ] 768 ≈ 640 or busts latency → 640 stays the ship model; stop pushing
      single-pass resolution, try tiling.

## Notes
