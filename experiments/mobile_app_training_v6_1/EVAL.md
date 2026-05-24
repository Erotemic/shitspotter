# v6.1 evaluation — TO BE FILLED IN AFTER RUNNING

## Hypothesis

The −0.057 AP gap on pico@416 between every kit run (v6/v7/v8) and v4's
corrected baseline (0.4548, kit-eval) is caused by v6/v7/v8 training on
the wrong source bundle (`simplified_train_imgs7350` with 2 564 images
vs v4's `train_imgs10671` with 10 671 images). Fixing the source bundle
should close most or all of the gap.

## Headline numbers

| Metric                | v4 (kit eval) | v6.0 | **v6.1** | Δ v6.1 vs v4 | Δ v6.1 vs v6.0 |
|-----------------------|---------------|------|----------|--------------|----------------|
| Test AP @ IoU=0.5     | 0.4548        | 0.386 (kit) | TBD | TBD          | TBD            |
| Desktop CPU mean (ms) | —             | 11.1 / 13.7 | TBD | —            | TBD            |
| Train wall-clock      | —             | ~3 h        | TBD | —            | —              |
| Train tiles produced  | 53 355 (v4)   | 12 820 (v6.0) | TBD | should be ~53k | ~4×            |

## Decision

- [ ] **v6.1 within ±0.01 of v4's 0.4548** → hypothesis confirmed.
      Kit pivot truly validated. Proceed to **v7.1** (multiscale +
      corrected bundle, both cells).
- [ ] **v6.1 in [0.42, 0.44]** → most of the gap is the data bundle,
      but ~0.02 residual to investigate. Likely a remaining detail
      (tile boundary handling differs, JPEG quality, etc.). Still
      proceed to v7.1, but flag the residual.
- [ ] **v6.1 ≤ v6.0 (0.386) or no change** → bundle isn't the cause.
      Bigger investigation: diff the v4 vs kit tile outputs image by
      image, look at JPEG quality, etc.

## Run identity

- Recipe: `experiments/mobile_app_training_v6_1/recipe.yaml`
- Kit commit at run: `<fill after build>`
- Workspace: `/data/joncrall/kcd/v6_1/`
- Manifest: `/data/joncrall/kcd/v6_1/manifest.tsv`

## Notes
