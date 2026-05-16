# v10 evaluation — TO BE FILLED IN AFTER RUNNING

This is the ship-cut row. Fill in `recipe.yaml` first (drawn from
v7-v9 winners), then run, then complete this file.

## Headline

| Cell        | v4 baseline | v9 distilled AP | v10 final AP | Δ vs v4 | Ship? |
|-------------|-------------|-----------------|--------------|---------|-------|
| pico@416    | 0.406       | TBD             | TBD          | TBD     | TBD   |
| n@640       | 0.520       | TBD             | TBD          | TBD     | TBD   |

## On-device numbers (Pixel 5)

| Cell        | Desktop ms | Pixel 5 ms | Pixel 5 FPS | Eligibility |
|-------------|-----------|------------|-------------|-------------|
| pico@416    | TBD       | TBD        | TBD         | TBD         |
| n@640       | TBD       | TBD        | TBD         | TBD         |

Pixel 5 numbers come from a separate device-benchmark pass after the
training run; the kit's `manifest --device_index <tsv>` consumes them.

## Ship artifacts

- [ ] ONNX file(s): `<paths>`
- [ ] Modelspec sidecar(s): `<paths>`
- [ ] Phone app `ModelRegistry` updated: PR/commit `<sha>`
- [ ] Repo tagged `mobile-v10-<cell>-<date>`: `<tag>`

## Notes
