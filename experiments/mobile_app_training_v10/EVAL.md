# v10 evaluation — completed 2026-06-01

Run finished 2026-06-01 10:38 UTC. Numbers from
`/media/joncrall/flash1/kcd-ssd/v10/manifest.json`. v9 distillation was
**not** used — v10 init from `deimv2_*_coco.pth` (distillation deferred; see
recipe.yaml "Pending integrations").

## Headline

| Cell        | v4 baseline | v9 distilled AP | v10 final AP | Δ vs v4 | Ship? |
|-------------|-------------|-----------------|--------------|---------|-------|
| pico@416    | 0.406       | n/a (not used)  | 0.478        | +0.072  | YES (clears +5 AP bar) |
| n@640       | 0.520       | n/a (not used)  | 0.511        | -0.009  | no (below v4) |

`v10 final AP` = `test_ap` (AP@0.5, kwcoco detect_metrics).

## On-device numbers (Pixel 5)

| Cell        | Desktop ms (p50) | Pixel 5 ms | Pixel 5 FPS | Eligibility       |
|-------------|------------------|------------|-------------|-------------------|
| pico@416    | 14.1             | PENDING    | PENDING     | HOST_PROMISING    |
| n@640       | 39.7             | PENDING    | PENDING     | HOST_PROMISING    |

Pixel 5 numbers come from a separate device-benchmark pass after the
training run; the kit's `manifest --device_index <tsv>` consumes them.
**Not yet run** — `device_eligible: TODO` in the manifest. This is the
remaining gate before pico@416 is ship-ready.

## Ship artifacts (pico@416 — the ship candidate)

- [x] ONNX: `/media/joncrall/flash1/kcd-ssd/v10/runs/deimv2_pico_416x416_multiscale_320_512/export/deimv2_h416_w416.onnx`
- [x] Modelspec sidecar: `.../export/deimv2_h416_w416.modelspec.json`
- [x] Phone app `ModelRegistry` updated: `scatspotter_app` commit `37ca5b9`
      (`DEIMV2_PICO_416_V10`, sideload as `deimv2_pico_h416_w416_v10.onnx`)
- [ ] Pixel 5 on-device bench run (fills `fpsHint` + manifest device cols)
- [ ] Repo tagged `mobile-v10-pico-<date>`: `<tag>`

## Notes
