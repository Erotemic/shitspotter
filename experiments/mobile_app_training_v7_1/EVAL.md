# v7.1 evaluation — TO BE FILLED IN AFTER RUNNING

## Hypothesis

v7's multi-scale gain (+0.012 AP over v6.0) stacks on top of v6.1's
+0.035 from the corrected source bundle, putting v7.1 within DETR
noise of v4.

## Headline numbers

| Cell        | v4 (kit eval) | v6.1  | **v7.1** | Δ v7.1 vs v4 | Δ v7.1 vs v6.1 |
|-------------|---------------|-------|----------|--------------|----------------|
| pico@416    | 0.4548        | 0.421 | TBD      | TBD          | TBD            |
| n@640       | 0.5553        | —     | TBD      | TBD          | —              |

## Decision

- [ ] **pico@416 within ±0.01 of v4** → kit pivot fully validated.
      Proceed to v9 distillation.
- [ ] **n@640 ≥ v4** → first kit cell to beat v4 cleanly. v10 ship.
- [ ] **Both within DETR noise but neither over** → still good enough
      to ship; v9 optional.
- [ ] **Regression vs v6.1** → multiscale isn't helping on the
      corrected bundle for some reason. Investigate.

## Provenance (will be auto-stamped)

`policy.json` and `detect_metrics.json` will both carry the embedded
provenance block. Expected SHAs at run time:
- kit: post-`059f60c` (or later)
- DEIMv2: `aeabc7e`
- OGDino: `9ddf1037`

## Run identity

- Recipe: `experiments/mobile_app_training_v7_1/recipe.yaml`
- Workspace: `/data/joncrall/kcd/v7_1/`
- Source: `/data/joncrall/kcd/v6_1/data/{train,vali}_tile_g2.kwcoco.zip`
- Test: v9 simplified test GT (same as every other vN)

## Notes
