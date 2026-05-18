# v7 evaluation — TO BE FILLED IN AFTER RUNNING

## How to fill this in (morning routine)

```bash
./reproduce/mobile_quality_push.sh compare
```

That prints a row per cell with v6 → v4 delta. v7's cells appear as
`v7:deimv2_pico@416` and `v7:deimv2_n@640`. Copy the AP values into
the table below.

Per-cell DEIMv2 internal val AP@0.5 trajectory is in the log file at
`/data/joncrall/kcd/v7/runs/<cell>/log.txt`. Grep for `test_coco_eval_bbox`
to see the per-epoch arc.

## Headline numbers

| Cell        | v4 fixed AP | v6 kit baseline | v7 multiscale AP | Δ vs v4 | Δ vs v6 | Verdict |
|-------------|-------------|-----------------|------------------|---------|---------|---------|
| pico@416    | 0.406       | 0.386           | TBD              | TBD     | TBD     | TBD     |
| n@640       | 0.520       | (not run in v6) | TBD              | TBD     | —       | TBD     |

| Cell        | Desktop ms (mean) | Eligibility class |
|-------------|-------------------|-------------------|
| pico@416    | TBD               | TBD               |
| n@640       | TBD               | TBD               |

## Decision

The bar isn't "match v4 exactly" — v6 settled the kit-pivot gap as
−0.020 AP (within DETR noise). For v7 we want the multiscale policy
to **beat v6's kit-baseline number** by at least ~+0.01 AP on each
cell, OR to leave the number unchanged with no regression.

- [ ] Both cells ≥ +0.01 AP over v6 → multiscale clearly helps, carry both into v8.
- [ ] One cell improves, one doesn't → carry the improved cell into v8, note the negative result for the other.
- [ ] Both regress vs v6 → multiscale is hurting, drop it; v8 uses v6's fixed policy.

## Run identity (fill after compare)

- Kit commit at run: `70c2270` round_loop init_checkpoint
- Workspace: `/data/joncrall/kcd/v7/`
- Manifest: `/data/joncrall/kcd/v7/manifest.tsv`

## Notes
