# v9 evaluation — TO BE FILLED IN AFTER RUNNING

| Cell        | v8 AP | v9 distilled AP | Δ vs v8 | Δ vs v4 |
|-------------|-------|-----------------|---------|---------|
| pico@416    | TBD   | TBD             | TBD     | TBD     |
| n@640       | TBD   | TBD             | TBD     | TBD     |

| Teacher artifact            | Count          |
|-----------------------------|----------------|
| Teacher boxes generated     | TBD            |
| Teacher boxes merged into GT| TBD            |
| Anns dropped (score < 0.3)  | TBD            |

## Decision

- [ ] +3 AP on n@640 → carry distillation forward into v10.
- [ ] +2 AP on pico@416 → carry forward into v10.
- [ ] Neither → log it; the additive of mining + multiscale alone may
      already be the ceiling for these capacities.

## Teacher quality sanity-check

If the merged training kwcoco's teacher-annotation distribution looks
different from human-annotation distribution (much smaller boxes, very
different aspect ratio, much higher per-image count), the teacher may
be over-predicting on tiles. Document it here.

## Notes
