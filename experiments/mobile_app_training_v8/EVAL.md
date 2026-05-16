# v8 evaluation — TO BE FILLED IN AFTER RUNNING

| Cell        | v7 AP | round 0 AP | round 1 AP | round 2 AP | Δ over v7 |
|-------------|-------|------------|------------|------------|-----------|
| pico@416    | TBD   | TBD        | TBD        | TBD        | TBD       |
| n@640       | TBD   | TBD        | TBD        | TBD        | TBD       |

| Cell        | hard-negs found round 0 | round 1 | round 2 |
|-------------|--------------------------|---------|---------|
| pico@416    | TBD                      | TBD     | TBD     |
| n@640       | TBD                      | TBD     | TBD     |

Round-over-round AP trajectory tells the story. Plateau by round 1 → 3
rounds is overkill; gain still climbing at round 2 → consider one more
round in v10.

## Decision

- [ ] +2 AP on at least one cell → advance to v9.
- [ ] No gain → log the AP plateau, but still advance (v9 distillation
      is independent and may help even when mining didn't).

## Notes
