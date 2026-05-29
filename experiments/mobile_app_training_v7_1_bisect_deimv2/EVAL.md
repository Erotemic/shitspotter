# DEIMv2 bisect — TO BE FILLED IN AFTER RUNNING

## Setup

- Recipe: `experiments/mobile_app_training_v7_1_bisect_deimv2/recipe.yaml`
- Kit commit at run: `a8ca45e` (the temporary submodule-rollback commit)
- **DEIMv2: `377e10a`** (v4's SHA, vs v7.1's `aeabc7e`)
- OGDino: `9ddf1037` (unchanged)
- Source bundle: v6.1's `/data/joncrall/kcd/v6_1/data/` (10 671-image
  base, ~53 K tiles after quadrant g2)

## Headline result

| Cell     | v4 (kit eval) | v7.1 (aeabc7e) | **bisect (377e10a)** | Δ bisect vs v7.1 | Δ bisect vs v4 |
|----------|---------------|----------------|----------------------|------------------|----------------|
| pico@416 | 0.4548        | 0.4329         | TBD                  | TBD              | TBD            |

Per the decision matrix in [README.md](README.md):

- [ ] **≥ 0.45**: DEIMv2 bump caused the residual.
- [ ] **~0.45 (Δ ≥ +0.015 vs v7.1)**: partially caused it.
- [ ] **~0.43 ± 0.01 (no movement)**: bump isn't the cause; look elsewhere.
- [ ] **< 0.42**: rollback hurt; unexpected.

## What this answers + what's next

(Fill in after the run.)

## Cleanup

After this file is filled in, restore `tpl/DEIMv2` to `aeabc7e` in
kit main so the sealions line isn't blocked. Don't leave kit commit
`a8ca45e` as the head state.
