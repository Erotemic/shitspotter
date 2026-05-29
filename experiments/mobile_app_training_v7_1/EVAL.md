# v7.1 evaluation — multiscale on corrected bundle

## Headline numbers

| Cell        | v4 (kit eval) | v6.1  | **v7.1**  | v7 (wrong bundle) | Δ v7.1 vs v4 | Δ v7.1 vs v6.1 |
|-------------|---------------|-------|-----------|-------------------|--------------|----------------|
| pico@416    | 0.4548        | 0.421 | **0.4329**| 0.398             | **−0.022**   | **+0.012**     |
| n@640       | 0.5553        | —     | **0.5350**| 0.535             | **−0.020**   | first measurement |

| Cell        | Desktop CPU mean (ms) | Eligibility class |
|-------------|-----------------------|-------------------|
| pico@416    | 15.1                  | HOST_PROMISING    |
| n@640       | 47.1                  | HOST_PROMISING    |

## Verdict

- [x] **Multiscale gain stacks as predicted on pico (+0.012).** v6.1
      added +0.035 from the corrected bundle; v7.1 adds another +0.012
      from multiscale. The two gains are cleanly additive.
- [ ] Within ±0.01 of v4 — no, still −0.020 to −0.022 on both cells.
      The remaining residual is real and isn't noise.

## The interesting finding: n@640 didn't benefit from the corrected bundle

v7's n@640 with the wrong bundle was **0.535**. v7.1's n@640 with the
corrected bundle is also **0.5350**. Identical to four decimals.

For n@640 specifically, the source-bundle fix made **no difference**.
That changes the story:

- **pico@416** was data-starved by the wrong (2 564-image) bundle.
  Fixing the bundle to v4's 10 671-image source closed +0.035 of the
  gap, multiscale closed another +0.012, leaving −0.022 unexplained.
- **n@640** had enough capacity to extract roughly the same signal
  from either bundle. The −0.020 gap to v4 on n@640 isn't from data
  quantity at all.

So whatever residual we still see — a fairly consistent ~0.020 on both
cells — is **not** the source bundle. It's something else.

## Candidates for the remaining ~0.02 residual (same on both cells)

In rough order of likelihood:

1. **DEIMv2 SHA bump** (`377e10a` → `aeabc7e`). v4 trained under
   `377e10a`; v7.1 trained under `aeabc7e`. "DDP loss-key alignment"
   shouldn't matter on single-GPU, but other changes in that diff
   could.
2. **Multi-class refactor side effects.** The 2026-05-24 kit merge
   moved category handling from singular to plural. For a 1-class
   task this should be a no-op but might not be.
3. **Tile-boundary handling.** kit's `tile.py` vs v4's
   `tile_kwcoco.py` quadrant-mode boundary semantics. We're using
   tile bundles produced by the kit; v4 used its own tiler.
4. **v4's `_train_deimv2_variant.sh` config detail** — e.g., the
   `num_top_queries` clamp logic the kit may not exactly replicate.

To bisect cleanly we'd need to roll DEIMv2 back to `377e10a` and retrain.
~3 GPU-hours for pico alone. The recovered AP might be ≤ 0.02; might
also be smaller. Not great expected value vs. the v9 distillation
experiment.

## Provenance (auto-stamped this run — confirmed working end-to-end)

From both `policy.json` and `detect_metrics.json`:

```json
"provenance": {
    "kit_sha": "059f60c7f91a...",
    "deimv2_sha": "aeabc7e400e5...",
    "opengroundingdino_sha": "9ddf10371a46..."
}
"eval_inputs": {
    "test_kwcoco": "/data/joncrall/dvc-repos/.../test.simplified.kwcoco.zip",
    "score_thresh": 0.001,
    "category_names": ["poop"]
}
```

First run with fully working two-sided stamps (kit `059f60c` fixed
the eval-side `category_name` → `category_names` regression caught in
v6.1's logs).

## Decision

- [x] **Proceed to v9 distillation.** The ~0.02 residual is a different
      kind of variable than what v6.1 / v7.1 isolated. Distillation
      from the v9 OGDino bbox teacher (AP=0.766) is independent of
      whatever's causing it, and the original plan estimated +0.03 to
      +0.08 from teacher knowledge. If v9 hits +0.03, we're at v4-parity
      or better on both cells.
- [ ] Bisect DEIMv2 `377e10a..aeabc7e` to chase the residual. ~3 GPU-h
      pico, uncertain payoff. Tabled.

## Run identity

- Recipe: `experiments/mobile_app_training_v7_1/recipe.yaml`
- Workspace: `/data/joncrall/kcd/v7_1/`
- Source bundle: `/data/joncrall/kcd/v6_1/data/{train,vali}_tile_g2.kwcoco.zip`
  (reused from v6.1 — built from `train_imgs10671`)
- Kit: `059f60c`, DEIMv2: `aeabc7e`, OGDino: `9ddf1037`
- Test set: v9 simplified test GT
