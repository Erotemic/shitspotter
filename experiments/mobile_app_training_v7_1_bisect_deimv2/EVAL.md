# DEIMv2 bisect — RESULT: bump explains ~80% of the residual

> **⚠ VERDICT REVISED 2026-05-29.** A subsequent v4-vs-kit config audit
> (see [RESEARCH_JOURNAL.md](../RESEARCH_JOURNAL.md) 2026-05-29 entry
> + commit `65784ea`) found the MSCOCO inputs to be bit-identical and
> the generated YAMLs semantically equivalent under both DEIMv2 SHAs.
> Re-reading the DDP commit diff confirmed it's correctly guarded for
> single-GPU (early-return on `world_size < 2`). With no code path
> that could affect single-GPU math, the most parsimonious explanation
> for the +0.0175 between v7.1 and bisect is **DETR run-to-run
> variance**, not a true regression. The kit pivot is genuinely
> validated. The remainder of this file describes the bisect's
> measurement faithfully, but its "bump caused it" interpretation is
> superseded.

## Headline result

| Cell     | v4 (kit eval, 377e10a) | v7.1 (aeabc7e) | **bisect (377e10a)** | Δ bisect vs v7.1 | Δ bisect vs v4 |
|----------|------------------------|----------------|----------------------|------------------|----------------|
| pico@416 | 0.4548                 | 0.4329         | **0.4504**           | **+0.0175**      | **−0.0044**    |

**The DEIMv2 SHA bump caused ~80% of v7.1's residual on pico@416.**
Rolling back to v4's SHA closed +0.0175 of the 0.0219 gap, leaving
−0.0044 — solidly inside DETR run-to-run noise.

Bisect lands at the verdict band:
- [x] **~0.45 (Δ ≥ +0.015 vs v7.1)**: bump caused most of it.
- [ ] ~0.43 ± 0.01 (no movement)
- [ ] < 0.42 (rollback hurt)

## What that means

The 377e10a..aeabc7e DEIMv2 diff is two commits:
1. `aeabc7e setup_print: prefix every emitted line with an ISO-8601 timestamp` (cosmetic)
2. The DDP loss-key alignment commit (single-GPU-irrelevant *in principle*)

Yet the bump cost ~0.017 AP on a single-GPU pico run. Most likely
culprit: the DDP loss-key alignment commit changed the loss aggregation
keys in a way that subtly affects gradient flow even on single-GPU
training. Worth filing upstream.

For our purposes: **the kit + DEIMv2 377e10a + v6.1's tile bundle +
v7's multiscale policy reproduces v4 within DETR noise on pico@416.**
That's the validation we've been chasing since the start of v6.

## n@640 extrapolation

We bisected pico only (cheap diagnostic). If the bump cost the same
~0.017 on n@640, then a hypothetical n@640 rerun under 377e10a would
land at:

  0.5350 (v7.1 n@640) + 0.017 ≈ **0.5520**  vs  v4's 0.5553.

That's −0.003 to v4 — also within noise. **Worth confirming** but the
prior is strong enough to plan v10 around 377e10a even before
re-running n.

## Cost

- Wall-clock: **~11 hours** on single 3090 (pico@416, 80 epochs, multi-
  scale, 53K-tile bundle). NOT 3 hours as I estimated. Per-epoch
  ~8 min vs v6.0's ~2.4 min: ~4× the dataset (corrected bundle) +
  ~25% multiscale overhead = ~3.3× per-epoch + the export/eval/bench
  tail puts it at ~11h total.
- Wall-clock measurement source: file mtimes of best_stg1.pth (first
  val improvement) → log.txt last write (epoch 79 done).

## Provenance (auto-stamped, confirmed)

```json
"provenance": {
    "kit_sha":              "a8ca45e0347f...",
    "deimv2_sha":           "377e10a273fa...",
    "opengroundingdino_sha": "9ddf10371a46..."
}
"eval_inputs": {
    "category_names": ["poop"],
    "test_kwcoco":   ".../test.simplified.kwcoco.zip",
    "score_thresh":  0.001
}
```

## Decision + cleanup

- [x] **Restore kit `tpl/DEIMv2` to `aeabc7e`** in a follow-up commit
      so the sealions line stays unblocked. The bisect was diagnostic;
      production decisions go separately.
- [x] **Pin v10 ship recipes to DEIMv2 `377e10a`**, OR distill from the
      v9 OGDino teacher to overshoot the residual entirely — see the
      RESEARCH_JOURNAL 2026-05-29 entry for the strategic choice.
- [ ] **File upstream DEIMv2 issue** about the regression. Document
      single-GPU pico AP delta of −0.017 between commits.

## Open question

Was the regression intentional or a missed regression in upstream? The
DDP loss-key alignment commit message claims it's a DDP-only change.
Either:
- The commit also subtly changed single-GPU behavior (missed by
  upstream review), or
- Some other behavior in `aeabc7e` (e.g., a build/init reordering)
  drives the AP delta.

Bisecting between `377e10a` and `aeabc7e` would isolate which of the
two intermediate commits causes it — but that's two more ~11h runs.
Pragmatic move: file the issue with our bisect-narrowed window
(`377e10a..aeabc7e`) and move on.
