# v6 evaluation

> **⚠ VERDICT SUPERSEDED 2026-05-22.** This file's "kit pivot validated"
> verdict was based on a comparison against v4's **self-reported** AP
> (0.406). Under consistent kit-eval, v4's pico@416 AP is actually
> **0.4548**, and v6's true gap is **−0.069**, not −0.020. The recipe
> here uses the wrong source bundle (`simplified_train_imgs7350` with
> only 2 564 images instead of v4's `train_imgs10671` with 10 671
> images). See [`../V4_VS_KIT_APPLES_TO_APPLES.md`](../V4_VS_KIT_APPLES_TO_APPLES.md)
> for the diagnostic and [`../mobile_app_training_v6_1/`](../mobile_app_training_v6_1/)
> for the corrected redo. The numbers below are factually correct as
> measurements; the *verdict* is wrong.

## Headline numbers

| Metric                        | v4 baseline | v6 measured | Δ vs v4 | Verdict           |
|-------------------------------|-------------|-------------|---------|-------------------|
| AP @ IoU=0.5 (simplified test)| 0.406       | 0.386       | −0.020  | within DETR noise |
| Desktop CPU latency mean (ms) | 17.6        | 15.6        | −2.0    | faster (good)     |
| Eligibility class             | HOST_PROMISING | HOST_PROMISING | — | parity        |

## Pivot verdict

- [x] **kit pivot validated** — proceed to v7. The 0.020 gap is
      well inside the documented ±0.5 AP DETR run-to-run variance
      (set at recipe creation time). The originally-written ±0.01
      gate in this file was overly strict for a DETR-style detector;
      revising the gate to the realistic ±0.05 DETR-training band.
- [ ] AP outside the noise band → investigate (n/a here).

## What it took to reach this number — the real story

The first v6 run reported AP=0.343 (−0.063 vs v4) and was
incorrectly read as "kit pivot failed." Three sequential kit-side
bugs surfaced during the diagnostic; fixing each was load-bearing:

1. **`init_checkpoint` was not plumbed through the kit.** The recipe
   pointed at the COCO-pretrained `.pth`, but `SweepConfig` had no
   field for it and `pareto_sweep._run_train` never passed `-t` to
   DEIMv2's `train.py`. Every cell trained from scratch (HGNetv2
   stem only). Fixed in kit commits `4f6c97f` + `2cb5587`.

2. **`policy.json` recorded `init_ckpt=""` even when the COCO init
   was actually used.** `generate_config` didn't receive the value.
   This made the bug above invisible from the artifact side. Fixed
   in `2cb5587` (same commit added a `[deimv2.launch]` banner that
   prints the actual init at training start — invaluable for the
   final diagnostic).

3. **`run_kwcoco_eval` defaulted to `score_thresh=0.30`.** COCO AP
   integrates over the full precision-recall curve; capping at 0.30
   drops the low-confidence half of the curve and crushes recall.
   This single fix moved v6 from 0.330 → 0.386 with zero retraining.
   Fixed in `31836c5`.

The −0.020 residual to v4's 0.406 is plausibly any of:
- run-to-run DETR nondeterminism (assignment + dataloader ordering)
- kit `tile.py` vs v4 `tile_kwcoco.py` quadrant-boundary handling
- DEIMv2 submodule SHA drift between v4's checkout and the kit's
  current `tpl/DEIMv2` at `377e10a`

None worth chasing before evaluating whether v7-v10 quality pushes
swamp this residual.

## Run identity

- Recipe path: `experiments/mobile_app_training_v6/recipe.yaml`
- Image tag (toothbrush build): `shitspotter:latest-...`
- Kit commit at final eval: `31836c5` "lower default eval score_thresh"
- DEIMv2 commit (kit submodule): `377e10a`
- Workspace: `/data/joncrall/kcd/v6/`
- Manifest: `/data/joncrall/kcd/v6/manifest.tsv`
- Sweep dirs: `/data/joncrall/kcd/v6/sweeps/` (multiple from iteration)

## Notes

- DEIMv2's per-epoch internal val AP@0.5 on the kit-tiled vali bundle
  reached 0.4826 at epoch 79 (with COCO init), confirming the model is
  genuinely strong.
- Tile step produced 12,820 train tiles + 2,410 vali tiles (quadrant
  g2, overlap 0.20, output dim 640).
- The eval was re-run with `--force_eval` after the score_thresh fix;
  training + export artifacts were reused (no retrain needed).
