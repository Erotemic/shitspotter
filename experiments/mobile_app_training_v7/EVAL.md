# v7 evaluation

> **⚠ VERDICT SUPERSEDED 2026-05-22.** The "n@640 WIN vs v4" claim
> below was based on v4's **self-reported** 0.520. Under consistent
> kit-eval, v4's n@640 AP is actually **0.5553**, and v7's true gap is
> **−0.020**, not +0.015. v7 does **not** beat v4. The recipe here also
> uses the wrong source bundle (see v6's superseded note). See
> [`../V4_VS_KIT_APPLES_TO_APPLES.md`](../V4_VS_KIT_APPLES_TO_APPLES.md).
> Numbers below are correct measurements; the verdict is wrong.

## Headline numbers

| Cell        | v4 fixed AP | v6 kit baseline | v7 multiscale AP | Δ vs v4   | Δ vs v6   | Verdict        |
|-------------|-------------|-----------------|------------------|-----------|-----------|----------------|
| pico@416    | 0.406       | 0.386           | **0.398**        | −0.008    | +0.012    | ~=v4 (in noise)|
| n@640       | 0.520       | (not run in v6) | **0.535**        | **+0.015**| —         | **WIN vs v4**  |

n@640 is the **first cell in the ladder to beat v4**. pico@416 closed
the v6 kit-baseline gap (+0.012 from multiscale) and landed effectively
at v4 (delta within the DETR run-to-run noise band).

## Decision

- [x] **Both cells advance to v8.** n@640 because it's already
      winning; pico@416 because it's at parity and v8's hard-neg
      mining is independent of the train policy that got us here.

## What it took — surprise architecture mismatch on n@640

v7 surfaced a kit bug that did not fire on the pico cells: when an
HGNetv2 variant doesn't explicitly set `DEIMTransformer.use_gateway`
in its upstream config, DEIMv2's YAMLConfig can pick a different
default at eval time than at train time. n@640 trained fine but the
post-train eval crashed with state_dict-vs-architecture mismatch
(saved gateway keys vs. eval-time non-gateway model).

Fixed in kit commit `78c5654` with two safety layers:

1. **Per-variant `use_gateway` table** in the kit's variant registry,
   written explicitly into every generated train.yml. Future runs
   can't silently disagree across train / export / eval.
2. **`DEIMv2Predictor` auto-detect** — inspects the saved state_dict
   for `gateway.*` keys and force-sets the YAML before building the
   model. This salvages already-trained checkpoints written by the
   pre-fix kit (which is how we got the n@640 number on this row
   without retraining).

## Run identity

- Kit commits at final eval:
  - `70c2270` round_loop init_checkpoint (pre-existing)
  - `78c5654` use_gateway per variant + predictor auto-detect (eval-time)
- Workspace: `/data/joncrall/kcd/v7/`
- Manifest: `/data/joncrall/kcd/v7/manifest.tsv`

## Caveat: exported ONNX may be wrong

The n@640 export step ran with the pre-fix YAML and may have written
an ONNX with `use_gateway=False` and partial weights. Desktop bench
timing was valid but ONNX-based predictions are suspect. If you need
the n@640 ONNX (for phone-app or any other downstream consumer), force
re-export so the new code path writes the correct architecture into
the generated train.yml and the export subprocess builds the right
model:

```bash
./reproduce/mobile_quality_push.sh v7 --force_export --force_eval
```

Not blocking for v8 (v8 trains from the .pth, not the ONNX).

## Notes

- Per-cell DEIMv2 internal val AP@0.5 trajectory is in
  `/data/joncrall/kcd/v7/runs/<cell>/log.txt`. Grep for
  `test_coco_eval_bbox` to see the per-epoch arc.
- Both cells used multi-scale tile training data shared with v6
  (`/data/joncrall/kcd/v6/data/`) plus a wider train-resolution band
  (multiscale_320_512 for pico, multiscale_512_768 for n).
