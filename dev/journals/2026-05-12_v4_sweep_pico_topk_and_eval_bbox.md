# 2026-05-12 — v4 sweep: pico topk + n@320 eval bbox

Context: the first long Pareto sweep
(`/data/joncrall/shitspotter_v4/sweeps/20260511T134137Z`, started
2026-05-11 13:41 UTC) ran for ~26 h and produced the first per-cell
signal across the deimv2_n × {320,416,512,640} + deimv2_pico ×
{320,416,512} matrix. `08_status.sh` (committed in `ba5a5a3`)
summarised what's on disk.

## Sweep state at the time of writing

| cell | train | onnx | eval AP (simplified, IoU=0.5) | desktop ms | sweep status |
|---|---|---|---|---|---|
| deimv2_n@320 fixed | ok | 13.7M | — | — | **fail_eval** |
| deimv2_n@416 fixed | ok | 13.7M | **0.4056** | 20.3 | ok |
| deimv2_pico@320 fixed | epoch 0 only | — | — | — | **fail_train** |
| deimv2_pico@416 fixed | epoch 0 only | — | — | — | **fail_train** |
| deimv2_n@512 fixed | ok | 13.7M | **0.4765** | 31.5 | ok |
| deimv2_pico@512 fixed | epoch 0 only | — | — | — | **fail_train** |
| deimv2_n@640 fixed | training (epoch 17 best so far, COCO AP@.50=0.554) | — | — | — | in-flight |

Healthy outcomes (n@416, n@512, n@640) confirm the pipeline works
end-to-end on the deimv2_n column. Two specific bugs broke the rest.

## Bug 1 — `num_top_queries > num_queries × num_classes` (deimv2_pico × {320,416,512})

All three pico cells died at the first val pass of epoch 0:

```text
RuntimeError: selected index k out of range
  at tpl/DEIMv2/engine/deim/postprocessor.py:59
  scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
```

Root cause: upstream default `num_top_queries: 300` (from
`tpl/DEIMv2/configs/base/dfine_hgnetv2.yml:76`), pico's
`num_queries: 200` (from
`tpl/DEIMv2/configs/deimv2/deimv2_hgnetv2_pico_coco.yml:44`),
shitspotter `num_classes: 1`. Flatten size = 200 × 1 = **200**;
topk(k=300) → out of range. Upstream's COCO configs at 80 classes
sit at 200 × 80 = 16000, so the inequality is never visible in their
benchmarks.

Same trap waits for `femto` (150 q) and `atto` (100 q) variants.

Proposed fix: emit `num_top_queries = min(upstream_value,
num_queries * num_classes)` in the v4-generated train.yml. For
shitspotter that's `min(300, num_queries)`.

## Bug 2 — kwcoco coco_eval `KeyError: 'bbox'` (deimv2_n@320 only)

The 320 cell trained successfully, exported a real 13.7M ONNX, and
`cli_predict_boxes` wrote 118/118 predictions cleanly. The
subsequent `kwcoco coco_eval` invocation died at
`kwcoco/coco_evaluator.py:510`:

```python
boxes=kwimage.Boxes([a['bbox'] for a in anns], 'xywh'),
# KeyError: 'bbox'
```

Same predictor at 416 and 512 produced AP fine. Strongest
hypothesis: at 320×320 input the model emits at least one
prediction that degenerates to zero-area, and the kwcoco-side ann
serialisation drops the `bbox` field instead of filtering the ann.

`04_eval_on_test.sh:50` already comments on the **true-side**
version of this bug (the v9 simplified test GT had it earlier and
the mock dispatcher works around it). The predicted-side wasn't
guarded.

Proposed fix: filter prediction anns without `bbox` in
`cli_predict_boxes` (the predictor owns the output contract), and
either add `--skip_invalid_anns` to the kwcoco eval call as belt-
and-braces or warn-and-skip in the evaluator.

## What's actionable now

1. **Don't kill the in-flight 640 cell.** It's past epoch 17 already
   with a healthy COCO AP@.50 of 0.554, and a clean n@640 row
   completes the Pareto front for the deimv2_n column.
2. **After the sweep finishes**, land both fixes and re-run the four
   broken cells:
   * 3× `deimv2_pico × {320,416,512}` after the topk clamp.
   * 1× `deimv2_n@320` after the predict-side bbox filter.
3. **Add the failure-mode hints** to the
   `_train_deimv2_variant.sh` error-helper block and to the
   `04_eval_on_test.sh` failure path so the next sweep prints the
   fix inline with the traceback.

## References

* Lessons: `dev/journals/lessons_learned.md` §2026-05-12.
* Benchmark candidates: `dev/benchmark-candidates/pipeline-bootstrap-questions.md`
  Q5 (num_top_queries clamp) and Q6 (predict-side bbox invariant).
* Status helper: `experiments/mobile_app_training_v4/08_status.sh`.
* On-disk layout: `experiments/mobile_app_training_v4/README.md`
  §"Where am I in the sweep?".
