# v13 — pico@768 multi-scale tiled corpus: evaluation & investigation log

Experiment: `shitspotter_v13_pico768_multiscale` (DEIMv2 pico, 768×768 fixed
input, trained on a multi-scale tile corpus + scale-balancing). Goal: scale
robustness for the move-around-the-yard / robot-pickup use case, to be measured
under a **tiled (windowed) inference protocol** (cf. sealions: whole-image
0.135 → tiled 0.840).

## TL;DR (2026-06-13)

- The completing run **`v13c` finished all 10 epochs cleanly.** Final reported
  test **AP 0.5999** (`best_stg2.pth`, which is **epoch 2**).
- **This number is NOT a fair test of the approach.** Despite `tiled_eval: true`
  in the recipe, the eval ran **whole-image**: predictions cover the 118 native
  1200×1600 test images, not tiles. The tile-trained 768 model was thus run on
  whole frames downscaled to 768 — the exact protocol tiling was meant to beat.
  **The tiled-eval plumbing did not engage; v13 is INCONCLUSIVE, not a clean
  negative.** Fix the plumbing and re-eval the v13c checkpoint under tiling
  before judging.
- **The extra epochs hurt.** Best checkpoint is epoch 2; the cosine-anneal
  phase (epochs 5–9) gave no gain. Training visibly **degraded at epoch 4**
  (train loss jumped 10.9 → 16.1, in-loop AP 0.469 → 0.428, never recovered),
  consistent with DEIM's augmentation schedule ramping strong aug mid-run —
  wrong for a short warm-started fine-tune.

## In-loop AP per epoch (whole-image, DEIM eval)

| epoch | AP | AP50 | train_loss | note |
|------:|------|------|-----------|------|
| 0 | 0.4674 | 0.6717 | 11.84 | warm-start (v13 epoch-3 ckpt) |
| 1 | 0.4694 | 0.6748 | 11.35 | |
| 2 | 0.4694 | 0.6738 | 11.08 | **best_stg2** |
| 3 | 0.4688 | 0.6728 | 10.89 | |
| 4 | 0.4254 | 0.6169 | 16.14 | **loss spike — aug ramp; AP drops** |
| 5 | 0.4345 | 0.6275 | 15.48 | flat_epoch=5 anneal begins |
| 6 | 0.4395 | 0.6330 | 15.22 | |
| 7 | 0.4435 | 0.6376 | 15.04 | |
| 8 | 0.4464 | 0.6404 | 14.97 | |
| 9 | 0.4481 | 0.6423 | 14.84 | never recovers to epoch-2 level |

## Comparison (all WHOLE-IMAGE / not the tiled protocol)

| run | model | AP | notes |
|-----|-------|----|-------|
| v10 baseline | pico@640 | 0.588 | prior ship candidate |
| v13b (partial) | pico@768 | 0.609 | epoch-3 partial, died at epoch-4 eval |
| **v13c (full)** | pico@768 | **0.5999** | epoch-2 best; 10 epochs; anneal no help |

So at the **whole-image** protocol, v13 (multiscale tiled corpus + scale
balance) does **not** beat v10. But whole-image is the wrong yardstick for a
tile-trained model — the tiled protocol is the one that motivated this whole
arm and it has not been run. **Verdict deferred** pending correct tiled eval.

## The epoch-4 in-loop eval crash (resolved: transient, not a version bug)

Both the original v13 and the v13b relaunch died at the **epoch-4 in-loop
eval** with:

```
faster_coco_eval/core/coco.py:278 loadAnns:  return [self.anns[i] for i in ids]
TypeError: 'int' object does not support the context manager protocol
```

Investigation (don't repeat the dead ends):

- **NOT a faster-coco-eval version regression.** Initially pinned
  `faster_coco_eval<1.7` on a hunch; **reverted** after the in-container
  `diagnose_coco_eval.py` proved clean 1.7.2 builds `cocoDt.anns` as a plain
  `dict` and evaluates synthetic input fine. A dict subscript cannot run a
  `with`, so the error implies a transient malformed state, not the library.
- **Transient, not a deterministic bad item.** `repro_eval_crash.sh` ran DEIM
  `--test-only` (`solver.val()` → the same `CocoEvaluator` path) over the full
  vali corpus on both the v13 warm-start and the v13b checkpoint — **neither
  reproduced it.** A deterministic bad annotation/prediction would have fired on
  a full pass.
- **Robustness, not a fix.** The kit DEIM fork's `CocoEvaluator.update`
  (`tpl/DEIMv2/engine/data/dataset/coco_eval.py`) now **dumps the offending
  batch** (`coco_eval_crash_dump_pid*.pkl`) + warns + **skips that batch**
  instead of crashing a multi-hour run. v13c completed with **no recurrence**
  (no dump written) — the patch was insurance, never triggered.

Diagnostic tooling left in this dir: `diagnose_coco_eval.py` (env/`anns`-type
probe), `repro_eval_crash.sh` (cheap eval-only repro), `analyze_eval_crash.py`
(bisects a dump to the offending detection).

> The kit-fork `coco_eval.py` patch is currently **uncommitted in the
> detached-HEAD DEIMv2 submodule** and runs live via the `MOUNT_KCD` /
> `CODE_MOUNT` bind-mount. Commit it upstream (branch in the fork + bump the kit
> pointer + rebuild image) once we're done iterating.

## Open items / next steps

1. **Fix the `tiled_eval` plumbing** in the kit sweep→eval path — `tiled_eval:
   true` / `tiled_eval_overlap: 0.2` did not produce a windowed eval. Until
   this works, no v13 conclusion is valid.
2. **Re-eval `v13c/best_stg2.pth` under tiling** and compare to v10 fairly.
3. The epoch-4 aug-ramp degradation suggests a **shorter / aug-disabled
   schedule** for warm-started fine-tunes (or just take `best_stg2`=epoch 2).
