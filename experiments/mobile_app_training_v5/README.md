# mobile_app_training_v5 — multi-scale tiles + hard-negative mining

v5 builds on v4 with two data-side changes that v4 cannot make:

1. **Multi-scale fixed-size tile extraction.** Each source image is
   pre-downscaled to several scales (default `1.0, 0.66, 0.40, 0.25`)
   and a fixed-size (default 320×320) sliding window cuts tiles from
   each scale. The same physical object appears in multiple tiles at
   different apparent sizes — the data-time mirror of the multi-
   resolution inference modes the phone app uses (`BALANCED_FULL_FRAME`,
   `ROI_HIGH_RES`, `TILED_SEARCH`).

2. **Round-based hard-negative mining.** Train on positives + a random
   subsample of negatives (round 0). Run the model on the *full*
   negative pool, find the false-positive predictions, retrain on
   positives + those hard negatives (round 1). Iterate.

v5 deliberately shares v4's trainer / exporter / evaluator / manifest —
all the model code is identical. v5 is a data and training-loop
strategy layered on top.

See [DESIGN.md](DESIGN.md) for the full rationale.

## Quick start

```bash
# Once per shell:
source experiments/mobile_app_training_v5/setup_env.sh

# End-to-end:
bash experiments/mobile_app_training_v5/run_all.sh
```

Defaults:

| knob                          | default               | meaning                                        |
|-------------------------------|-----------------------|------------------------------------------------|
| `V5_ROOT`                     | `~/data/shitspotter_v5` | writable workspace                           |
| `V5_TILE_SIZE`                | `320`                 | fixed output tile size                         |
| `V5_SOURCE_SCALES`            | `1.0,0.66,0.40,0.25`  | source downscale factors                       |
| `V5_STRIDE_FRAC`              | `0.5`                 | tile stride = fraction of tile size            |
| `V5_MIN_GT_AREA_FRAC`         | `0.005`               | tile is positive iff GT area ≥ this × tile area |
| `V5_MIN_KEPT_BOX_FRAC`        | `0.30`                | drop GT annotations clipped below this         |
| `V5_NUM_ROUNDS`               | `3`                   | round-0 + 2 rounds of hard-neg mining          |
| `V5_ROUND0_NEG_OVER_POS`      | `3.0`                 | round-0 negatives = 3× the positive count      |
| `V5_MINE_SCORE_THRESH`        | `0.30`                | tile is "hard" iff max pred score ≥ this       |
| `V5_MAX_HARD_PER_ROUND`       | `5000`                | cap hard-neg count per round                   |
| `V5_ROUND_EPOCHS`             | `20`                  | epochs per round                               |
| `V5_VARIANT`                  | `deimv2_n`            | model variant (passed to v4 trainer)           |
| `V5_INPUT_HW`                 | `320 320`             | model input HxW (matches V5_TILE_SIZE)         |

## File layout

```
experiments/mobile_app_training_v5/
├── README.md                          — this file
├── DESIGN.md                          — full rationale
├── common.sh / setup_env.sh           — env (chains to v4's)
├── 00_setup.sh                        — delegates to v4 setup
├── 01_make_multiscale_tile_dataset.sh — calls v5_tile.py for train+vali
├── v5_tile.py                         — multi-scale fixed-size tile extractor
├── v5_merge.py                        — pos + neg -> next-round training kwcoco
├── v5_mine.py                         — score negatives, emit hard negs
├── 02_train_round.sh                  — single round: merge -> train (dispatches v4)
├── 03_mine_hard_negatives.sh          — single mining pass
├── run_round_loop.sh                  — loops train -> mine -> train -> ...
├── 05_export_onnx.sh                  — final round -> ONNX via v4
├── 06_eval_on_test.sh                 — final round eval via v4 (v9 simplified test)
└── run_all.sh                         — one-shot driver
```

Outputs under `$V5_ROOT`:

```
$V5_ROOT/
├── data/
│   ├── train_tiles.kwcoco.zip         — all train tiles (pos + neg)
│   ├── train_tiles_pos.kwcoco.zip     — positives only (training pool)
│   ├── train_tiles_neg.kwcoco.zip     — negatives only (mining pool)
│   ├── vali_tiles_pos.kwcoco.zip      — vali positives only
│   └── <bundle>_assets/               — actual .jpg files
└── rounds/
    └── round{0,1,2}/
        ├── train_round.kwcoco.zip     — merged training kwcoco for this round
        ├── hard_negs.kwcoco.zip       — round N's mined negs (input to round N+1)
        └── v4_root/                   — v4-style trainer workdir for this round
            └── runs/<candidate_id>/
                ├── best_stg2.pth
                ├── policy.json
                ├── generated_configs/train.yml
                └── export/<name>.onnx (after step 5)
```

## How v5 talks to v4

Every step that needs the model goes through v4's existing scripts.
v5's `02_train_round.sh` sets `V4_ROOT=<round_dir>/v4_root` plus a
handful of `V4_*` env vars and invokes
`experiments/mobile_app_training_v4/_train_deimv2_variant.sh`. Same
for export (`v4/03_export_onnx.sh`) and eval (`v4/04_eval_on_test.sh`).

This means:

* Every v4 fix automatically helps v5.
* v5 inherits v4's pytest coverage for the model side.
* v5's own pytest coverage focuses on the *data* side (tile extractor,
  merger, miner).
* The phone-app integration story (`v4/07_register_in_phone_app.md`)
  carries over unchanged — a v5 export is just another DEIMv2 ONNX.

## Where to run what

| stage                                     | VM | host GPU |
|-------------------------------------------|----|----------|
| `01_make_multiscale_tile_dataset.sh`      | ✓ CPU |  ✓     |
| `02_train_round.sh` / round loop          |    |  ✓ GPU |
| `03_mine_hard_negatives.sh`               |    |  ✓ GPU |
| `05_export_onnx.sh` / `06_eval_on_test.sh`| ✓ CPU |  ✓     |

## Comparison target

The v9 OpenGroundingDINO + tuned SAM2 package reaches AP=0.766 on the
simplified test GT. v4 numbers fill in the apples-to-apples row for
single-scale DEIMv2 detectors. **v5's question:** does multi-scale
tile training + hard-neg mining close more of the gap to v9?

Each round's eval AP lands in `$V5_ROOT/rounds/round<N>/v4_root/eval/.../eval/detect_metrics.json`,
read by the same `nocls_measures.ap` selector v4/v9 use.
