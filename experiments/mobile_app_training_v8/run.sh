#!/bin/bash
# v8 driver: multi-scale tile -> kit round-loop -> manifest.
#
# Knobs (export to override before running):
#   V8_CELLS              "pico:416 n:640"  (variant:export_size space-separated)
#   V8_NUM_ROUNDS         3
#   V8_ROUND0_NEG_OVER_POS 3.0
#   V8_MINE_SCORE_THRESH  0.30
#   V8_MAX_HARD_PER_ROUND 5000
#   V8_ROUND_EPOCHS       20
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RAW_DPATH=${RAW_DPATH:-/data/joncrall/dvc-repos/shitspotter_dvc}
RAW_TRAIN=${RAW_TRAIN:-$RAW_DPATH/simplified_train_imgs7350_4f0174d0.kwcoco.zip}
RAW_VALI=${RAW_VALI:-$RAW_DPATH/simplified_vali_imgs1258_07ec447d.kwcoco.zip}
TEST_KWCOCO=${TEST_KWCOCO:-/data/joncrall/dvc-repos/shitspotter_expt_dvc/foundation_detseg_v3/v9/detector_prepared/test.simplified.kwcoco.zip}

V8_ROOT=${V8_ROOT:-/data/joncrall/kcd/v8}
V8_DATA=$V8_ROOT/data
mkdir -p "$V8_DATA"

V8_CELLS=${V8_CELLS:-"pico:416 n:640"}
V8_NUM_ROUNDS=${V8_NUM_ROUNDS:-3}
V8_ROUND0_NEG_OVER_POS=${V8_ROUND0_NEG_OVER_POS:-3.0}
V8_MINE_SCORE_THRESH=${V8_MINE_SCORE_THRESH:-0.30}
V8_MAX_HARD_PER_ROUND=${V8_MAX_HARD_PER_ROUND:-5000}
V8_ROUND_EPOCHS=${V8_ROUND_EPOCHS:-20}

# Multi-scale tiles for the mining pool. Produces pos + neg bundles.
# v5 default scales (1.0, 0.66, 0.40, 0.25) cover the operating modes
# the phone app exposes (FAST/BALANCED/ROI/TILED).
for split in train vali; do
    src_var="RAW_${split^^}"
    src="${!src_var}"
    pos="$V8_DATA/${split}_tiles_pos.kwcoco.zip"
    neg="$V8_DATA/${split}_tiles_neg.kwcoco.zip"
    if [ ! -f "$pos" ] || [ ! -f "$neg" ]; then
        echo "[v8] multi-scale tile $split"
        kwcoco-detector-kit tile "$src" "$V8_DATA/${split}_tiles.kwcoco.zip" \
            --mode multiscale \
            --category_name poop \
            --tile_size 320 \
            --source_scales "1.0,0.66,0.40,0.25" \
            --stride_frac 0.5 \
            --min_gt_area_frac 0.005 \
            --keep_negative True
        # The kit's multiscale tile mode writes a single bundle with
        # tile_role={positive,negative}. Split into two bundles for
        # round-loop's pos/neg inputs. (If the kit already emits these
        # as separate bundles, replace this block with two copies.)
        python3 -c "
import kwcoco
src = kwcoco.CocoDataset('$V8_DATA/${split}_tiles.kwcoco.zip')
for role, dst in [('positive', '$pos'), ('negative', '$neg')]:
    keep = [g['id'] for g in src.imgs.values()
            if g.get('tile_role') == role]
    sub = src.subset(keep)
    sub.fpath = dst
    sub.dump()
    print(f'wrote {dst} ({len(keep)} images)')
"
    fi
done

for cell in $V8_CELLS; do
    variant="deimv2_${cell%%:*}"
    size="${cell##*:}"
    workdir_tag="${variant}_${size}x${size}"
    echo
    echo "[v8] === round-loop for $workdir_tag ==="
    KCD_ROOT="$V8_ROOT/$workdir_tag" \
    kwcoco-detector-kit round-loop \
        --pos_tiles_kwcoco "$V8_DATA/train_tiles_pos.kwcoco.zip" \
        --neg_tiles_kwcoco "$V8_DATA/train_tiles_neg.kwcoco.zip" \
        --vali_kwcoco      "$V8_DATA/vali_tiles_pos.kwcoco.zip" \
        --test_kwcoco      "$TEST_KWCOCO" \
        --kcd_root         "$V8_ROOT/$workdir_tag" \
        --trainer          deimv2 \
        --variant          "$variant" \
        --input_hw         "[$size,$size]" \
        --train_policy     fixed \
        --category_name    poop \
        --num_classes      1 \
        --num_rounds       "$V8_NUM_ROUNDS" \
        --round0_neg_over_pos "$V8_ROUND0_NEG_OVER_POS" \
        --mine_score_thresh "$V8_MINE_SCORE_THRESH" \
        --max_hard_per_round "$V8_MAX_HARD_PER_ROUND" \
        --num_epochs       "$V8_ROUND_EPOCHS" \
        --batch_size       16 \
        --val_batch_size   32 \
        --lr               5.0e-4 \
        --backbone_lr      2.5e-5 \
        --use_amp          True \
        --scale_tier       M \
        --num_gpus         1
done

# Aggregate the final rounds' artifacts into one cross-cell manifest.
echo
echo "[v8] aggregating manifest"
kwcoco-detector-kit manifest --auto \
    --kcd_root "$V8_ROOT" \
    --out "$V8_ROOT/manifest.tsv" \
    --out_json "$V8_ROOT/manifest.json" \
    --max_desktop_ms 80.0 \
    --min_device_fps 1.0 \
    --print_winner True

echo "[v8] manifest -> $V8_ROOT/manifest.tsv"
