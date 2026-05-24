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
V8_ROUND_EPOCHS=${V8_ROUND_EPOCHS:-30}
# Mining budget: cap negative-tile scoring per round. The multi-scale
# tile pool is ~1.8M negatives; without a budget, mining round 0 alone
# takes ~16 h on a 3090. 50000 stratified-by-image gives a balanced
# sample of every source scene at roughly 30x speedup. Set to 0 to
# disable the cap and score everything (legacy v8.0 behavior).
V8_MINE_MAX_CANDIDATES=${V8_MINE_MAX_CANDIDATES:-50000}
V8_MINE_CANDIDATE_STRATEGY=${V8_MINE_CANDIDATE_STRATEGY:-stratified_by_image}

# Round 0 of each cell fine-tunes from the matching DEIMv2 COCO-pretrained
# checkpoint. Rounds 1+ resume from the prior round's best_stg2.pth
# automatically. Per-cell paths -- override via env if your host stores
# the .pth files elsewhere.
V8_PICO_INIT=${V8_PICO_INIT:-/data/joncrall/shitspotter_v4/pretrained/deimv2/deimv2_pico_coco.pth}
V8_N_INIT=${V8_N_INIT:-/data/joncrall/shitspotter_v4/pretrained/deimv2/deimv2_n_coco.pth}

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
            --category_names poop \
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
    variant_short="${cell%%:*}"
    variant="deimv2_${variant_short}"
    size="${cell##*:}"
    workdir_tag="${variant}_${size}x${size}"

    case "$variant_short" in
        pico) init_ckpt="$V8_PICO_INIT" ;;
        n)    init_ckpt="$V8_N_INIT" ;;
        *)    init_ckpt="" ;;
    esac
    if [ -n "$init_ckpt" ] && [ ! -f "$init_ckpt" ]; then
        echo "[v8] ERROR: init_checkpoint missing: $init_ckpt" >&2
        exit 2
    fi

    echo
    echo "[v8] === round-loop for $workdir_tag (init=$init_ckpt) ==="
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
        --category_names    poop \
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
        --num_gpus         1 \
        --mine_max_candidates    "$V8_MINE_MAX_CANDIDATES" \
        --mine_candidate_strategy "$V8_MINE_CANDIDATE_STRATEGY" \
        ${init_ckpt:+--init_checkpoint "$init_ckpt"}
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
