#!/bin/bash
# Cheaply reproduce the v13 in-loop eval crash (minutes, NO training) to capture
# the offending-batch artifact dumped by the patched DEIM CocoEvaluator
# (kwcoco_detector_kit/tpl/DEIMv2/engine/data/dataset/coco_eval.py).
#
# Both v13 and v13b died at the epoch-4 eval; the warm-start checkpoint IS the
# epoch-3 state whose predictions trip loadRes->loadAnns. DEIM `--test-only`
# runs solver.val() -> evaluate() -> CocoEvaluator.update over the vali corpus,
# i.e. the exact crash path, in one pass.
#
# Run from the shitspotter repo root, with the kit bind-mounted so the patched
# coco_eval.py is live (no image rebuild needed):
#
#   MOUNT_KCD=1 reproduce/in_docker.sh bash experiments/mobile_app_training_v13/repro_eval_crash.sh
#
# On the degenerate batch the patch writes coco_eval_crash_dump_pid<PID>.pkl
# into $OUT (the CWD) and re-raises. Inspect it with analyze_eval_crash.py.
set -euo pipefail

DEIM=${DEIM:-/root/code/kwcoco_detector_kit/tpl/DEIMv2}
CFG=${CFG:-/media/joncrall/flash1/kcd-ssd/v13b/runs/deimv2_pico_768x768_fixed/generated_configs/train.yml}
# The epoch-3 warm-start whose predictions crash (use v13b best if you prefer).
CKPT=${CKPT:-/media/joncrall/flash1/kcd-ssd/v13/runs/deimv2_pico_768x768_fixed/best_stg2.pth}
OUT=${OUT:-/media/joncrall/flash1/kcd-ssd/v13b/repro_eval_crash}

mkdir -p "$OUT"
echo "[repro] DEIM=$DEIM"
echo "[repro] config=$CFG"
echo "[repro] ckpt=$CKPT"
echo "[repro] dump dir (CWD)=$OUT"
[ -f "$CFG" ]  || { echo "[repro] missing config: $CFG (run.sh generates it)"   >&2; exit 2; }
[ -f "$CKPT" ] || { echo "[repro] missing checkpoint: $CKPT"                     >&2; exit 2; }

# PYTHONPATH so `from engine import ...` resolves; cd to OUT so the crash dump
# (written to os.getcwd()) lands there.
cd "$OUT"
PYTHONPATH="$DEIM:${PYTHONPATH:-}" python "$DEIM/train.py" \
    -c "$CFG" -r "$CKPT" --test-only --output-dir "$OUT" --seed 0

echo "[repro] eval-only finished WITHOUT crashing -- the bug did NOT reproduce"
echo "[repro] on this checkpoint. Try CKPT=v13b/.../best_stg2.pth or a later epoch."
