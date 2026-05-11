#!/bin/bash
# Train DEIMv2-N (HGNetv2-N backbone) — primary Pixel 5 phone candidate.
#
# Default input: 320x320 → smallest plausible "real" detector for live mode.
# Override with V4_INPUT_HW="416 416" or "640 640" if you want the larger
# input to feed straight into the existing 640x640 phone-app model slot.
#
# Memory rough estimate for a 24 GB GPU at 320:
#   total_batch_size=128 fits comfortably; halve if you OOM.

set -euo pipefail

_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_script_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi

export V4_VARIANT="deimv2_n"
export V4_INPUT_HW="${V4_INPUT_HW:-320 320}"
# HGNetv2 hybrid encoder doesn't support per-batch resize — see
# _train_deimv2_variant.sh's per-variant default block. Keep at fixed.
export V4_TRAIN_POLICY="${V4_TRAIN_POLICY:-fixed}"
export V4_TRAIN_BATCH="${V4_TRAIN_BATCH:-32}"
export V4_VAL_BATCH="${V4_VAL_BATCH:-64}"
# 60 epochs is enough to fine-tune from COCO weights; the upstream config
# defaults to 160 from a cold start.
export V4_NUM_EPOCHS="${V4_NUM_EPOCHS:-60}"

bash "$_v4_script_dpath/_train_deimv2_variant.sh"
