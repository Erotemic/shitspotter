#!/bin/bash
# Train DEIMv2-Pico (HGNetv2-Pico backbone) — speed fallback for Pixel 5.
#
# Use this if DEIMv2-N misses the 10-FPS target on-device. Pico has ~1.5M
# params and ~5.2 GFLOPs vs N's 3.6M / 6.8.

set -euo pipefail

_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_script_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi

export V4_VARIANT="deimv2_pico"
export V4_INPUT_HW="${V4_INPUT_HW:-320 320}"
# HGNetv2 hybrid encoder doesn't support per-batch resize.
export V4_TRAIN_POLICY="${V4_TRAIN_POLICY:-fixed}"
export V4_TRAIN_BATCH="${V4_TRAIN_BATCH:-64}"
export V4_VAL_BATCH="${V4_VAL_BATCH:-128}"
export V4_NUM_EPOCHS="${V4_NUM_EPOCHS:-80}"

bash "$_v4_script_dpath/_train_deimv2_variant.sh"
