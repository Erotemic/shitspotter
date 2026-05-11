#!/bin/bash
# Train DEIMv2-S (DINOv3-backed ViT-Tiny backbone) — quality reference and
# future teacher.
#
# This is NOT the live phone candidate. Use it to:
#   * establish an upper bound on what the v4 tile-augmented training data
#     can reach;
#   * compare against the v9 OpenGroundingDINO teacher;
#   * (future) act as the source for distilling DEIMv2-N / Pico.

set -euo pipefail

_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_script_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi

export V4_VARIANT="deimv2_s"
export V4_INPUT_HW="${V4_INPUT_HW:-640 640}"
export V4_TRAIN_BATCH="${V4_TRAIN_BATCH:-32}"
export V4_VAL_BATCH="${V4_VAL_BATCH:-64}"
# DEIMv2-S converges much faster than the upstream 132 epoch config when
# fine-tuning from the COCO checkpoint.
export V4_NUM_EPOCHS="${V4_NUM_EPOCHS:-30}"

bash "$_v4_script_dpath/_train_deimv2_variant.sh"
