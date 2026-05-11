#!/bin/bash
# v5 setup: delegates to v4's 00_setup.sh because v5 shares all of
# v4's deps (DEIMv2, geowatch, onnx trio, pretrained checkpoints).
# Plus a small v5-specific dir-layout step.

set -euo pipefail

_v5_source="${BASH_SOURCE[0]-}"
if [ -n "$_v5_source" ] && [ "$_v5_source" != "bash" ] && [ "$_v5_source" != "-bash" ]; then
    _v5_script_dpath="$(cd "$(dirname "$_v5_source")" && pwd)"
else
    _v5_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v5"
fi
# shellcheck source=experiments/mobile_app_training_v5/common.sh
source "$_v5_script_dpath/common.sh"
unset _v5_source _v5_script_dpath

echo "=== mobile_app_training_v5 / 00 setup ==="
v5_print_env
echo
v5_setup_dirs
echo "  ensured v5 dir layout under $V5_ROOT"

echo
echo "=== Delegating to v4 setup for deps + pretrained ckpts ==="
bash "$V5_DEV_DPATH/../mobile_app_training_v4/00_setup.sh"
