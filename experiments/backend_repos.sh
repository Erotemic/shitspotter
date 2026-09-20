#!/bin/bash
# Canonical locations for generic detector / segmenter repositories.
#
# ShitSpotter owns experiment policy and domain-specific artifacts. Generic
# training backends live in the sibling kwcoco_detector_kit repository. Keep
# the legacy SHITSPOTTER_* variable names because historical experiment
# scripts already consume them, but point their defaults at KDK.
#
# Override KWCOCO_DETECTOR_KIT_DPATH before sourcing this file if the two
# repositories are not siblings under the same parent directory.

_backend_repos_source="${BASH_SOURCE[0]-}"
if [ -n "$_backend_repos_source" ] && [ "$_backend_repos_source" != "bash" ] && [ "$_backend_repos_source" != "-bash" ]; then
    _backend_repos_script_dpath="$(cd "$(dirname "$_backend_repos_source")" && pwd)"
    _backend_repos_repo_dpath="$(cd "$_backend_repos_script_dpath/.." && pwd)"
else
    _backend_repos_repo_dpath="${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}"
fi
unset _backend_repos_source
unset _backend_repos_script_dpath

export SHITSPOTTER_DPATH="${SHITSPOTTER_DPATH:-$_backend_repos_repo_dpath}"
unset _backend_repos_repo_dpath

export KWCOCO_DETECTOR_KIT_DPATH="${KWCOCO_DETECTOR_KIT_DPATH:-$(cd "$SHITSPOTTER_DPATH/.." && pwd)/kwcoco_detector_kit}"

export SHITSPOTTER_DEIMV2_REPO_DPATH="${SHITSPOTTER_DEIMV2_REPO_DPATH:-$KWCOCO_DETECTOR_KIT_DPATH/tpl/DEIMv2}"
export SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH="${SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH:-$KWCOCO_DETECTOR_KIT_DPATH/tpl/Open-GroundingDino}"
export SHITSPOTTER_SAM2_REPO_DPATH="${SHITSPOTTER_SAM2_REPO_DPATH:-$KWCOCO_DETECTOR_KIT_DPATH/tpl/segment-anything-2}"
export SHITSPOTTER_MASKDINO_REPO_DPATH="${SHITSPOTTER_MASKDINO_REPO_DPATH:-$KWCOCO_DETECTOR_KIT_DPATH/tpl/MaskDINO}"
export SHITSPOTTER_YOLOX_REPO_DPATH="${SHITSPOTTER_YOLOX_REPO_DPATH:-$KWCOCO_DETECTOR_KIT_DPATH/tpl/YOLOX}"
export SHITSPOTTER_YOLOV9_REPO_DPATH="${SHITSPOTTER_YOLOV9_REPO_DPATH:-$KWCOCO_DETECTOR_KIT_DPATH/tpl/YOLO-v9}"
