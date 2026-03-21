#!/bin/bash
set -euo pipefail

# Convenience wrapper to launch several small-data experiment families on the
# same cohort. This is intentionally conservative: it runs one family at a time
# so logs and output directories remain easy to inspect.

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"

COHORT_NAME="${COHORT_NAME:-}"
COHORT_DPATH="${COHORT_DPATH:-}"
MODELS=(${MODELS:-maskrcnn yolo grounding_dino deimv2 deimv2_sam2})

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cohort_name) COHORT_NAME="$2"; shift 2 ;;
        --cohort_dpath) COHORT_DPATH="$2"; shift 2 ;;
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$COHORT_NAME" ] && [ -z "$COHORT_DPATH" ]; then
    echo "Specify --cohort_name or --cohort_dpath" >&2
    exit 1
fi

COMMON_ARGS=()
if [ -n "$COHORT_NAME" ]; then
    COMMON_ARGS+=(--cohort_name "$COHORT_NAME")
fi
if [ -n "$COHORT_DPATH" ]; then
    COMMON_ARGS+=(--cohort_dpath "$COHORT_DPATH")
fi

for model in "${MODELS[@]}"; do
    case "$model" in
        maskrcnn)
            bash "$_small_data_script_dpath/run_maskrcnn_small_experiment.sh" "${COMMON_ARGS[@]}"
            ;;
        yolo)
            bash "$_small_data_script_dpath/run_yolo_small_experiment.sh" "${COMMON_ARGS[@]}"
            ;;
        grounding_dino)
            bash "$_small_data_script_dpath/run_grounding_dino_small_experiment.sh" "${COMMON_ARGS[@]}"
            ;;
        deimv2)
            bash "$_small_data_script_dpath/run_deimv2_small_experiment.sh" "${COMMON_ARGS[@]}"
            ;;
        deimv2_sam2)
            bash "$_small_data_script_dpath/run_deimv2_sam2_small_experiment.sh" "${COMMON_ARGS[@]}"
            ;;
        *)
            echo "Unknown model family: $model" >&2
            exit 1
            ;;
    esac
done
