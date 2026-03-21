#!/bin/bash
set -euo pipefail

small_data_canonical_existing_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
    (cd "$path" && pwd -P)
}

small_data_repo_dpath() {
    small_data_canonical_existing_path "${SHITSPOTTER_DPATH:-/home/joncrall/code/shitspotter}"
}

small_data_data_dpath() {
    small_data_canonical_existing_path "${DVC_DATA_DPATH:-/home/joncrall/data/dvc-repos/shitspotter_dvc}"
}

small_data_expt_dpath() {
    small_data_canonical_existing_path "${DVC_EXPT_DPATH:-/home/joncrall/data/dvc-repos/shitspotter_expt_dvc}"
}

small_data_root() {
    printf '%s\n' "$(small_data_expt_dpath)/small_data_tuning"
}

small_data_cohorts_root() {
    printf '%s\n' "$(small_data_root)/cohorts"
}

small_data_runs_root() {
    printf '%s\n' "$(small_data_root)/runs"
}

small_data_manifest_path() {
    local cohort_dpath="$1"
    printf '%s\n' "$cohort_dpath/cohort_manifest.json"
}

small_data_require_cohort() {
    local cohort_dpath="$1"
    local manifest_fpath
    manifest_fpath="$(small_data_manifest_path "$cohort_dpath")"
    if [ ! -f "$manifest_fpath" ]; then
        echo "Expected cohort manifest missing: $manifest_fpath" >&2
        exit 1
    fi
}

small_data_export_cohort_env() {
    local cohort_dpath="$1"
    local manifest_fpath
    manifest_fpath="$(small_data_manifest_path "$cohort_dpath")"
    small_data_require_cohort "$cohort_dpath"
    "${PYTHON_BIN:-python}" - "$manifest_fpath" <<'PY'
import json
import pathlib
import shlex
import sys

manifest_fpath = pathlib.Path(sys.argv[1])
data = json.loads(manifest_fpath.read_text())
subsets = data['subsets']

def emit(name, value):
    print(f'export {name}={shlex.quote(str(value))}')

emit('SMALL_DATA_COHORT_NAME', data['cohort_name'])
emit('SMALL_DATA_COHORT_DPATH', manifest_fpath.parent)
emit('SMALL_DATA_COHORT_MANIFEST_FPATH', manifest_fpath)
emit('SMALL_DATA_SELECTOR_METHOD', data['selector']['method'])
emit('SMALL_DATA_SELECTOR_SEED', data['selector']['seed'])

for split_name in ['train', 'vali', 'test']:
    split = subsets[split_name]
    prefix = f'SMALL_DATA_{split_name.upper()}'
    emit(f'{prefix}_KWCOCO_FPATH', split['kwcoco_fpath'])
    emit(f'{prefix}_MSCOCO_FPATH', split['mscoco_fpath'])
    emit(f'{prefix}_NUM_IMAGES', split['stats']['num_images'])
    emit(f'{prefix}_NUM_ANNOTS', split['stats']['num_annotations'])
PY
}

small_data_cohort_from_name() {
    local cohort_name="$1"
    printf '%s\n' "$(small_data_cohorts_root)/$cohort_name"
}

small_data_default_train_fpath() {
    printf '%s\n' "$(small_data_data_dpath)/train_imgs5747_1e73d54f.kwcoco.zip"
}

small_data_default_vali_fpath() {
    printf '%s\n' "$(small_data_data_dpath)/vali_imgs691_99b22ad0.kwcoco.zip"
}

small_data_default_test_fpath() {
    printf '%s\n' "$(small_data_data_dpath)/test_imgs121_6cb3b6ff.kwcoco.zip"
}
