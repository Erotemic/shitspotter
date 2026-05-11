"""Tests for the candidate_kind (smoke|real) selection rule.

A v4_mock_tiny candidate must:

 * be tagged candidate_kind=smoke in policy.json (written by v4_mock.py)
   and in the eligibility_manifest output;
 * be EXCLUDED from the default winner-selection pool;
 * be INCLUDED when --include_smoke_models is given;
 * be the ONLY candidate considered when --smoke_only is given.

A deimv2_* candidate must:

 * be tagged candidate_kind=real (or fall back to "real" via inference
   when the field is missing in older policy.json files);
 * be the only kind considered by the default winner selection.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = REPO_ROOT / 'experiments' / 'mobile_app_training_v4'


def _make_workdir(v4_root: Path, candidate_id: str, *, kind: str = 'real',
                  variant: str = 'deimv2_n', test_ap=0.5, mean_ms=10.0,
                  export_h=320, export_w=320, train_policy='fixed',
                  include_kind_field=True) -> Path:
    """Synthesise a per-cell workdir + eval result + bench."""
    workdir = v4_root / 'runs' / candidate_id
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'generated_configs').mkdir(exist_ok=True)
    (workdir / 'export').mkdir(exist_ok=True)
    (workdir / 'best_stg2.pth').write_bytes(b'\x00')
    (workdir / 'export' / f'{variant}_h{export_h}_w{export_w}.onnx').write_bytes(b'\x00')

    policy = {
        'candidate_id': candidate_id,
        'variant': variant,
        'export_input_h': int(export_h),
        'export_input_w': int(export_w),
        'train_resolution_policy': train_policy,
        'effective_train_scales': [int(export_h)],
        'effective_train_scale_min': int(export_h),
        'effective_train_scale_max': int(export_h),
        'tile_training_policy': 'tile_g2_overlap0.20_out320',
    }
    if include_kind_field:
        policy['candidate_kind'] = kind
    (workdir / 'policy.json').write_text(json.dumps(policy))

    eval_dpath = v4_root / 'eval' / candidate_id / 'eval'
    eval_dpath.mkdir(parents=True, exist_ok=True)
    (eval_dpath / 'detect_metrics.json').write_text(json.dumps({
        'area_range=all,iou_thresh=0.5': {
            'nocls_measures': {'ap': float(test_ap)},
        },
    }))

    timings = [float(mean_ms)] * 5
    (workdir / 'export' / f'{variant}_h{export_h}_w{export_w}.bench.json').write_text(
        json.dumps({'mean_ms': float(mean_ms),
                    'timings_ms': timings,
                    'fps': 1000.0 / mean_ms})
    )
    return workdir


def _aggregate(v4_root: Path, *, include_smoke=False, smoke_only=False) -> dict:
    out_json = v4_root / 'manifest.json'
    cmd = [
        sys.executable, str(V4_DIR / 'eligibility_manifest.py'),
        '--auto', '--v4_root', str(v4_root),
        '--max_desktop_ms', '80',
        '--out', str(v4_root / 'manifest.tsv'),
        '--out_json', str(out_json),
    ]
    if include_smoke:
        cmd.append('--include_smoke_models')
    if smoke_only:
        cmd.append('--smoke_only')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return {
        'rows': json.loads(out_json.read_text()),
        'stdout': proc.stdout,
    }


def test_default_pool_excludes_smoke(v4_workspace):
    """A v4_mock_tiny cell with high AP must NOT win the default pool."""
    _make_workdir(v4_workspace, 'real_a',
                  kind='real', variant='deimv2_n', test_ap=0.40)
    _make_workdir(v4_workspace, 'mock_a',
                  kind='smoke', variant='v4_mock_tiny', test_ap=0.95)

    result = _aggregate(v4_workspace)
    assert 'host-promising winner' in result['stdout']
    # The default winner is the real one (lower AP) because smoke is
    # excluded from the default pool.
    assert 'real_a' in result['stdout']
    assert 'mock_a' not in result['stdout'].split('host-promising winner')[1]
    assert 'smoke candidate(s) excluded' in result['stdout']


def test_include_smoke_lets_mock_win(v4_workspace):
    """With --include_smoke_models, the higher-AP mock can win the pool."""
    _make_workdir(v4_workspace, 'real_a',
                  kind='real', variant='deimv2_n', test_ap=0.40)
    _make_workdir(v4_workspace, 'mock_a',
                  kind='smoke', variant='v4_mock_tiny', test_ap=0.95)

    result = _aggregate(v4_workspace, include_smoke=True)
    # Now the higher-AP smoke wins.
    after = result['stdout'].split('host-promising winner')[1]
    assert 'mock_a' in after


def test_smoke_only_picks_smoke(v4_workspace):
    """With --smoke_only, only smoke candidates participate."""
    _make_workdir(v4_workspace, 'real_a',
                  kind='real', variant='deimv2_n', test_ap=0.99)
    _make_workdir(v4_workspace, 'mock_a',
                  kind='smoke', variant='v4_mock_tiny', test_ap=0.30)
    _make_workdir(v4_workspace, 'mock_b',
                  kind='smoke', variant='v4_mock_tiny', test_ap=0.50)

    result = _aggregate(v4_workspace, smoke_only=True)
    after = result['stdout'].split('host-promising winner')[1]
    # mock_b (0.50) wins over mock_a (0.30); real_a is invisible.
    assert 'mock_b' in after
    assert 'real_a' not in after


def test_kind_inferred_from_variant_when_missing(v4_workspace):
    """Older policy.json files without candidate_kind still work — kind
    is inferred from the variant prefix (v4_mock* -> smoke, else real)."""
    _make_workdir(v4_workspace, 'old_real',
                  variant='deimv2_n', test_ap=0.40,
                  include_kind_field=False)
    _make_workdir(v4_workspace, 'old_smoke',
                  variant='v4_mock_tiny', test_ap=0.95,
                  include_kind_field=False)

    result = _aggregate(v4_workspace)
    rows = {r['candidate_id']: r for r in result['rows']}
    assert rows['old_real']['candidate_kind'] == 'real'
    assert rows['old_smoke']['candidate_kind'] == 'smoke'
    # Default pool excludes smoke even when inferred.
    after = result['stdout'].split('host-promising winner')[1]
    assert 'old_real' in after
    assert 'old_smoke' not in after


def test_candidate_kind_in_manifest_columns(v4_workspace):
    """The TSV/JSON output carries the candidate_kind column."""
    _make_workdir(v4_workspace, 'a', kind='real', variant='deimv2_n')
    result = _aggregate(v4_workspace)
    assert 'candidate_kind' in result['rows'][0]
    tsv_text = (v4_workspace / 'manifest.tsv').read_text()
    assert 'candidate_kind' in tsv_text.splitlines()[0]


def test_v4_mock_train_writes_smoke_kind(synthetic_kwcoco, tmp_path):
    """v4_mock train writes candidate_kind=smoke into policy.json."""
    workdir = tmp_path / 'mock_run'
    workdir.mkdir()
    v4_mock_py = V4_DIR / 'v4_mock.py'

    subprocess.run([
        sys.executable, str(v4_mock_py), 'train',
        '--train_kwcoco', str(synthetic_kwcoco),
        '--vali_kwcoco', str(synthetic_kwcoco),
        '--workdir', str(workdir),
        '--input_h', '128', '--input_w', '128',
        '--num_epochs', '1', '--batch_size', '2',
        '--num_queries', '4', '--lr', '1.0', '--seed', '0',
    ], check=True)

    policy = json.loads((workdir / 'policy.json').read_text())
    assert policy['candidate_kind'] == 'smoke'
