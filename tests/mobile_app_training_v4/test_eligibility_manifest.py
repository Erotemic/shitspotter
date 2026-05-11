"""Tests for eligibility_manifest.py — the per-cell aggregator + winner picker.

The state machine is the load-bearing part of v4. These tests pin down
the four eligibility classes at every transition and the model-ID
canonicalisation contract that lets the manifest's `phone_model_id`
match the modelspec sidecar's `modelId`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_workdir(v4_root: Path, candidate_id: str, *,
                  test_ap=None, mean_ms=None, with_ckpt=True, with_onnx=True,
                  variant='v4_mock_tiny', export_h=256, export_w=256,
                  train_policy='fixed') -> Path:
    """Synthesise an on-disk run + eval layout the manifest can read."""
    workdir = v4_root / 'runs' / candidate_id
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'generated_configs').mkdir(exist_ok=True)
    (workdir / 'export').mkdir(exist_ok=True)
    if with_ckpt:
        (workdir / 'best_stg2.pth').write_bytes(b'\x00')
    if with_onnx:
        (workdir / 'export' / f'{variant}_h{export_h}_w{export_w}.onnx').write_bytes(b'\x00')

    policy = {
        'candidate_id': candidate_id,
        'variant': variant,
        'export_input_h': int(export_h),
        'export_input_w': int(export_w),
        'train_resolution_policy': train_policy,
        'requested_train_resolution_min': int(export_h),
        'requested_train_resolution_max': int(export_h),
        'effective_train_scales': [int(export_h)],
        'effective_train_scale_min': int(export_h),
        'effective_train_scale_max': int(export_h),
        'tile_training_policy': 'tile_g2_overlap0.20_out320',
    }
    (workdir / 'policy.json').write_text(json.dumps(policy))

    if test_ap is not None:
        eval_dpath = v4_root / 'eval' / candidate_id / 'eval'
        eval_dpath.mkdir(parents=True, exist_ok=True)
        (eval_dpath / 'detect_metrics.json').write_text(json.dumps({
            'area_range=all,iou_thresh=0.5': {
                'nocls_measures': {'ap': float(test_ap)},
            },
        }))

    if mean_ms is not None:
        timings = [float(mean_ms)] * 5
        (workdir / 'export' / f'{variant}_h{export_h}_w{export_w}.bench.json').write_text(
            json.dumps({'mean_ms': float(mean_ms), 'timings_ms': timings, 'fps': 1000.0 / mean_ms})
        )
    return workdir


def _aggregate(v4_root: Path, max_desktop_ms=80.0, pixel5_index=None,
               allow_missing_desktop_bench=False) -> dict:
    """Run eligibility_manifest in --auto mode against v4_root, return rows."""
    import subprocess
    import sys
    out_json = v4_root / 'manifest.json'
    out_tsv = v4_root / 'manifest.tsv'
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[2]
            / 'experiments' / 'mobile_app_training_v4' / 'eligibility_manifest.py'),
        '--auto',
        '--v4_root', str(v4_root),
        '--max_desktop_ms', str(max_desktop_ms),
        '--out', str(out_tsv),
        '--out_json', str(out_json),
    ]
    if allow_missing_desktop_bench:
        cmd.append('--allow_missing_desktop_bench')
    if pixel5_index is not None:
        cmd += ['--pixel5_index', str(pixel5_index)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out_json.read_text())


def test_class_NOT_READY_when_no_checkpoint(v4_workspace):
    _make_workdir(v4_workspace, 'a', with_ckpt=False, test_ap=0.5, mean_ms=10.0)
    rows = _aggregate(v4_workspace)
    assert rows[0]['eligibility_class'] == 'NOT_READY'
    assert rows[0]['status'] == 'no_checkpoint'


def test_class_NOT_READY_when_no_onnx(v4_workspace):
    _make_workdir(v4_workspace, 'a', with_onnx=False, test_ap=0.5, mean_ms=10.0)
    rows = _aggregate(v4_workspace)
    assert rows[0]['eligibility_class'] == 'NOT_READY'
    assert rows[0]['status'] == 'no_onnx'


def test_class_NOT_READY_when_no_eval(v4_workspace):
    _make_workdir(v4_workspace, 'a', test_ap=None, mean_ms=10.0)
    rows = _aggregate(v4_workspace)
    assert rows[0]['eligibility_class'] == 'NOT_READY'
    assert rows[0]['status'] == 'no_eval'


def test_class_NOT_READY_when_bench_missing_default(v4_workspace):
    """The reviewer-mandated rule: missing desktop bench -> NOT_READY by default."""
    _make_workdir(v4_workspace, 'a', test_ap=0.5, mean_ms=None)
    rows = _aggregate(v4_workspace)
    assert rows[0]['eligibility_class'] == 'NOT_READY'
    # Reason must explicitly mention the missing bench.
    assert any('no desktop benchmark' in r for r in rows[0]['reasons'])


def test_class_HOST_PROMISING_when_bench_missing_with_override(v4_workspace):
    _make_workdir(v4_workspace, 'a', test_ap=0.5, mean_ms=None)
    rows = _aggregate(v4_workspace, allow_missing_desktop_bench=True)
    assert rows[0]['eligibility_class'] == 'HOST_PROMISING'


def test_class_PHONE_INELIGIBLE_when_desktop_too_slow(v4_workspace):
    _make_workdir(v4_workspace, 'a', test_ap=0.5, mean_ms=200.0)
    rows = _aggregate(v4_workspace, max_desktop_ms=80.0)
    assert rows[0]['eligibility_class'] == 'PHONE_INELIGIBLE'


def test_class_HOST_PROMISING_when_all_host_gates_pass(v4_workspace):
    _make_workdir(v4_workspace, 'a', test_ap=0.5, mean_ms=10.0)
    rows = _aggregate(v4_workspace, max_desktop_ms=80.0)
    assert rows[0]['eligibility_class'] == 'HOST_PROMISING'


def test_class_PHONE_ELIGIBLE_with_pixel5_pass(v4_workspace, tmp_path):
    _make_workdir(v4_workspace, 'a', test_ap=0.5, mean_ms=10.0)
    p5 = tmp_path / 'p5.tsv'
    p5.write_text('candidate_id\tlatency_ms\tfps\na\t60\t16.6\n')
    rows = _aggregate(v4_workspace, pixel5_index=p5, max_desktop_ms=80.0)
    assert rows[0]['eligibility_class'] == 'PHONE_ELIGIBLE'
    assert rows[0]['pixel5_eligible'] == 'yes'


def test_class_PHONE_INELIGIBLE_with_pixel5_below_gate(v4_workspace, tmp_path):
    _make_workdir(v4_workspace, 'a', test_ap=0.5, mean_ms=10.0)
    p5 = tmp_path / 'p5.tsv'
    p5.write_text('candidate_id\tlatency_ms\tfps\na\t300\t3.3\n')
    rows = _aggregate(v4_workspace, pixel5_index=p5)
    assert rows[0]['eligibility_class'] == 'PHONE_INELIGIBLE'


def test_phone_model_id_format(v4_workspace):
    _make_workdir(v4_workspace, 'a', variant='v4_mock_tiny', export_h=320, export_w=320,
                  train_policy='multiscale_256_416', test_ap=0.5, mean_ms=10.0)
    rows = _aggregate(v4_workspace)
    assert rows[0]['phone_model_id'] == 'shitspotter-v4_mock_tiny-h320w320-multiscale_256_416'


def test_winner_is_max_AP_among_eligible(v4_workspace):
    _make_workdir(v4_workspace, 'a', test_ap=0.20, mean_ms=10.0)
    _make_workdir(v4_workspace, 'b', test_ap=0.55, mean_ms=10.0)
    _make_workdir(v4_workspace, 'c', test_ap=0.99, mean_ms=200.0)  # PHONE_INELIGIBLE
    rows = _aggregate(v4_workspace, max_desktop_ms=80.0)
    by_id = {r['candidate_id']: r for r in rows}
    assert by_id['a']['eligibility_class'] == 'HOST_PROMISING'
    assert by_id['b']['eligibility_class'] == 'HOST_PROMISING'
    assert by_id['c']['eligibility_class'] == 'PHONE_INELIGIBLE'
    # The HOST_PROMISING winner has the higher AP.
    promising = [r for r in rows if r['eligibility_class'] == 'HOST_PROMISING']
    winner = max(promising, key=lambda r: r['test_ap_simplified'])
    assert winner['candidate_id'] == 'b'
