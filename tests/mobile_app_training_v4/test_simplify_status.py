"""Tests for the tile_build_manifest.json simplify_status field.

Three cases the manifest must record durably:

  * simplify succeeds       -> simplify_status=simplified
  * simplify fails, V4_FORCE_SIMPLIFY=0  -> simplify_status=copied_fallback
  * simplify fails, V4_FORCE_SIMPLIFY=1  -> non-zero exit, no manifest

Plus: setup_env.sh PYTHONPATH idempotency (small but nice to pin).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = REPO_ROOT / 'experiments' / 'mobile_app_training_v4'


@pytest.fixture
def fake_v4_root(tmp_path, synthetic_kwcoco):
    """V4_ROOT-shaped scratch dir + a tiny tile bundle ready for simplify."""
    import shutil
    import sys

    v4_root = tmp_path / 'v4_root'
    (v4_root / 'data').mkdir(parents=True)
    # Materialise a tiny tile bundle from synthetic_kwcoco.
    tile_py = V4_DIR / 'tile_kwcoco.py'
    tile_dst = v4_root / 'data' / 'train_tile_g2.kwcoco.zip'
    subprocess.run([
        sys.executable, str(tile_py),
        '--src', str(synthetic_kwcoco),
        '--dst', str(tile_dst),
        '--full_dim', '160', '--tile_grid', '2', '--tile_overlap', '0.20',
        '--tile_output_dim', '96', '--keep_full', 'True',
        '--progress', 'False',
    ], check=True)
    # Reuse the same bundle for vali to keep the fixture small.
    shutil.copyfile(tile_dst, v4_root / 'data' / 'vali_tile_g2.kwcoco.zip')
    return v4_root


def _run_step01(fake_v4_root: Path, *, force_simplify: str = '0',
                fake_simplify_failure: bool = False) -> subprocess.CompletedProcess:
    import sys

    env = os.environ.copy()
    env['V4_ROOT'] = str(fake_v4_root)
    env['V4_TRAIN_FPATH'] = str(fake_v4_root / 'data' / 'train_tile_g2.kwcoco.zip')
    env['V4_VALI_FPATH'] = str(fake_v4_root / 'data' / 'vali_tile_g2.kwcoco.zip')
    env['V4_TEST_FPATH'] = str(fake_v4_root / 'data' / 'train_tile_g2.kwcoco.zip')
    env['V4_TILE_GRID'] = '2'
    env['V4_TILE_OVERLAP'] = '0.20'
    env['V4_TILE_OUTPUT_DIM'] = '96'
    env['V4_RESIZE_MAX_DIM'] = '160'
    env['V4_SIMPLIFY_MIN_INSTANCES'] = '1'
    env['V4_FORCE_SIMPLIFY'] = force_simplify

    if fake_simplify_failure:
        # Wrap the python interpreter to fail when invoked with the
        # simplify_kwcoco module. Other invocations pass through.
        wrap = fake_v4_root / 'pyfail.sh'
        wrap.write_text(
            '#!/bin/bash\n'
            'case " $* " in\n'
            '  *"shitspotter.cli.simplify_kwcoco"*)\n'
            '    echo "fake simplify failure (test)" >&2 ; exit 1 ;;\n'
            f'  *) exec {sys.executable} "$@" ;;\n'
            'esac\n'
        )
        wrap.chmod(0o755)
        env['PYTHON_BIN'] = str(wrap)
    else:
        env['PYTHON_BIN'] = sys.executable

    return subprocess.run(
        ['bash', str(V4_DIR / '01_make_tile_augmented_kwcoco.sh')],
        capture_output=True, text=True, env=env,
    )


def test_simplify_succeeds_marks_simplified(fake_v4_root):
    """Real simplify run on real kwcoco — manifest says simplified."""
    proc = _run_step01(fake_v4_root, force_simplify='0',
                       fake_simplify_failure=False)
    assert proc.returncode == 0, proc.stderr

    manifest = json.loads(
        (fake_v4_root / 'data' / 'tile_build_manifest.json').read_text())
    for split in ('train', 'vali'):
        s = manifest['splits'][split]
        assert s['simplify_status'] == 'simplified', f'{split}: {s}'
        assert s['simplify_error'] == ''


def test_simplify_fails_falls_back_when_force_off(fake_v4_root):
    """Simplify failure with V4_FORCE_SIMPLIFY=0 -> copied_fallback."""
    proc = _run_step01(fake_v4_root, force_simplify='0',
                       fake_simplify_failure=True)
    # Must NOT abort (soft fallback).
    assert proc.returncode == 0, proc.stderr
    # Both .simplified.kwcoco.zip files exist (cp'd from tile bundle).
    assert (fake_v4_root / 'data' / 'train_tile_g2.simplified.kwcoco.zip').exists()
    assert (fake_v4_root / 'data' / 'vali_tile_g2.simplified.kwcoco.zip').exists()
    # Manifest tells the truth.
    manifest = json.loads(
        (fake_v4_root / 'data' / 'tile_build_manifest.json').read_text())
    for split in ('train', 'vali'):
        s = manifest['splits'][split]
        assert s['simplify_status'] == 'copied_fallback', f'{split}: {s}'
        assert 'fake simplify failure' in s['simplify_error']


def test_simplify_fails_aborts_when_force_on(fake_v4_root):
    """Simplify failure with V4_FORCE_SIMPLIFY=1 -> non-zero exit."""
    proc = _run_step01(fake_v4_root, force_simplify='1',
                       fake_simplify_failure=True)
    assert proc.returncode != 0, (
        f'expected non-zero exit; got {proc.returncode}\nSTDERR: {proc.stderr}')


def test_simplify_status_reused_on_idempotent_rerun(fake_v4_root):
    """A second run with existing simplified files records simplify_status=reused."""
    # First run: simplified.
    proc1 = _run_step01(fake_v4_root)
    assert proc1.returncode == 0
    # Second run: same fixture, files exist -> reused.
    proc2 = _run_step01(fake_v4_root)
    assert proc2.returncode == 0
    manifest = json.loads(
        (fake_v4_root / 'data' / 'tile_build_manifest.json').read_text())
    for split in ('train', 'vali'):
        assert manifest['splits'][split]['simplify_status'] == 'reused'


# ---------------------------------------------------------------------------
# PYTHONPATH idempotency
# ---------------------------------------------------------------------------

def test_setup_env_pythonpath_is_idempotent():
    """Sourcing setup_env.sh repeatedly does not duplicate PYTHONPATH entries."""
    proc = subprocess.run(
        ['bash', '-c',
         f'source {V4_DIR / "setup_env.sh"} >/dev/null && '
         f'src1="$PYTHONPATH" && '
         f'source {V4_DIR / "setup_env.sh"} >/dev/null && '
         f'source {V4_DIR / "setup_env.sh"} >/dev/null && '
         f'echo "$src1"; echo "$PYTHONPATH"'],
        capture_output=True, text=True, check=True,
    )
    src1, src3 = proc.stdout.strip().split('\n')
    assert src1 == src3, f'PYTHONPATH grew between source calls:\n  1: {src1}\n  3: {src3}'
