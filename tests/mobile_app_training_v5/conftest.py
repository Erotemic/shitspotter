"""Shared fixtures for mobile_app_training_v5 tests.

Reuses the synthetic_kwcoco fixture from v4 (hand-built tiny kwcoco
bundle), and adds a tile-pool fixture that runs v5_tile.py once per
session.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_DIR = REPO_ROOT / 'experiments' / 'mobile_app_training_v5'
V4_TEST_DIR = REPO_ROOT / 'tests' / 'mobile_app_training_v4'

# Re-use v4 test machinery (synthetic_kwcoco + v4_workspace fixtures).
sys.path.insert(0, str(V4_TEST_DIR))
from conftest import synthetic_kwcoco, v4_workspace  # noqa: F401,E402

# Make v5 modules importable as flat names.
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))


@pytest.fixture(scope='session')
def v5_tile_bundle(synthetic_kwcoco, tmp_path_factory):
    """Run v5_tile.py once at session scope; return the output kwcoco."""
    import subprocess
    out_dpath = tmp_path_factory.mktemp('v5_tile_bundle')
    dst = out_dpath / 'tiles.kwcoco.zip'
    proc = subprocess.run([
        sys.executable, str(V5_DIR / 'v5_tile.py'),
        '--src', str(synthetic_kwcoco),
        '--dst', str(dst),
        '--tile_size', '128',
        '--source_scales', '1.0,0.5,0.25',
        '--stride_frac', '0.5',
        '--min_gt_area_frac', '0.005',
        '--keep_negative', 'True',
        '--progress', 'False',
        '--jpeg_quality', '85',
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return dst
