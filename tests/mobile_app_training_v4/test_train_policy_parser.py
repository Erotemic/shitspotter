"""Tests for the train-resolution policy parser in _train_deimv2_variant.sh.

The policy parser is a bash case statement, so we exercise it by
sourcing the shell script in a controlled environment and reading
the resolved variables back. This catches regressions in:

  * fixed -> single-scale fallback
  * multiscale -> ±25% band centred on max(H, W)
  * multiscale_<S> -> centred on <S>
  * multiscale_<lo>_<hi> -> base picked so band ≈ [lo, hi]

Failures here are tightly scoped — the policy translation lives in
the shell library and is referenced everywhere downstream.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = REPO_ROOT / 'experiments' / 'mobile_app_training_v4'
TRAIN_SH = V4_DIR / '_train_deimv2_variant.sh'


def _run_policy_block(policy: str, input_h: int = 320, input_w: int = 320,
                      multiscale_repeat: int = 12) -> dict:
    """Source the policy-translation block and return the resolved values.

    We extract just the policy block (between `# Translate the policy
    string` and the loud-banner echo) and run it in a tiny bash shim
    so the test stays fast and doesn't exercise unrelated code paths.
    """
    text = TRAIN_SH.read_text()
    start_marker = '# Translate the policy string into (multiscale_repeat'
    end_marker = '# Loud banner.'
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    assert start != -1 and end != -1, 'policy-translation block markers moved'
    block = text[start:end]

    # Stub out the python helper that computes effective scales — for
    # this unit test we only care about the bash-level decisions.
    shim = f"""
set -e
INPUT_H={input_h}
INPUT_W={input_w}
V4_TRAIN_POLICY={policy!r}
V4_MULTISCALE_REPEAT={multiscale_repeat}
PYTHON_BIN=:                          # no-op; the python heredoc is downstream
{block}
echo "MS_REPEAT=$MS_REPEAT"
echo "MS_BASE=$MS_BASE"
echo "REQUESTED_MIN=${{REQUESTED_MIN:-}}"
echo "REQUESTED_MAX=${{REQUESTED_MAX:-}}"
"""
    proc = subprocess.run(['bash', '-c', shim],
                          capture_output=True, text=True, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def test_policy_fixed_no_jitter():
    out = _run_policy_block('fixed', input_h=320, input_w=320)
    assert out['MS_REPEAT'] == '0'
    assert out['MS_BASE'] == '320'
    assert out['REQUESTED_MIN'] == '320'
    assert out['REQUESTED_MAX'] == '320'


def test_policy_multiscale_default_band_around_export():
    out = _run_policy_block('multiscale', input_h=320, input_w=320)
    assert int(out['MS_REPEAT']) == 12
    assert out['MS_BASE'] == '320'
    # ±25% band of 320 is [240, 400].
    assert out['REQUESTED_MIN'] == '240'
    assert out['REQUESTED_MAX'] == '400'


def test_policy_multiscale_explicit_centre():
    out = _run_policy_block('multiscale_416', input_h=320, input_w=320)
    assert out['MS_BASE'] == '416'
    # ±25% band of 416 is [312, 520].
    assert out['REQUESTED_MIN'] == '312'
    assert out['REQUESTED_MAX'] == '520'


def test_policy_multiscale_lo_hi():
    out = _run_policy_block('multiscale_320_512', input_h=320, input_w=320)
    # Base = (320 + 512) / 2 = 416, rounded to 32 = 416.
    assert out['MS_BASE'] == '416'
    assert out['REQUESTED_MIN'] == '320'
    assert out['REQUESTED_MAX'] == '512'


def test_policy_multiscale_lo_hi_rounding_to_32():
    # (300 + 500) / 2 = 400, +16 = 416, // 32 * 32 = 416.
    out = _run_policy_block('multiscale_300_500', input_h=320, input_w=320)
    assert out['MS_BASE'] == '416'
    # Requested bounds are preserved verbatim, even when the rounded
    # base means effective scales spill outside [lo, hi]. The loud
    # banner downstream warns the user.
    assert out['REQUESTED_MIN'] == '300'
    assert out['REQUESTED_MAX'] == '500'


def test_policy_unknown_string_errors():
    with pytest.raises(subprocess.CalledProcessError):
        _run_policy_block('multiscale-typo-not-an-underscore',
                          input_h=320, input_w=320)


def test_policy_export_uses_max_dimension_when_rectangular():
    # H > W: export_long should be H; band centres on max(H, W).
    out = _run_policy_block('multiscale', input_h=640, input_w=480)
    assert out['MS_BASE'] == '640'
    # ±25% of 640 is [480, 800].
    assert out['REQUESTED_MIN'] == '480'
    assert out['REQUESTED_MAX'] == '800'
