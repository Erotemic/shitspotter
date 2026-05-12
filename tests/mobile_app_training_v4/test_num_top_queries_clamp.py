"""Tests for the num_top_queries clamp in _train_deimv2_variant.sh.

Q5 in dev/benchmark-candidates/pipeline-bootstrap-questions.md:
num_top_queries must be <= num_queries × num_classes, else the DEIMv2
postprocessor's `torch.topk(scores.flatten(1), num_top_queries)` raises
`RuntimeError: selected index k out of range` in the first val pass.
"""
from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = REPO_ROOT / 'experiments' / 'mobile_app_training_v4'
TRAIN_SH = V4_DIR / '_train_deimv2_variant.sh'


def _resolve_clamp(variant: str) -> tuple[int, int]:
    """Run the variant-dispatch + clamp block as a sub-shell and read
    back (V4_NUM_QUERIES, V4_NUM_TOP_QUERIES)."""
    text = TRAIN_SH.read_text()
    # Extract everything from the case-statement header to just after
    # the clamp's closing `fi`. Both markers are stable strings in the
    # script — if they move, the assertion below will surface the move
    # clearly.
    start_marker = '# Per-variant num_queries (matches the upstream'
    end_marker_re = re.compile(r'\nfi\n', re.MULTILINE)
    start = text.find(start_marker)
    assert start != -1, 'num_queries block start marker moved'
    body = text[start:]
    # Stop at the FIRST `fi` after the clamp's `if [...]; then ... fi`.
    # The block contains a case statement (no `fi`) and a single `if`
    # for the clamp.
    m = end_marker_re.search(body)
    assert m is not None, 'num_queries clamp end marker not found'
    block = body[:m.end()]

    shim = textwrap.dedent(f'''
        set -e
        V4_VARIANT={variant!r}
        {block}
        echo "V4_NUM_QUERIES=$V4_NUM_QUERIES"
        echo "V4_NUM_TOP_QUERIES=$V4_NUM_TOP_QUERIES"
    ''')
    res = subprocess.run(['bash', '-c', shim], capture_output=True, text=True,
                         check=True)
    vals = {}
    for line in res.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            vals[k.strip()] = int(v.strip())
    return vals['V4_NUM_QUERIES'], vals['V4_NUM_TOP_QUERIES']


@pytest.mark.parametrize('variant,expected_nq,expected_ntq', [
    ('deimv2_atto',  100, 100),
    ('deimv2_femto', 150, 150),
    ('deimv2_pico',  200, 200),  # the bug cell: pre-fix this was 300 > 200×1
    ('deimv2_n',     300, 300),
    ('deimv2_s',     300, 300),
    ('deimv2_m',     300, 300),
    ('deimv2_l',     300, 300),
    ('deimv2_x',     300, 300),
])
def test_num_top_queries_clamp_per_variant(variant, expected_nq, expected_ntq):
    nq, ntq = _resolve_clamp(variant)
    assert nq == expected_nq, f'{variant}: num_queries should be {expected_nq}, got {nq}'
    assert ntq == expected_ntq, f'{variant}: num_top_queries should be {expected_ntq}, got {ntq}'
    # The shitspotter invariant: ntq <= nq × num_classes (num_classes=1).
    assert ntq <= nq, (
        f'{variant}: clamp invariant violated — num_top_queries={ntq} '
        f'> num_queries={nq}'
    )


def test_unknown_variant_falls_through_to_300():
    """An unrecognised variant should land on the default branch (300)
    rather than leaving V4_NUM_QUERIES unset and exploding the trainer."""
    nq, ntq = _resolve_clamp('deimv2_unknown_xl')
    assert nq == 300
    assert ntq == 300
