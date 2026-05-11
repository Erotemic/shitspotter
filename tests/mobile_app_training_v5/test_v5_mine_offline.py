"""Offline tests for v5_mine.py.

The miner needs a real DEIMv2 checkpoint to actually score, which we
don't have on a clean test env. These tests cover:

  * CLI signature + import (catches the kinds of bugs that bit v4_mock).
  * The `_score_tile` interface contract.
  * Module-level guard: rejecting a missing checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_DIR = REPO_ROOT / 'experiments' / 'mobile_app_training_v5'


def test_v5_mine_help_imports():
    """The miner module imports cleanly and exposes a --help that mentions
    every documented flag."""
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(V5_DIR / 'v5_mine.py'), '--help'],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout
    for required_flag in (
            '--neg_kwcoco', '--workdir', '--dst',
            '--score_thresh', '--max_hard_per_round',
            '--device', '--batch_size'):
        assert required_flag in text, f'missing flag {required_flag} in --help'


def test_v5_mine_rejects_missing_workdir(tmp_path):
    """Mining against a non-existent workdir must fail loudly."""
    import subprocess
    neg = tmp_path / 'neg.kwcoco.zip'
    # Minimal valid kwcoco.
    import kwcoco
    d = kwcoco.CocoDataset(); d.add_category(name='poop')
    d.fpath = str(neg); d.dump()

    proc = subprocess.run([
        sys.executable, str(V5_DIR / 'v5_mine.py'),
        '--neg_kwcoco', str(neg),
        '--workdir', str(tmp_path / 'does_not_exist'),
        '--dst', str(tmp_path / 'out.kwcoco.zip'),
        '--device', 'cpu',
        '--progress', 'False',
    ], capture_output=True, text=True)
    # Either crashes with FileNotFoundError at _load_model, or with an
    # ImportError when geowatch's preimport bombs without GDAL on a lean
    # env. Either way, non-zero exit and a recognisable error message.
    assert proc.returncode != 0
    msg = (proc.stderr + proc.stdout).lower()
    assert any(s in msg for s in (
        'best_stg2.pth', 'no such file', 'no module', 'filenotfounderror'))


def test_v5_mine_score_tile_returns_float():
    """The _score_tile helper returns a float scalar, never None or tensor."""
    # We test via monkeypatch: replace _load_model output with a tiny
    # fake model that returns deterministic scores.
    import sys as _sys
    import types
    import numpy as np
    import v5_mine
    import torch

    class _FakeModel(torch.nn.Module):
        def __init__(self, val):
            super().__init__()
            self.val = float(val)

        def forward(self, im, sz):
            n = im.shape[0]
            labels = torch.zeros(n, 3, dtype=torch.long)
            boxes = torch.zeros(n, 3, 4)
            scores = torch.full((n, 3), self.val)
            return labels, boxes, scores

    class _Identity:
        """Stand-in for the torchvision transform — we just want a tensor."""
        def __call__(self, pil):
            arr = np.array(pil)
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)

    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    score = v5_mine._score_tile(_FakeModel(0.42), _Identity(), img, 'cpu')
    assert isinstance(score, float)
    assert score == pytest.approx(0.42, abs=1e-3)


def test_v5_mine_score_tile_empty_returns_zero():
    """If the model returns an empty score tensor, _score_tile returns 0.0."""
    import numpy as np
    import v5_mine
    import torch

    class _NoDetections(torch.nn.Module):
        def forward(self, im, sz):
            n = im.shape[0]
            return (torch.zeros(n, 0, dtype=torch.long),
                    torch.zeros(n, 0, 4),
                    torch.zeros(n, 0))

    class _Identity:
        def __call__(self, pil):
            arr = np.array(pil)
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)

    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    score = v5_mine._score_tile(_NoDetections(), _Identity(), img, 'cpu')
    assert score == 0.0
