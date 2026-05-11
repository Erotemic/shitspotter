"""Tests for v5_merge.py — union positives + negatives into a round kwcoco.

Round-0 semantics:
  pos + random sample of neg_kwcoco at ratio neg_over_pos.

Round-N>0 semantics:
  pos + all of (hard) neg_kwcoco.

These are exercised by passing different (neg_over_pos, neg pool size).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_DIR = REPO_ROOT / 'experiments' / 'mobile_app_training_v5'


def _split_pos_neg(v5_tile_bundle, tmp_path):
    """Split the session-scope tile bundle into pos-only + neg-only kwcoco files."""
    import kwcoco
    src = kwcoco.CocoDataset.coerce(str(v5_tile_bundle))
    pos_gids = [img['id'] for img in src.images().objs
                if img.get('tile_role') == 'positive']
    neg_gids = [img['id'] for img in src.images().objs
                if img.get('tile_role') == 'negative']
    if not pos_gids or not neg_gids:
        pytest.skip(f'synthetic bundle has pos={len(pos_gids)} neg={len(neg_gids)};'
                    ' need both')
    pos = src.subset(pos_gids)
    pos.fpath = str(tmp_path / 'pos.kwcoco.zip'); pos.dump()
    neg = src.subset(neg_gids)
    neg.fpath = str(tmp_path / 'neg.kwcoco.zip'); neg.dump()
    return Path(pos.fpath), Path(neg.fpath), len(pos_gids), len(neg_gids)


def _run_merge(pos_fpath, neg_fpath, dst_fpath, *, neg_over_pos=3.0,
               seed=0, round_index=0):
    proc = subprocess.run([
        sys.executable, str(V5_DIR / 'v5_merge.py'),
        '--pos_kwcoco', str(pos_fpath),
        '--neg_kwcoco', str(neg_fpath),
        '--dst', str(dst_fpath),
        '--neg_over_pos', str(neg_over_pos),
        '--seed', str(seed),
        '--round_index', str(round_index),
    ], capture_output=True, text=True, check=True)
    return proc


def test_merge_round0_subsamples_negs(v5_tile_bundle, tmp_path):
    """Round 0 with neg_over_pos=2.0 should hit roughly 2x pos count."""
    import kwcoco
    pos_fpath, neg_fpath, n_pos, n_neg = _split_pos_neg(v5_tile_bundle, tmp_path)
    dst = tmp_path / 'round0.kwcoco.zip'
    _run_merge(pos_fpath, neg_fpath, dst, neg_over_pos=2.0, round_index=0)

    out = kwcoco.CocoDataset.coerce(str(dst))
    n_out_pos = sum(1 for img in out.images().objs
                    if img.get('tile_role') == 'positive')
    n_out_neg = sum(1 for img in out.images().objs
                    if img.get('tile_role') == 'negative')
    assert n_out_pos == n_pos
    assert n_out_neg == min(int(2 * n_pos), n_neg)


def test_merge_round_n_keeps_all_negs(v5_tile_bundle, tmp_path):
    """neg_over_pos=0 should keep the full neg pool (round N>0 semantics)."""
    import kwcoco
    pos_fpath, neg_fpath, n_pos, n_neg = _split_pos_neg(v5_tile_bundle, tmp_path)
    dst = tmp_path / 'roundn.kwcoco.zip'
    _run_merge(pos_fpath, neg_fpath, dst, neg_over_pos=0, round_index=1)

    out = kwcoco.CocoDataset.coerce(str(dst))
    n_out_neg = sum(1 for img in out.images().objs
                    if img.get('tile_role') == 'negative')
    assert n_out_neg == n_neg


def test_merge_carries_pos_annotations(v5_tile_bundle, tmp_path):
    """Positive tiles' annotations survive the merge."""
    import kwcoco
    pos_fpath, neg_fpath, _, _ = _split_pos_neg(v5_tile_bundle, tmp_path)
    dst = tmp_path / 'merge.kwcoco.zip'
    _run_merge(pos_fpath, neg_fpath, dst, neg_over_pos=1.0)

    pos = kwcoco.CocoDataset.coerce(str(pos_fpath))
    out = kwcoco.CocoDataset.coerce(str(dst))
    # Total annotations preserved (no negatives have anns by definition).
    assert out.n_annots == pos.n_annots


def test_merge_seed_determinism(v5_tile_bundle, tmp_path):
    """Same seed -> same neg subsample."""
    import kwcoco
    pos_fpath, neg_fpath, _, _ = _split_pos_neg(v5_tile_bundle, tmp_path)
    dst_a = tmp_path / 'a.kwcoco.zip'
    dst_b = tmp_path / 'b.kwcoco.zip'
    _run_merge(pos_fpath, neg_fpath, dst_a, neg_over_pos=1.5, seed=42)
    _run_merge(pos_fpath, neg_fpath, dst_b, neg_over_pos=1.5, seed=42)
    a = kwcoco.CocoDataset.coerce(str(dst_a))
    b = kwcoco.CocoDataset.coerce(str(dst_b))
    a_names = sorted(img.get('name', img.get('file_name')) for img in a.images().objs)
    b_names = sorted(img.get('name', img.get('file_name')) for img in b.images().objs)
    assert a_names == b_names


def test_merge_aborts_on_empty_positives(tmp_path):
    """No positives -> hard error (training without positives is degenerate)."""
    import kwcoco
    # Build an empty-positive pos.kwcoco and a non-empty neg.kwcoco.
    pos = kwcoco.CocoDataset(); pos.add_category(name='poop')
    pos.fpath = str(tmp_path / 'pos.kwcoco.zip'); pos.dump()

    neg = kwcoco.CocoDataset(); neg.add_category(name='poop')
    # Add a fake image so subset isn't empty either; tile_role=negative.
    neg.add_image(file_name='nonexistent.jpg', width=128, height=128,
                  tile_role='negative')
    neg.fpath = str(tmp_path / 'neg.kwcoco.zip'); neg.dump()

    dst = tmp_path / 'out.kwcoco.zip'
    proc = subprocess.run([
        sys.executable, str(V5_DIR / 'v5_merge.py'),
        '--pos_kwcoco', str(pos.fpath),
        '--neg_kwcoco', str(neg.fpath),
        '--dst', str(dst),
    ], capture_output=True, text=True)
    assert proc.returncode != 0
    assert 'no positive tiles' in (proc.stderr + proc.stdout).lower()
