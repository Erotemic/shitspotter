"""Tests for v5_tile.py — the multi-scale fixed-size tile extractor."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_parse_scales_basic():
    import v5_tile
    out = v5_tile._parse_scales('1.0,0.66,0.4,0.25')
    assert [name for name, _ in out] == ['s10', 's07', 's04', 's02']
    factors = [s for _, s in out]
    assert factors == pytest.approx([1.0, 0.66, 0.4, 0.25])


def test_parse_scales_rejects_bad():
    import v5_tile
    with pytest.raises(ValueError):
        v5_tile._parse_scales('0')
    with pytest.raises(ValueError):
        v5_tile._parse_scales('-1.0')
    with pytest.raises(ValueError):
        v5_tile._parse_scales('5.0')  # > 4


def test_grid_positions_covers_extent():
    import v5_tile
    pos = v5_tile._grid_positions(800, 320, 160)
    assert pos[0] == 0
    # Last position must place the tile flush with the right edge.
    assert pos[-1] == 800 - 320
    # No duplicates.
    assert pos == sorted(set(pos))


def test_grid_positions_smaller_than_tile():
    import v5_tile
    # If image is smaller than tile, return [0] (the only viable position;
    # tile will be edge-padded).
    assert v5_tile._grid_positions(200, 320, 160) == [0]


def test_clip_bbox_keeps_majority():
    import v5_tile
    out = v5_tile._clip_bbox_xywh([100, 100, 100, 100],
                                  50, 50, 200, 200, 0.30)
    assert out is not None
    new_xywh, keep = out
    assert pytest.approx(keep) == 1.0
    assert new_xywh == [50, 50, 100, 100]


def test_clip_bbox_drops_majority_loss():
    import v5_tile
    out = v5_tile._clip_bbox_xywh([0, 0, 100, 100],
                                  90, 90, 200, 200, 0.30)
    assert out is None


def test_v5_tile_produces_multi_scale_tiles(v5_tile_bundle):
    """Extractor produces tiles at all configured scales."""
    import kwcoco
    dset = kwcoco.CocoDataset.coerce(str(v5_tile_bundle))

    # We configured 3 scales: s10, s05, s02 (1.0, 0.5, 0.25)
    scales = {img.get('tile_scale_name') for img in dset.images().objs}
    # At least one scale must produce tiles (others may be skipped via
    # min_source_scale_long_side for very small images).
    assert scales, 'no tiles produced at any scale'
    assert scales.issubset({'s10', 's05', 's02'}), f'unexpected scales: {scales}'


def test_v5_tile_all_outputs_are_fixed_size(v5_tile_bundle):
    """Every output tile must be exactly tile_size x tile_size."""
    import kwcoco
    dset = kwcoco.CocoDataset.coerce(str(v5_tile_bundle))
    for img in dset.images().objs:
        assert img['width'] == 128
        assert img['height'] == 128


def test_v5_tile_tags_positive_vs_negative(v5_tile_bundle):
    """Every tile is labelled 'positive' or 'negative'."""
    import kwcoco
    dset = kwcoco.CocoDataset.coerce(str(v5_tile_bundle))
    roles = {img.get('tile_role') for img in dset.images().objs}
    assert roles.issubset({'positive', 'negative'})
    # Pos tiles should have at least one annotation; neg tiles must have none.
    ann_by_gid = {}
    for ann in dset.annots().objs:
        ann_by_gid.setdefault(ann['image_id'], []).append(ann)
    for img in dset.images().objs:
        n = len(ann_by_gid.get(img['id'], []))
        if img['tile_role'] == 'positive':
            assert n >= 1, f'positive tile {img["id"]} has no anns'
        else:
            assert n == 0, f'negative tile {img["id"]} has {n} anns'


def test_v5_tile_records_source_traceability(v5_tile_bundle):
    """Every tile records source_gid, scale, and extent for debugging."""
    import kwcoco
    dset = kwcoco.CocoDataset.coerce(str(v5_tile_bundle))
    for img in dset.images().objs:
        assert 'tile_source_gid' in img
        assert 'tile_scale_name' in img
        assert 'tile_scale_factor' in img
        assert 'tile_extent_xyxy_in_source' in img
        x0, y0, x1, y1 = img['tile_extent_xyxy_in_source']
        assert x1 > x0 and y1 > y0


def test_v5_tile_annotations_lie_inside_tile(v5_tile_bundle):
    """Warped annotations must be inside the tile bounds."""
    import kwcoco
    dset = kwcoco.CocoDataset.coerce(str(v5_tile_bundle))
    img_dims = {img['id']: (img['width'], img['height'])
                for img in dset.images().objs}
    for ann in dset.annots().objs:
        bx, by, bw, bh = ann['bbox']
        W, H = img_dims[ann['image_id']]
        assert bw > 0 and bh > 0
        assert -1 <= bx <= W
        assert -1 <= by <= H
        assert bx + bw <= W + 1
        assert by + bh <= H + 1


def test_v5_tile_keep_negative_false(synthetic_kwcoco, tmp_path):
    """With keep_negative=False, only positive tiles are written."""
    import kwcoco
    import subprocess
    import sys
    dst = tmp_path / 'pos_only.kwcoco.zip'
    proc = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[2]
                            / 'experiments' / 'mobile_app_training_v5' / 'v5_tile.py'),
        '--src', str(synthetic_kwcoco),
        '--dst', str(dst),
        '--tile_size', '128',
        '--source_scales', '1.0,0.5',
        '--keep_negative', 'False',
        '--progress', 'False',
    ], capture_output=True, text=True, check=True)
    dset = kwcoco.CocoDataset.coerce(str(dst))
    roles = {img.get('tile_role') for img in dset.images().objs}
    assert roles == {'positive'} or roles == set(), (
        f'expected positive-only or empty; got {roles}')
