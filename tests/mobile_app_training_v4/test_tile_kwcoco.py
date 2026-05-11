"""Tests for tile_kwcoco.py — the coarse-to-fine training data builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_tile_extents_grid_one_returns_full_image():
    import tile_kwcoco
    extents = tile_kwcoco._tile_extents(800, 600, grid=1, overlap=0.0)
    assert extents == [(0, 0, 800, 600)]


def test_tile_extents_2x2_covers_image():
    import tile_kwcoco
    extents = tile_kwcoco._tile_extents(1280, 960, grid=2, overlap=0.20)
    assert len(extents) == 4
    # Every tile must lie inside [0, W] x [0, H].
    for x0, y0, x1, y1 in extents:
        assert 0 <= x0 < x1 <= 1280
        assert 0 <= y0 < y1 <= 960
    # The union must reach the right and bottom edges.
    assert max(e[2] for e in extents) == 1280
    assert max(e[3] for e in extents) == 960


def test_clip_bbox_keeps_majority():
    import tile_kwcoco
    # bbox at (100, 100) size 100x100. Tile is (50, 50)-(200, 200).
    out = tile_kwcoco._clip_bbox_xywh(
        [100, 100, 100, 100], 50, 50, 200, 200, min_keep_fraction=0.30)
    assert out is not None
    new_xywh, keep = out
    assert pytest.approx(keep) == 1.0  # entire box visible
    assert new_xywh == [50, 50, 100, 100]


def test_clip_bbox_drops_majority_loss():
    import tile_kwcoco
    # bbox at (0, 0) size 100x100. Tile is (90, 90)-(200, 200).
    # Visible region is 10x10 of original 100x100 -> 1% kept.
    out = tile_kwcoco._clip_bbox_xywh(
        [0, 0, 100, 100], 90, 90, 200, 200, min_keep_fraction=0.30)
    assert out is None


def test_resize_with_long_side_preserves_aspect():
    import numpy as np
    import tile_kwcoco
    img = (np.random.rand(1200, 1600, 3) * 255).astype(np.uint8)
    resized, scale = tile_kwcoco._resize_with_long_side(img, max_dim=800)
    assert resized.shape[1] == 800       # long side W=1600 -> 800
    assert resized.shape[0] == 600       # short side H=1200 -> 600
    assert pytest.approx(scale) == 0.5


def test_resize_with_long_side_no_op_if_small():
    import numpy as np
    import tile_kwcoco
    img = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
    resized, scale = tile_kwcoco._resize_with_long_side(img, max_dim=800)
    assert resized.shape == img.shape
    assert scale == 1.0


def test_tile_kwcoco_end_to_end(synthetic_kwcoco, tmp_path):
    """Smoke: full bundle -> tile bundle, verifying images + boxes."""
    import kwcoco
    import tile_kwcoco as tk

    dst = tmp_path / 'tiled.kwcoco.zip'
    config = tk.TileKwcocoCLI(
        src=str(synthetic_kwcoco),
        dst=str(dst),
        full_dim=320,
        tile_grid=2,
        tile_overlap=0.20,
        tile_output_dim=192,
        keep_full=True,
        min_keep_fraction=0.10,
        output_ext='.jpg',
        progress=False,
    )
    tk.run(config)
    assert dst.exists()

    out = kwcoco.CocoDataset.coerce(str(dst))
    src = kwcoco.CocoDataset.coerce(str(synthetic_kwcoco))
    n_full = sum(1 for img in out.images().objs if img.get('tile_role') == 'full')
    n_tile = sum(1 for img in out.images().objs if img.get('tile_role') == 'tile')

    assert n_full == src.n_images
    # 2x2 grid -> 4 tiles per source.
    assert n_tile == src.n_images * 4

    # All warped boxes must lie inside the resized image they belong to.
    img_dims = {img['id']: (img['width'], img['height']) for img in out.images().objs}
    for ann in out.annots().objs:
        bx, by, bw, bh = ann['bbox']
        W, H = img_dims[ann['image_id']]
        assert 0 <= bx <= W
        assert 0 <= by <= H
        assert bw > 0 and bh > 0
        assert bx + bw <= W + 1   # +1 for rounding
        assert by + bh <= H + 1

    # Every tile image carries its source gid and extent.
    for img in out.images().objs:
        if img.get('tile_role') != 'tile':
            continue
        assert 'tile_source_gid' in img
        assert 'tile_extent_xyxy' in img
        x0, y0, x1, y1 = img['tile_extent_xyxy']
        assert x1 > x0 and y1 > y0


def test_tile_kwcoco_records_config_in_info(synthetic_kwcoco, tmp_path):
    """The output bundle must carry enough info to reproduce the run."""
    import kwcoco
    import tile_kwcoco as tk

    dst = tmp_path / 'tiled.kwcoco.zip'
    config = tk.TileKwcocoCLI(
        src=str(synthetic_kwcoco), dst=str(dst),
        full_dim=320, tile_grid=2, tile_overlap=0.10, tile_output_dim=160,
        keep_full=True, min_keep_fraction=0.20, progress=False,
    )
    tk.run(config)
    out = kwcoco.CocoDataset.coerce(str(dst))
    assert any(
        info.get('name') == 'mobile_app_training_v4.tile_kwcoco'
        for info in (out.dataset.get('info', []) or [])
    ), 'tile_kwcoco must add an info row to the output kwcoco for traceability'
