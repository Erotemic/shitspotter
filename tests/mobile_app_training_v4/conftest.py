"""
Shared fixtures for the mobile_app_training_v4 test suite.

The fixtures favour real-data subsets when the shitspotter DVC roots
are readable, falling back to fully-synthetic kwcoco bundles when not.
That way the same tests run cold on a clean dev VM and against the
real splits on the host.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = REPO_ROOT / 'experiments' / 'mobile_app_training_v4'

# Make the v4 directory importable as a flat module path so tests can
# `import tile_kwcoco`, `import v4_mock`, `import eligibility_manifest`.
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))


# ---------------------------------------------------------------------------
# Synthetic kwcoco fixture — always available
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def synthetic_kwcoco(tmp_path_factory) -> Path:
    """Build a tiny self-contained kwcoco bundle with poop annotations.

    Hand-built (no kwcoco.demo) so the fixture survives torchvision/torch
    ABI mismatches in the test env. 6 random RGB images with 1-2 poop
    boxes each. Always available — does not depend on the DVC mounts.
    """
    import json

    import kwcoco
    import kwimage
    import numpy as np

    bundle_dpath = tmp_path_factory.mktemp('synthetic_kwcoco')
    img_dpath = bundle_dpath / 'images'
    img_dpath.mkdir()

    rng = np.random.RandomState(42)
    dset = kwcoco.CocoDataset()
    cat_id = dset.add_category(name='poop')

    H, W = 480, 640
    for i in range(6):
        arr = (rng.rand(H, W, 3) * 255).astype(np.uint8)
        fpath = img_dpath / f'img_{i:03d}.jpg'
        kwimage.imwrite(str(fpath), arr)
        gid = dset.add_image(file_name=str(fpath.relative_to(bundle_dpath)),
                             width=W, height=H)
        # 1 or 2 boxes per image.
        n_boxes = 1 + (i % 2)
        for _ in range(n_boxes):
            bw = int(rng.randint(40, 200))
            bh = int(rng.randint(40, 200))
            bx = int(rng.randint(0, W - bw))
            by = int(rng.randint(0, H - bh))
            dset.add_annotation(image_id=gid, category_id=cat_id,
                                bbox=[bx, by, bw, bh], area=bw * bh)

    dset.fpath = str(bundle_dpath / 'synthetic.kwcoco.zip')
    dset.dump()
    return Path(dset.fpath)


# ---------------------------------------------------------------------------
# Real-subset fixture — opt-in, skipped when DVC mounts unavailable
# ---------------------------------------------------------------------------

DVC_DATA_DPATH = Path(os.environ.get(
    'DVC_DATA_DPATH', '/data/joncrall/dvc-repos/shitspotter_dvc'))

REAL_SUBSET_AVAILABLE = (
    DVC_DATA_DPATH.exists()
    and (DVC_DATA_DPATH / 'train_imgs10671_b277c63d.kwcoco.zip').exists()
)


@pytest.fixture(scope='session')
def real_subset_train(tmp_path_factory) -> Path:
    if not REAL_SUBSET_AVAILABLE:
        pytest.skip('shitspotter DVC mount not available')
    out_dpath = tmp_path_factory.mktemp('real_subset')
    out_fpath = out_dpath / 'train_subset.kwcoco.zip'
    src = DVC_DATA_DPATH / 'train_imgs10671_b277c63d.kwcoco.zip'
    import subprocess
    subprocess.run([
        sys.executable, '-m', 'kwcoco', 'subset',
        '--src', str(src),
        '--dst', str(out_fpath),
        '--gids', '1,2,3,4,5,6,7,8',
    ], check=True)
    return out_fpath


# ---------------------------------------------------------------------------
# Workspace fixture — V4_ROOT-shaped scratch space
# ---------------------------------------------------------------------------

@pytest.fixture
def v4_workspace(tmp_path) -> Path:
    """Per-test V4_ROOT-shaped scratch directory."""
    (tmp_path / 'data').mkdir()
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'eval').mkdir()
    return tmp_path
