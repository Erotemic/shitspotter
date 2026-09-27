from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    exp = repo / 'experiments' / 'rfdetr_seg_v1'
    sys.path.insert(0, str(exp))
    try:
        spec = importlib.util.spec_from_file_location(
            'rfdetr_seg_v1_run_overnight', exp / 'run_overnight.py'
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_find_resume_checkpoint_prefers_last(tmp_path):
    module = _load_module()
    (tmp_path / 'checkpoint_epoch=1.ckpt').write_text('epoch')
    last = tmp_path / 'last.ckpt'
    last.write_text('last')
    assert module._find_resume_checkpoint(tmp_path) == last


def test_find_resume_checkpoint_chooses_latest_epoch(tmp_path):
    module = _load_module()
    for epoch in [1, 9, 3]:
        (tmp_path / f'checkpoint_epoch={epoch}.ckpt').write_text(str(epoch))
    assert module._find_resume_checkpoint(tmp_path).name == 'checkpoint_epoch=9.ckpt'


def test_find_resume_checkpoint_ignores_lightweight_best(tmp_path):
    module = _load_module()
    (tmp_path / 'checkpoint_best_ema.pth').write_text('weights only')
    assert module._find_resume_checkpoint(tmp_path) is None
