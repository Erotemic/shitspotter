#!/usr/bin/env python3
"""
v5 hard-negative miner.

Given a trained DEIMv2 checkpoint + a kwcoco bundle of NEGATIVE tiles
(tile_role == 'negative' as produced by v5_tile.py), score each tile
with the model and emit a kwcoco subset of "hard" negatives — tiles
where the model produces a high-confidence false detection.

The output kwcoco can then be unioned with the positive-tile bundle
to form the next training round.

Pipeline contract
-----------------
Inputs:
    --neg_kwcoco           kwcoco bundle of negative tiles
    --workdir              v4-style workdir for the trained model
                           (we read best_stg2.pth + generated_configs/train.yml
                           directly, the same way 04_eval_on_test.sh does)
    --score_thresh         a tile is "hard" iff its max prediction score
                           exceeds this threshold (default 0.30)
    --max_hard_per_round   cap the number of hard negatives emitted; if
                           more tiles qualify, keep the highest-scoring
                           ones. Default 5000 (≈ enough to balance a
                           positives pool of 1000–5000 tiles).
    --device               torch device (default cuda:0)

Output:
    --dst                  kwcoco bundle containing only the hard
                           negative tiles, with their max_pred_score
                           recorded as image metadata.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import scriptconfig as scfg


class V5MineCLI(scfg.DataConfig):
    neg_kwcoco = scfg.Value(None, help='input kwcoco of negative tiles', required=True)
    workdir = scfg.Value(None, help='v4-style trainer workdir with best_stg2.pth', required=True)
    dst = scfg.Value(None, help='output kwcoco of hard negatives', required=True)

    score_thresh = scfg.Value(0.30, help='tile is "hard" iff max pred score >= this')
    max_hard_per_round = scfg.Value(5000,
        help='cap total hard negatives; if exceeded, keep highest-scoring')
    device = scfg.Value('cuda:0', help='torch device')
    batch_size = scfg.Value(8, help='inference batch size')
    deimv2_repo = scfg.Value(None,
        help='override DEIMv2 repo path; defaults to $SHITSPOTTER_DEIMV2_REPO_DPATH')
    progress = scfg.Value(True, help='show ProgIter')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _resolve_deimv2_repo(config) -> Path:
    repo = config.deimv2_repo or os.environ.get('SHITSPOTTER_DEIMV2_REPO_DPATH')
    if not repo:
        raise EnvironmentError(
            'Set --deimv2_repo or $SHITSPOTTER_DEIMV2_REPO_DPATH so we can '
            'import engine.core.YAMLConfig.'
        )
    repo = Path(repo).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _load_model(workdir: Path, device: str, deimv2_repo: Path):
    """Load a trained DEIMv2 detector in deploy mode.

    Mirrors what v4's `shitspotter.algo_foundation_v3.detector_deimv2`
    does, but locally so v5 doesn't need the foundation_v3 import to
    be on PYTHONPATH.
    """
    import torch
    import torch.nn as nn
    import torchvision.transforms as T

    _ = deimv2_repo  # already on sys.path
    YAMLConfig = __import__('engine.core', fromlist=['YAMLConfig']).YAMLConfig

    ckpt_fpath = workdir / 'best_stg2.pth'
    if not ckpt_fpath.exists():
        # Fall back to highest-numbered checkpoint
        cands = sorted(workdir.glob('checkpoint*.pth'))
        if not cands:
            raise FileNotFoundError(
                f'No checkpoint in {workdir} (no best_stg2.pth or checkpoint*.pth)')
        ckpt_fpath = cands[-1]

    cfg_fpath = workdir / 'generated_configs' / 'train.yml'
    if not cfg_fpath.exists():
        raise FileNotFoundError(cfg_fpath)

    state = torch.load(str(ckpt_fpath), map_location='cpu')
    if isinstance(state, dict):
        if 'ema' in state and 'module' in state['ema']:
            state = state['ema']['module']
        elif 'model' in state:
            state = state['model']

    cfg = YAMLConfig(str(cfg_fpath), resume=str(ckpt_fpath))
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.model.load_state_dict(state)

    class _Wrapper(nn.Module):
        def __init__(self, cfg_):
            super().__init__()
            self.model = cfg_.model.deploy()
            self.post = cfg_.postprocessor.deploy()

        def forward(self, im, sz):
            return self.post(self.model(im), sz)

    eval_h, eval_w = cfg.yaml_cfg['eval_spatial_size']
    transform = T.Compose([T.Resize((int(eval_h), int(eval_w))), T.ToTensor()])
    model = _Wrapper(cfg).to(device).eval()
    return model, transform, (int(eval_h), int(eval_w))


def _score_tile(model, transform, image_np, device: str) -> float:
    """Return the max prediction score for a single tile."""
    import torch
    from PIL import Image
    pil = Image.fromarray(image_np)
    width, height = pil.size
    im_t = transform(pil).unsqueeze(0).to(device)
    sz_t = torch.tensor([[width, height]], device=device)
    with torch.no_grad():
        _labels, _boxes, scores = model(im_t, sz_t)
    if scores.numel() == 0:
        return 0.0
    return float(scores.max().item())


def run(config):
    import kwcoco
    import numpy as np
    import torch
    import ubelt as ub

    deimv2_repo = _resolve_deimv2_repo(config)
    workdir = Path(str(config.workdir)).expanduser().resolve()
    neg_fpath = Path(str(config.neg_kwcoco)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()

    print(f'v5_mine: workdir={workdir}')
    print(f'         neg_kwcoco={neg_fpath}')
    print(f'         dst={dst_fpath}')
    print(f'         score_thresh={config.score_thresh}')
    print(f'         max_hard_per_round={config.max_hard_per_round}')

    model, transform, eval_hw = _load_model(workdir, str(config.device), deimv2_repo)
    print(f'         model eval_spatial_size={eval_hw}')

    neg_dset = kwcoco.CocoDataset.coerce(str(neg_fpath))

    # Filter to negative-role tiles. v5_tile.py tags them; if the input
    # was an arbitrary kwcoco without tile_role, score everything.
    candidate_gids = []
    for img in neg_dset.images().objs:
        role = img.get('tile_role')
        if role in (None, 'negative'):
            candidate_gids.append(img['id'])
    print(f'         candidates: {len(candidate_gids)} negative tiles')

    # Score every candidate tile.
    scored: List[Tuple[float, int]] = []  # (score, gid)
    iterator = ub.ProgIter(candidate_gids, desc='v5_mine score neg tiles',
                           enabled=bool(config.progress))
    for gid in iterator:
        try:
            arr = neg_dset.coco_image(gid).imdelay().finalize()
        except Exception as ex:
            print(f'  warn: failed to read gid {gid}: {ex}')
            continue
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        s = _score_tile(model, transform, arr, str(config.device))
        scored.append((s, gid))

    # Keep highest-scoring N above threshold.
    thresh = float(config.score_thresh)
    max_keep = int(config.max_hard_per_round)
    hard = [(s, g) for (s, g) in scored if s >= thresh]
    hard.sort(reverse=True)  # descending by score
    if len(hard) > max_keep:
        hard = hard[:max_keep]
    hard_gids = {g for _s, g in hard}
    score_by_gid = {g: s for s, g in hard}

    print(f'         {len(hard)} hard negatives kept '
          f'(of {len(scored)} scored; threshold {thresh})')

    # Build output kwcoco as a subset of the input.
    out_dset = kwcoco.CocoDataset()
    out_dset.fpath = str(dst_fpath)
    cat_id = out_dset.add_category(name='poop')
    n_kept_imgs = 0
    for img in neg_dset.images().objs:
        gid = img['id']
        if gid not in hard_gids:
            continue
        new = {k: v for k, v in img.items() if k not in ('id',)}
        new['max_pred_score'] = float(score_by_gid[gid])
        new['mined_for_round'] = int(os.environ.get('V5_ROUND', '0'))
        out_dset.add_image(id=gid, **new)
        n_kept_imgs += 1
    # No annotations on hard negatives — they're negatives by definition.
    _ = cat_id

    out_dset.dump()
    print(f'  wrote {n_kept_imgs} hard-neg tile images to {dst_fpath}')

    # Sidecar JSON with score histogram for debugging.
    if scored:
        bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.01]
        hist = [0] * (len(bins) - 1)
        for s, _ in scored:
            for i in range(len(bins) - 1):
                if bins[i] <= s < bins[i + 1]:
                    hist[i] += 1
                    break
        sidecar = dst_fpath.with_suffix('.mine_stats.json')
        sidecar.write_text(json.dumps({
            'n_scored': len(scored),
            'n_hard': len(hard),
            'score_thresh': thresh,
            'max_hard_per_round': max_keep,
            'score_bins': bins,
            'score_hist': hist,
        }, indent=2))
        print(f'  wrote score histogram to {sidecar}')


__cli__ = V5MineCLI


if __name__ == '__main__':
    __cli__.main()
