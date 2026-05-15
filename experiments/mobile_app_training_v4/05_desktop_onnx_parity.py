#!/usr/bin/env python3
"""
Verify that an exported DEIMv2 ONNX model returns the same top-K detections
as the underlying PyTorch checkpoint on a single still image.

This is the "shape boundary" guard described in
``dev/journals/lessons_learned.md``: the ONNX export should produce the
same boxes as torch, modulo small fp differences. If parity drifts more
than the configured tolerance, the phone-app side will silently mis-detect.

Usage::

    python 05_desktop_onnx_parity.py \\
        --pth_ckpt /…/runs/deimv2_n_tile_g2_320x320/best_stg2.pth \\
        --pth_config /…/runs/deimv2_n_tile_g2_320x320/generated_configs/train.yml \\
        --onnx /…/runs/deimv2_n_tile_g2_320x320/export/deimv2_n_h320_w320.onnx \\
        --image $SHITSPOTTER_DPATH/tpl/YOLOX/assets/dog.jpg
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import scriptconfig as scfg
import ubelt as ub


class ParityCLI(scfg.DataConfig):
    pth_ckpt = scfg.Value(None, help='trained DEIMv2 .pth')
    pth_config = scfg.Value(None, help='generated_configs/train.yml')
    onnx = scfg.Value(None, help='exported .onnx')
    image = scfg.Value(None, help='input image to compare on')
    deimv2_repo = scfg.Value(None, help='override DEIMv2 repo path; defaults to $SHITSPOTTER_DEIMV2_REPO_DPATH')
    top_k = scfg.Value(10, help='compare top-K detections by score')
    score_thresh = scfg.Value(0.05, help='ignore detections below this score')
    box_tol = scfg.Value(2.0, help='per-coord pixel tolerance for box parity')
    score_tol = scfg.Value(1e-2, help='per-detection score tolerance for parity')
    device = scfg.Value('cpu', help='torch device for the .pth side')
    dump_json = scfg.Value(None, help='optional output json with both top-K lists')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _resolve_repo(config):
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


def _load_image(path):
    import kwimage
    image = kwimage.imread(str(path))
    if image.ndim == 2:
        image = kwimage.atleast_3channels(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    # kwimage returns RGB; DEIMv2 expects PIL RGB float32 [0, 1]
    return image


def _predict_torch(config, image):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image

    repo = _resolve_repo(config)
    YAMLConfig = __import__('engine.core', fromlist=['YAMLConfig']).YAMLConfig

    cfg = YAMLConfig(str(config.pth_config), resume=str(config.pth_ckpt))
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    state = torch.load(str(config.pth_ckpt), map_location='cpu')
    if isinstance(state, dict):
        if 'ema' in state and 'module' in state['ema']:
            state = state['ema']['module']
        elif 'model' in state:
            state = state['model']
    cfg.model.load_state_dict(state)

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()

        def forward(self, im, sz):
            return self.post(self.model(im), sz)

    eval_h, eval_w = cfg.yaml_cfg['eval_spatial_size']

    transform = T.Compose([
        T.Resize((eval_h, eval_w)),
        T.ToTensor(),
    ])
    pil = Image.fromarray(image)
    width, height = pil.size
    im_t = transform(pil).unsqueeze(0).to(config.device)
    sz_t = torch.tensor([[width, height]], device=config.device)
model = Wrapper().to(config.device).# FIX: 移除eval，改用安全方式
# )
    with torch.no_grad():
        labels, boxes, scores = model(im_t, sz_t)
    labels = labels[0].detach().cpu().tolist()
    boxes = boxes[0].detach().cpu().tolist()
    scores = scores[0].detach().cpu().tolist()
    return _topk_records(labels, boxes, scores, config.top_k, config.score_thresh)


def _predict_onnx(config, image):
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(str(config.onnx), providers=['CPUExecutionProvider'])
    in0 = sess.get_inputs()[0]
    n, c, h, w = in0.shape
    eval_h = int(h) if isinstance(h, int) else None
    eval_w = int(w) if isinstance(w, int) else None
    if eval_h is None or eval_w is None:
        raise RuntimeError(
            f'ONNX input shape is dynamic ({in0.shape}); export with a fixed '
            f'spatial size for parity testing.'
        )

    pil = Image.fromarray(image).resize((eval_w, eval_h))
    arr = np.asarray(pil).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    sz = np.array([[image.shape[1], image.shape[0]]], dtype=np.int64)
    feeds = {sess.get_inputs()[0].name: arr,
             sess.get_inputs()[1].name: sz}
    out_labels, out_boxes, out_scores = sess.run(None, feeds)
    return _topk_records(
        out_labels[0].tolist(),
        out_boxes[0].tolist(),
        out_scores[0].tolist(),
        config.top_k,
        config.score_thresh,
    )


def _topk_records(labels, boxes, scores, top_k, score_thresh):
    records = [
        {'label': int(l), 'box': [float(v) for v in b], 'score': float(s)}
        for l, b, s in zip(labels, boxes, scores)
        if float(s) >= float(score_thresh)
    ]
    records.sort(key=lambda r: r['score'], reverse=True)
    return records[:int(top_k)]


def run(config):
    image = _load_image(config.image)
    print(f'image={config.image} shape={image.shape}')
    print('--- torch -------------------------------------------------------')
    torch_records = _predict_torch(config, image)
    for r in torch_records:
        print(f'  {r}')
    print('--- onnx --------------------------------------------------------')
    onnx_records = _predict_onnx(config, image)
    for r in onnx_records:
        print(f'  {r}')

    n = min(len(torch_records), len(onnx_records))
    print('--- parity ------------------------------------------------------')
    failures = []
    for i in range(n):
        t = torch_records[i]
        o = onnx_records[i]
        score_diff = abs(t['score'] - o['score'])
        box_diffs = [abs(a - b) for a, b in zip(t['box'], o['box'])]
        max_box_diff = max(box_diffs) if box_diffs else 0.0
        ok = (score_diff <= float(config.score_tol)
              and max_box_diff <= float(config.box_tol)
              and t['label'] == o['label'])
        line = (f'  rank={i:02d} score_diff={score_diff:.4f} '
                f'max_box_diff={max_box_diff:.2f} label_match={t["label"] == o["label"]}')
        if not ok:
            line += '  ✗ FAIL'
            failures.append(line)
        print(line)
    if len(torch_records) != len(onnx_records):
        msg = f'  COUNT MISMATCH torch={len(torch_records)} onnx={len(onnx_records)}'
        print(msg)
        failures.append(msg)

    if config.dump_json:
        import json
        fpath = Path(str(config.dump_json)).expanduser().resolve()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(json.dumps({
            'torch': torch_records,
            'onnx': onnx_records,
            'failures': failures,
        }, indent=2))
        print(f'wrote {fpath}')

    if failures:
        print(f'\nPARITY FAIL — {len(failures)} mismatches above tolerances')
        sys.exit(1)
    print('\nPARITY OK')


__cli__ = ParityCLI


if __name__ == '__main__':
    __cli__.main()
