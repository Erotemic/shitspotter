#!/usr/bin/env python3
"""
v4_mock — a deliberately tiny torch detector that exercises the v4
pipeline end-to-end on CPU in seconds, without needing DEIMv2 or any
of its upstream submodules.

Why this exists
---------------
The v4 pipeline (sweep -> train -> export -> eval -> bench -> manifest)
is bash-orchestrated and takes the real DEIMv2 trainer many GPU-hours
per cell. To smoke-test the *plumbing* — generated YAML structure,
checkpoint discovery, ONNX export, eval driver, modelspec sidecar,
manifest aggregation — we need a model that:

  * Has DEIMv2-shaped IO: inputs (images NxCxHxW float32,
    orig_target_sizes Nx2 int64), outputs (labels NxK int64, boxes
    NxKx4 float, scores NxK float), boxes in pixel coords w.r.t.
    orig_target_sizes.
  * Trains on CPU in <60s on 8 toy images.
  * Has measurable loss decrease over a handful of iterations
    (verifies gradients flow, not just code paths).
  * Exports cleanly to ONNX with a fixed input shape.
  * Writes the same on-disk artifacts the DEIMv2 trainer writes
    (`best_stg2.pth` so the existing checkpoint discovery works,
    `policy.json`, `generated_configs/train.yml`).

Architecture
------------
Single 3->8 stride-8 conv + GAP + a per-query (cx, cy, w, h, obj)
head. K query priors at fixed grid centres in normalized image space;
network learns small deltas. Total params ~few hundred. Inductive
bias: a query that starts near a GT box reaches it after a few
gradient steps.

This is *not* a serious detector. It exists to make the pipeline
testable on CPU.

Subcommands
-----------
    v4_mock.py train  --train_kwcoco ... --vali_kwcoco ... --workdir ...
    v4_mock.py export --workdir ... --export_h H --export_w W
    v4_mock.py eval   --workdir ... --test_kwcoco ...
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import scriptconfig as scfg
import ubelt as ub


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

NUM_QUERIES_DEFAULT = 16
NUM_CLASSES = 1


def _build_model(num_queries=NUM_QUERIES_DEFAULT, prior_boxes_norm=None):
    """
    prior_boxes_norm: optional (K, 4) tensor in normalised [0, 1] xyxy.
        These are the oracle query priors — typically derived from the
        training set's GT boxes by `_collect_prior_boxes`. They get
        baked in as a non-learnable buffer; the only learnable
        parameter of consequence is a single scalar `gate_logit` that
        controls confidence. This means a few steps of gradient
        descent on a "be confident at the matched query" objective
        flip the gate from off to on, producing real AP > 0 against
        the test set without needing real backbone capacity.
    """
    import torch
    import torch.nn as nn

    K = int(num_queries)
    if prior_boxes_norm is None:
        # Fallback grid: K queries on a sqrt(K) x sqrt(K) lattice with
        # default size 0.20.
        side = max(1, int(round(K ** 0.5)))
        ys, xs = torch.meshgrid(
            torch.linspace(0.20, 0.80, side),
            torch.linspace(0.20, 0.80, side),
            indexing='ij',
        )
        cxcy = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
        if cxcy.shape[0] < K:
            cxcy = torch.cat(
                [cxcy, torch.full((K - cxcy.shape[0], 2), 0.5)], dim=0)
        cxcy = cxcy[:K]
        wh = torch.full_like(cxcy, 0.20)
        xy0 = (cxcy - wh / 2).clamp(0, 1)
        xy1 = (cxcy + wh / 2).clamp(0, 1)
        prior_boxes_norm = torch.cat([xy0, xy1], dim=-1)

    prior_boxes_norm = prior_boxes_norm.float().clamp(0, 1)
    if prior_boxes_norm.shape != (K, 4):
        raise ValueError(
            f'prior_boxes_norm must be shape ({K}, 4), got {tuple(prior_boxes_norm.shape)}')

    class V4MockTinyDetector(nn.Module):
        """
        Hard-coded oracle priors + a single learnable scalar gate.

        Inductive bias: the priors are baked in as a non-learnable
        buffer — the model already knows where to look. The single
        scalar `gate_logit` (initialised to ~ -2.5, sigmoid ≈ 0.075)
        controls whether predictions exceed the score threshold. A
        few gradient steps of "matched queries should have score 1"
        push the gate up enough to clear the 0.30 threshold and
        produce real AP.

        A vestigial conv stem is included so the graph has a real
        learnable forward path through the image — this matters for
        ONNX export (otherwise the input tensor is dead) and makes
        the model honest about being a "small detector that consumes
        an image", but the stem's output is gated to a small additive
        offset on the gate. So gradients still flow into the stem,
        but the dominant control is the gate scalar.

        Outputs (labels, boxes, scores) match DEIMv2's deploy()
        postprocessor format.
        """

        def __init__(self):
            super().__init__()
            self.K = K
            self.register_buffer('priors_xyxy_norm', prior_boxes_norm.clone())
            # Single scalar gate, initialised so sigmoid(gate) ~ 0.075
            # (below typical 0.30 threshold). Gradient at this value
            # is ~ 0.07, healthy.
            self.gate_logit = nn.Parameter(torch.tensor(-2.5))
            # Vestigial stem so the image actually flows through the
            # network. Its global-pooled feature contributes a small
            # additive bias to the gate (per-image confidence wobble).
            self.stem = nn.Conv2d(3, 1, kernel_size=3, stride=16, padding=1)
            self.image_bias_scale = nn.Parameter(torch.tensor(0.05))

        def forward(self, images, orig_target_sizes):
            import torch
            import torch.nn.functional as F
            N = images.shape[0]
            # Per-image confidence wobble from the stem (small).
            feat = F.relu(self.stem(images))
            img_bias = F.adaptive_avg_pool2d(feat, 1).flatten(1).squeeze(-1)  # N
            gate = self.gate_logit + self.image_bias_scale * img_bias  # N
            scores_scalar = torch.sigmoid(gate)                        # N
            scores = scores_scalar.unsqueeze(-1).expand(N, self.K)     # N x K

            # Boxes: priors scaled to pixel coords using orig_target_sizes.
            sizes_f = orig_target_sizes.float()                        # N x 2  [W, H]
            scale = torch.stack([sizes_f[:, 0], sizes_f[:, 1],
                                 sizes_f[:, 0], sizes_f[:, 1]], dim=-1)  # N x 4
            boxes = self.priors_xyxy_norm.unsqueeze(0) * scale.unsqueeze(1)  # N x K x 4
            labels = torch.zeros_like(scores, dtype=torch.long)
            return labels, boxes, scores

    return V4MockTinyDetector()


def _collect_prior_boxes(kwcoco_fpath, num_queries=NUM_QUERIES_DEFAULT):
    """Collect K prior boxes in normalised [0, 1] xyxy from a kwcoco bundle.

    We take every poop annotation, normalise its bbox by the parent
    image's (W, H), and keep the first K. If fewer than K are
    available, pad with a neutral central box. This produces "oracle
    priors" for the mock detector: query boxes that already cover the
    true GT distribution, so the only thing the mock needs to learn is
    the scalar gate.
    """
    import kwcoco
    import torch

    K = int(num_queries)
    dset = kwcoco.CocoDataset.coerce(str(kwcoco_fpath))
    cats_by_name = {c['name']: c['id'] for c in dset.dataset.get('categories', [])}
    target_cid = cats_by_name.get('poop')

    priors = []
    for ann in dset.annots().objs:
        if target_cid is not None and ann.get('category_id') != target_cid:
            continue
        bbox = ann.get('bbox')
        if not bbox:
            continue
        gid = ann['image_id']
        img = dset.imgs[gid]
        W = float(img.get('width', 0))
        H = float(img.get('height', 0))
        if W <= 0 or H <= 0:
            continue
        bx, by, bw, bh = bbox
        x1 = max(0.0, min(1.0, bx / W))
        y1 = max(0.0, min(1.0, by / H))
        x2 = max(0.0, min(1.0, (bx + bw) / W))
        y2 = max(0.0, min(1.0, (by + bh) / H))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        priors.append([x1, y1, x2, y2])
        if len(priors) >= K:
            break

    while len(priors) < K:
        # Fill with a central neutral box.
        priors.append([0.30, 0.30, 0.70, 0.70])

    return torch.tensor(priors[:K], dtype=torch.float32)


def _matched_loss(pred_boxes_xyxy, pred_scores, gt_boxes_xyxy, gt_present,
                  orig_sizes):
    """Image-level confidence loss for the scalar-gate mock model.

    pred_boxes_xyxy : N x K x 4   pixel coords
    pred_scores     : N x K       in [0,1] (all K equal in mock — single
                                  scalar gate)
    gt_boxes_xyxy   : N x M x 4   pixel coords (zero-padded)
    gt_present      : N x M       bool (which GTs are real vs padding)
    orig_sizes      : N x 2       int64 [W, H] per image — unused in
                                  this version (kept for API parity).

    With the oracle-prior + scalar-gate design, every prediction is
    already at a learned-good position; what the model learns is the
    single scalar that controls *whether to be confident*. So the
    loss is purely image-level objectness: target = 1 if the image
    has any GT, 0 if it has none. A single gradient step flips the
    gate; a few steps push it to ~1.0 on positive images.
    """
    import torch
    import torch.nn.functional as F

    # Image-level positive flag.
    has_gt = gt_present.any(dim=1).float()  # N
    # Scores are scalar-replicated across queries — take score[:, 0].
    img_scores = pred_scores[:, 0].clamp(1e-6, 1 - 1e-6)
    obj_loss = F.binary_cross_entropy(img_scores, has_gt, reduction='mean')

    # Box loss is informational only with fixed priors — report 0 so
    # the train log stays interpretable. Keeping the API stable.
    box_loss = torch.zeros((), device=pred_boxes_xyxy.device)
    return obj_loss + box_loss, float(box_loss.item()), float(obj_loss.item())


# ---------------------------------------------------------------------------
# Data loading (kwcoco)
# ---------------------------------------------------------------------------

def _coco_to_batches(kwcoco_fpath, input_h, input_w, batch_size=2,
                    max_gt=8, shuffle=True, seed=0):
    """Yield (images, orig_sizes, gt_boxes, gt_present) tensors per batch.

    images: B x 3 x input_h x input_w float in [0,1]
    orig_sizes: B x 2 int64 [W, H] of *original* image
    gt_boxes: B x max_gt x 4 float (pixel coords in *original* size)
    gt_present: B x max_gt bool
    """
    import kwcoco
    import kwimage
    import numpy as np
    import torch

    rng = np.random.RandomState(int(seed))
    dset = kwcoco.CocoDataset.coerce(str(kwcoco_fpath))
    cats_by_name = {c['name']: c['id'] for c in dset.dataset.get('categories', [])}
    target_cid = cats_by_name.get('poop')
    img_ids = list(dset.images())
    if shuffle:
        rng.shuffle(img_ids)

    def _read_image(gid):
        try:
            return dset.coco_image(gid).imdelay().finalize()
        except Exception:
            fpath = dset.get_image_fpath(gid)
            return kwimage.imread(str(fpath))

    for start in range(0, len(img_ids), batch_size):
        batch_gids = img_ids[start:start + batch_size]
        if not batch_gids:
            break
        imgs = []
        sizes = []
        gts = []
        gts_present = []
        for gid in batch_gids:
            arr = _read_image(gid)
            if arr is None:
                continue
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            orig_h, orig_w = arr.shape[:2]
            try:
                resized = kwimage.imresize(arr, dsize=(input_w, input_h),
                                           interpolation='area')
            except NotImplementedError:
                resized = kwimage.imresize(arr, dsize=(input_w, input_h),
                                           interpolation='linear')
            chw = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
            imgs.append(chw)
            sizes.append([orig_w, orig_h])
            anns = [a for a in dset.annots(gid=gid).objs
                    if (target_cid is None or a.get('category_id') == target_cid)
                    and a.get('bbox') is not None]
            gt_xyxy = np.zeros((max_gt, 4), dtype=np.float32)
            gt_pres = np.zeros((max_gt,), dtype=np.bool_)
            for k, ann in enumerate(anns[:max_gt]):
                bx, by, bw, bh = ann['bbox']
                gt_xyxy[k] = [bx, by, bx + bw, by + bh]
                gt_pres[k] = True
            gts.append(gt_xyxy)
            gts_present.append(gt_pres)
        if not imgs:
            continue
        yield (
            torch.from_numpy(np.stack(imgs)),
            torch.tensor(sizes, dtype=torch.int64),
            torch.from_numpy(np.stack(gts)),
            torch.from_numpy(np.stack(gts_present)),
        )


# ---------------------------------------------------------------------------
# Train subcommand
# ---------------------------------------------------------------------------

class V4MockCLI(scfg.ModalCLI):
    """v4_mock — tiny torch detector for end-to-end pipeline smoke testing."""


@V4MockCLI.register
class train(scfg.DataConfig):
    """Fine-tune the v4 mock detector on a kwcoco bundle."""
    train_kwcoco = scfg.Value(None, help='kwcoco bundle to train on')
    vali_kwcoco = scfg.Value(None, help='kwcoco bundle to validate on')
    workdir = scfg.Value(None, help='workdir under $V4_ROOT/runs/...')
    input_h = scfg.Value(320, help='training image height (model fixed input)')
    input_w = scfg.Value(320, help='training image width (model fixed input)')
    num_epochs = scfg.Value(2, help='number of epochs')
    batch_size = scfg.Value(2, help='batch size')
    num_queries = scfg.Value(NUM_QUERIES_DEFAULT, help='K queries')
    lr = scfg.Value(1e-2, help='learning rate')
    seed = scfg.Value(0, help='RNG seed')

    @classmethod
    def main(cls, argv=1, **kwargs):
        import torch
        config = cls.cli(argv=argv, data=kwargs, strict=True)

        torch.manual_seed(int(config.seed))
        workdir = Path(str(config.workdir)).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        # Oracle priors derived from training-set GT boxes — see
        # _build_model docstring for the design rationale.
        prior_boxes = _collect_prior_boxes(
            config.train_kwcoco, num_queries=int(config.num_queries))
        model = _build_model(num_queries=int(config.num_queries),
                             prior_boxes_norm=prior_boxes)
        opt = torch.optim.Adam(model.parameters(), lr=float(config.lr))
        model.train()

        n_params = sum(p.numel() for p in model.parameters())
        print(f'v4_mock train: model has {n_params} parameters')

        history: List[Tuple[int, int, float, float]] = []
        for epoch in range(int(config.num_epochs)):
            for step, (imgs, sizes, gt_xyxy, gt_present) in enumerate(_coco_to_batches(
                config.train_kwcoco, int(config.input_h), int(config.input_w),
                batch_size=int(config.batch_size), shuffle=True,
                seed=int(config.seed) + epoch,
            )):
                opt.zero_grad()
                _, pred_boxes, pred_scores = model(imgs, sizes)
                loss, lbox, lobj = _matched_loss(pred_boxes, pred_scores,
                                                 gt_xyxy, gt_present, sizes)
                loss.backward()
                opt.step()
                history.append((epoch, step, float(loss.item()), lbox))
                print(f'  epoch={epoch} step={step} loss={loss.item():.4f} '
                      f'(box={lbox:.4f} obj={lobj:.4f})')

        # Validation pass — just compute mean loss for reporting.
        model.eval()
        vali_losses = []
        with torch.no_grad():
            for imgs, sizes, gt_xyxy, gt_present in _coco_to_batches(
                config.vali_kwcoco, int(config.input_h), int(config.input_w),
                batch_size=int(config.batch_size), shuffle=False,
                seed=int(config.seed),
            ):
                _, pred_boxes, pred_scores = model(imgs, sizes)
                loss, _, _ = _matched_loss(pred_boxes, pred_scores,
                                           gt_xyxy, gt_present, sizes)
                vali_losses.append(float(loss.item()))
        vali_mean = sum(vali_losses) / max(len(vali_losses), 1)
        print(f'  vali_mean_loss = {vali_mean:.4f}')

        # Save checkpoint in DEIMv2-compatible naming so the rest of
        # the v4 pipeline finds it (best_stg2.pth is the first thing
        # 03/04 look for).
        ckpt_fpath = workdir / 'best_stg2.pth'
        torch.save({
            'model': model.state_dict(),
            'meta': {
                'kind': 'v4_mock',
                'num_queries': int(config.num_queries),
                'input_h': int(config.input_h),
                'input_w': int(config.input_w),
                'history': history,
                'vali_mean_loss': vali_mean,
                'prior_boxes_norm': prior_boxes.tolist(),
            },
        }, ckpt_fpath)
        print(f'  saved {ckpt_fpath}')

        # Also write a stub policy.json + generated_configs/train.yml so
        # the rest of the v4 pipeline (03_export, 04_eval, manifest)
        # finds the same artifacts as the DEIMv2 trainer. The shell
        # dispatcher passes the candidate identity in via env vars.
        gen_cfg_dpath = workdir / 'generated_configs'
        gen_cfg_dpath.mkdir(parents=True, exist_ok=True)
        gen_cfg_fpath = gen_cfg_dpath / 'train.yml'
        gen_cfg_fpath.write_text(
            f'# v4_mock generated config\n'
            f'kind: v4_mock\n'
            f'eval_spatial_size: [{int(config.input_h)}, {int(config.input_w)}]\n'
            f'num_queries: {int(config.num_queries)}\n'
        )

        policy = {
            'candidate_id': os.environ.get('V4_CANDIDATE_ID',
                f'v4_mock_tiny_{int(config.input_h)}x{int(config.input_w)}'),
            'variant': os.environ.get('V4_VARIANT', 'v4_mock_tiny'),
            'run_tag': os.environ.get('V4_RUN_TAG', 'mock'),
            'export_input_h': int(config.input_h),
            'export_input_w': int(config.input_w),
            'train_resolution_policy': os.environ.get('V4_TRAIN_POLICY', 'fixed'),
            'requested_train_resolution_min': int(config.input_h),
            'requested_train_resolution_max': int(config.input_h),
            'multiscale_base_size': int(config.input_h),
            'multiscale_repeat': 0,
            'multiscale_stop_epoch': int(config.num_epochs),
            'tile_training_policy': os.environ.get('V4_TILE_POLICY', ''),
            'tile_grid': int(os.environ.get('V4_TILE_GRID', '0') or 0),
            'tile_overlap': float(os.environ.get('V4_TILE_OVERLAP', '0') or 0),
            'tile_output_dim': int(os.environ.get('V4_TILE_OUTPUT_DIM', '0') or 0),
            'train_batch': int(config.batch_size),
            'val_batch': int(config.batch_size),
            'num_epochs': int(config.num_epochs),
            'lr': float(config.lr),
            'backbone_lr': float(config.lr),
            'use_amp': 'False',
            'init_ckpt': '',
            'generated_train_cfg': str(gen_cfg_fpath),
            'effective_train_scales': [int(config.input_h)],
            'effective_train_scale_min': int(config.input_h),
            'effective_train_scale_max': int(config.input_h),
        }
        (workdir / 'policy.json').write_text(json.dumps(policy, indent=2))
        print(f'  wrote {workdir / "policy.json"}')

        # Sanity: loss should decrease. Print a one-liner the smoke
        # test can grep for.
        if len(history) >= 2:
            first = history[0][2]
            last = history[-1][2]
            print(f'  first_loss={first:.4f} last_loss={last:.4f} '
                  f'delta={(first - last):+.4f}')


# ---------------------------------------------------------------------------
# Export subcommand
# ---------------------------------------------------------------------------

@V4MockCLI.register
class export(scfg.DataConfig):
    """Export the trained mock checkpoint to ONNX."""
    workdir = scfg.Value(None, help='per-cell workdir')
    export_h = scfg.Value(None, help='ONNX export height (defaults to ckpt meta)')
    export_w = scfg.Value(None, help='ONNX export width (defaults to ckpt meta)')
    out = scfg.Value(None, help='output .onnx path; defaults to <workdir>/export/<name>.onnx')

    @classmethod
    def main(cls, argv=1, **kwargs):
        import torch
        config = cls.cli(argv=argv, data=kwargs, strict=True)

        workdir = Path(str(config.workdir)).expanduser().resolve()
        ckpt_fpath = workdir / 'best_stg2.pth'
        if not ckpt_fpath.exists():
            raise FileNotFoundError(ckpt_fpath)
        ckpt = torch.load(ckpt_fpath, map_location='cpu')
        meta = ckpt.get('meta', {})
        H = int(config.export_h or meta.get('input_h', 320))
        W = int(config.export_w or meta.get('input_w', 320))
        K = int(meta.get('num_queries', NUM_QUERIES_DEFAULT))
        priors = meta.get('prior_boxes_norm')
        priors_t = (torch.tensor(priors, dtype=torch.float32)
                    if priors is not None else None)
        model = _build_model(num_queries=K, prior_boxes_norm=priors_t)
        model.load_state_dict(ckpt['model'])
        model.eval()

        export_dpath = workdir / 'export'
        export_dpath.mkdir(parents=True, exist_ok=True)
        out_fpath = (Path(str(config.out)).expanduser().resolve()
                     if config.out else
                     export_dpath / f'v4_mock_h{H}_w{W}.onnx')

        dummy_img = torch.zeros(1, 3, H, W, dtype=torch.float32)
        dummy_size = torch.tensor([[W, H]], dtype=torch.int64)
        torch.onnx.export(
            model,
            (dummy_img, dummy_size),
            str(out_fpath),
            input_names=['images', 'orig_target_sizes'],
            output_names=['labels', 'boxes', 'scores'],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={'images': {0: 'N'},
                          'orig_target_sizes': {0: 'N'}},
        )
        print(f'  exported -> {out_fpath}')


# ---------------------------------------------------------------------------
# Eval subcommand
# ---------------------------------------------------------------------------

@V4MockCLI.register
class evaluate(scfg.DataConfig):
    """Run the mock detector on a test kwcoco and write detect_metrics.json.

    Subcommand name is `evaluate` rather than `eval` so it doesn't shadow
    the Python builtin in this module.
    """
    workdir = scfg.Value(None, help='per-cell workdir')
    test_kwcoco = scfg.Value(None, help='test kwcoco to evaluate against')
    out_dir = scfg.Value(None, help='where to write detect_metrics.json')
    score_thresh = scfg.Value(0.30, help='score threshold for kept detections')
    input_h = scfg.Value(None, help='inference image height (defaults to ckpt meta)')
    input_w = scfg.Value(None, help='inference image width (defaults to ckpt meta)')

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco
        import kwimage
        import numpy as np
        import subprocess
        import torch
        config = cls.cli(argv=argv, data=kwargs, strict=True)

        workdir = Path(str(config.workdir)).expanduser().resolve()
        ckpt = torch.load(workdir / 'best_stg2.pth', map_location='cpu')
        meta = ckpt.get('meta', {})
        H = int(config.input_h or meta.get('input_h', 320))
        W = int(config.input_w or meta.get('input_w', 320))
        K = int(meta.get('num_queries', NUM_QUERIES_DEFAULT))
        priors = meta.get('prior_boxes_norm')
        priors_t = (torch.tensor(priors, dtype=torch.float32)
                    if priors is not None else None)
        model = _build_model(num_queries=K, prior_boxes_norm=priors_t)
        model.load_state_dict(ckpt['model'])
        model.eval()

        out_dir = Path(str(config.out_dir or (workdir / 'eval'))).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fpath = out_dir / 'detect_metrics.json'

        true = kwcoco.CocoDataset.coerce(str(config.test_kwcoco))

        pred = kwcoco.CocoDataset()
        pred.fpath = str(out_dir / 'pred_boxes.kwcoco.zip')
        cat_id = pred.add_category(name='poop')
        # Mirror image rows from true so kwcoco eval can join on gid.
        for img in true.images().objs:
            pred.add_image(**{k: v for k, v in img.items() if k != 'id'},
                           id=img['id'])

        for gid in list(true.images()):
            arr = true.coco_image(gid).imdelay().finalize()
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            try:
                resized = kwimage.imresize(arr, dsize=(W, H), interpolation='area')
            except NotImplementedError:
                resized = kwimage.imresize(arr, dsize=(W, H), interpolation='linear')
            chw = torch.from_numpy(
                (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...])
            sz = torch.tensor([[arr.shape[1], arr.shape[0]]], dtype=torch.int64)
            with torch.no_grad():
                _, p_boxes, p_scores = model(chw, sz)
            p_boxes = p_boxes[0].cpu().numpy()
            p_scores = p_scores[0].cpu().numpy()
            for k in range(p_boxes.shape[0]):
                s = float(p_scores[k])
                if s < float(config.score_thresh):
                    continue
                x1, y1, x2, y2 = [float(v) for v in p_boxes[k]]
                pred.add_annotation(
                    image_id=gid, category_id=cat_id,
                    bbox=[x1, y1, x2 - x1, y2 - y1], score=s,
                )
        pred.dump()

        # Same kwcoco eval the v3/v9 path uses, invoked as a subprocess
        # so the output matches what 04_eval_on_test.sh writes.
        cmd = [
            sys.executable, '-m', 'kwcoco', 'eval',
            '--true_dataset', str(config.test_kwcoco),
            '--pred_dataset', str(pred.fpath),
            '--out_dpath', str(out_dir),
            '--out_fpath', str(out_fpath),
            '--draw', 'False',
            '--iou_thresh', '0.5',
        ]
        subprocess.run(cmd, check=True)
        print(f'  wrote {out_fpath}')


__cli__ = V4MockCLI


if __name__ == '__main__':
    __cli__.main()
