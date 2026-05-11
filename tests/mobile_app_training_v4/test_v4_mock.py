"""Tests for v4_mock — the tiny torch detector used for pipeline smoke tests.

Verifies that:

  1. The model produces DEIMv2-shaped outputs (labels, boxes, scores)
     with the right dtype/shape.
  2. Boxes are in pixel coords w.r.t. orig_target_sizes.
  3. The scalar gate is the only "obviously trainable" parameter and
     a few gradient steps push it from off (~0.075) to on (~1.0).
  4. End-to-end train -> save -> load -> ONNX export -> ORT inference
     all preserve outputs within tolerance.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_model_output_shapes_and_dtypes(synthetic_kwcoco):
    import torch
    import v4_mock

    priors = v4_mock._collect_prior_boxes(str(synthetic_kwcoco), num_queries=8)
    assert priors.shape == (8, 4)
    assert (priors >= 0.0).all() and (priors <= 1.0).all()

    model = v4_mock._build_model(num_queries=8, prior_boxes_norm=priors)
    images = torch.zeros(2, 3, 64, 64, dtype=torch.float32)
    sizes = torch.tensor([[1280, 960], [800, 600]], dtype=torch.int64)
    labels, boxes, scores = model(images, sizes)
    assert labels.shape == (2, 8) and labels.dtype == torch.int64
    assert boxes.shape == (2, 8, 4) and boxes.dtype == torch.float32
    assert scores.shape == (2, 8) and scores.dtype == torch.float32

    # Boxes for image 0 should land in [0, 1280] x [0, 960].
    assert (boxes[0, :, 0] >= 0).all() and (boxes[0, :, 2] <= 1280 + 1e-3).all()
    assert (boxes[0, :, 1] >= 0).all() and (boxes[0, :, 3] <= 960 + 1e-3).all()


def test_model_initial_gate_is_off():
    import torch
    import v4_mock
    model = v4_mock._build_model(num_queries=8)
    images = torch.zeros(1, 3, 64, 64)
    sizes = torch.tensor([[640, 640]], dtype=torch.int64)
    _, _, scores = model(images, sizes)
    # Initial gate ~ -2.5 -> sigmoid ~ 0.075 — well below typical 0.30.
    assert scores.max() < 0.20


def test_few_gradient_steps_flip_gate(synthetic_kwcoco):
    """A handful of "be confident" steps push the gate from off to on."""
    import torch
    import v4_mock

    priors = v4_mock._collect_prior_boxes(str(synthetic_kwcoco), num_queries=8)
    model = v4_mock._build_model(num_queries=8, prior_boxes_norm=priors)
    opt = torch.optim.Adam(model.parameters(), lr=1.0)

    images = torch.rand(4, 3, 64, 64)
    sizes = torch.full((4, 2), 640, dtype=torch.int64)
    target = torch.ones((4,), dtype=torch.float32)  # "be confident"

    initial = model(images, sizes)[2].max().item()
    for _ in range(20):
        opt.zero_grad()
        _, _, scores = model(images, sizes)
        loss = torch.nn.functional.binary_cross_entropy(
            scores[:, 0].clamp(1e-6, 1 - 1e-6), target)
        loss.backward()
        opt.step()
    final = model(images, sizes)[2].max().item()

    assert initial < 0.20, f'initial scores should be off: {initial}'
    assert final > 0.80, f'gate should flip on after a few steps: {final}'


def test_train_export_load_roundtrip(synthetic_kwcoco, tmp_path):
    """Train mock -> save .pth -> reload -> ONNX export -> ORT inference,
    verifying outputs survive the full pipeline within tolerance.

    Drives v4_mock as a subprocess so the test goes through the same
    CLI dispatch path that the sweep uses, rather than the (currently
    None) class binding the @register decorator leaves on the module.
    """
    import subprocess
    import sys

    import numpy as np
    import torch

    workdir = tmp_path / 'v4_mock_run'
    workdir.mkdir()
    v4_mock_py = (Path(__file__).resolve().parents[2]
                  / 'experiments' / 'mobile_app_training_v4' / 'v4_mock.py')

    subprocess.run([
        sys.executable, str(v4_mock_py), 'train',
        '--train_kwcoco', str(synthetic_kwcoco),
        '--vali_kwcoco', str(synthetic_kwcoco),
        '--workdir', str(workdir),
        '--input_h', '128', '--input_w', '128',
        '--num_epochs', '2', '--batch_size', '2',
        '--num_queries', '8', '--lr', '1.0', '--seed', '0',
    ], check=True)
    assert (workdir / 'best_stg2.pth').exists()
    assert (workdir / 'policy.json').exists()

    onnx_fpath = workdir / 'export' / 'v4_mock_h128_w128.onnx'
    subprocess.run([
        sys.executable, str(v4_mock_py), 'export',
        '--workdir', str(workdir),
        '--export_h', '128', '--export_w', '128',
    ], check=True)
    assert onnx_fpath.exists()

    # Inference parity: torch vs ORT on a constant image.
    import onnxruntime as ort
    import v4_mock as _v4_mock_mod
    sess = ort.InferenceSession(str(onnx_fpath), providers=['CPUExecutionProvider'])
    img_np = np.zeros((1, 3, 128, 128), dtype=np.float32)
    sz_np = np.array([[640, 480]], dtype=np.int64)
    inputs = {sess.get_inputs()[0].name: img_np,
              sess.get_inputs()[1].name: sz_np}
    onnx_labels, onnx_boxes, onnx_scores = sess.run(None, inputs)

    ckpt = torch.load(workdir / 'best_stg2.pth', map_location='cpu', weights_only=False)
    priors = torch.tensor(ckpt['meta']['prior_boxes_norm'], dtype=torch.float32)
    model = _v4_mock_mod._build_model(num_queries=8, prior_boxes_norm=priors)
    model.load_state_dict(ckpt['model'])
    model.eval()
    with torch.no_grad():
        torch_labels, torch_boxes, torch_scores = model(
            torch.zeros(1, 3, 128, 128), torch.tensor([[640, 480]], dtype=torch.int64))

    np.testing.assert_allclose(onnx_boxes, torch_boxes.numpy(), atol=1e-4)
    np.testing.assert_allclose(onnx_scores, torch_scores.numpy(), atol=1e-4)


def test_collect_prior_boxes_handles_missing_category(tmp_path):
    """If the kwcoco bundle has no 'poop' category, we still get K priors."""
    import kwcoco
    import v4_mock
    dset = kwcoco.CocoDataset()
    dset.add_category(name='not_poop')
    dset.add_image(file_name='nonexistent.jpg', width=100, height=100, id=1)
    dset.fpath = str(tmp_path / 'd.kwcoco.json')
    dset.dump()
    priors = v4_mock._collect_prior_boxes(str(dset.fpath), num_queries=8)
    assert priors.shape == (8, 4)
    # No GT -> all-neutral fallback.
    assert (priors == priors[0]).all()
