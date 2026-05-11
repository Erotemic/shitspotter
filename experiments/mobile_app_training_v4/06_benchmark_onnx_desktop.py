#!/usr/bin/env python3
"""
Time an exported DEIMv2 ONNX model on the desktop CPU.

This is a *desktop* proxy — Pixel 5 latency is what actually matters, but
desktop CPU latency is a useful early signal: if a 320x320 model takes
200 ms on a desktop x86 CPU, it is unlikely to hit 10 FPS on a Pixel 5.

Usage::

    python 06_benchmark_onnx_desktop.py \\
        --onnx /…/runs/deimv2_n_tile_g2_320x320/export/deimv2_n_h320_w320.onnx \\
        --image $SHITSPOTTER_DPATH/tpl/poop_models/dog.jpg \\
        --warmup 5 --iters 50

Outputs a one-line summary plus a pXX latency table.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import scriptconfig as scfg


class BenchCLI(scfg.DataConfig):
    onnx = scfg.Value(None, help='path to exported .onnx', required=True)
    image = scfg.Value(None, help='input image; if omitted, use a random tensor', alias=['img'])
    warmup = scfg.Value(5, help='untimed warmup iterations')
    iters = scfg.Value(50, help='timed iterations')
    threads = scfg.Value(0, help='ONNX Runtime intra-op threads (0 = ORT default)')
    provider = scfg.Value('cpu', help='cpu | cuda | dml | nnapi | …')
    dump_json = scfg.Value(None, help='optional output json with per-iter timings')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _make_session(config):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if int(config.threads) > 0:
        so.intra_op_num_threads = int(config.threads)
    providers = {
        'cpu': ['CPUExecutionProvider'],
        'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'dml': ['DmlExecutionProvider', 'CPUExecutionProvider'],
        'nnapi': ['NnapiExecutionProvider', 'CPUExecutionProvider'],
    }.get(str(config.provider).lower(), [str(config.provider) + 'ExecutionProvider'])
    sess = ort.InferenceSession(str(config.onnx), so, providers=providers)
    return sess


def _build_input(sess, image_path):
    import numpy as np
    in0 = sess.get_inputs()[0]
    in1 = sess.get_inputs()[1]
    n, c, h, w = in0.shape
    h = int(h) if isinstance(h, int) else 320
    w = int(w) if isinstance(w, int) else 320
    if image_path:
        from PIL import Image
        pil = Image.open(image_path).convert('RGB').resize((w, h))
        arr = np.asarray(pil).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, ...]
        orig_w, orig_h = pil.size
    else:
        arr = np.random.rand(1, 3, h, w).astype(np.float32)
        orig_w, orig_h = w, h
    sz = np.array([[orig_w, orig_h]], dtype=np.int64)
    return {in0.name: arr, in1.name: sz}


def run(config):
    import onnxruntime as ort
    print(f'onnxruntime={ort.__version__} providers={ort.get_available_providers()}')

    sess = _make_session(config)
    feeds = _build_input(sess, config.image)
    in0_shape = sess.get_inputs()[0].shape
    print(f'input shape={in0_shape} provider={sess.get_providers()[0]}')

    # warmup
    for _ in range(int(config.warmup)):
        sess.run(None, feeds)
    timings = []
    for _ in range(int(config.iters)):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings.sort()

    def _pct(p):
        if not timings:
            return float('nan')
        idx = max(0, min(len(timings) - 1, int(round((p / 100.0) * (len(timings) - 1)))))
        return timings[idx]

    mean_ms = statistics.fmean(timings)
    fps = 1000.0 / mean_ms if mean_ms > 0 else float('nan')
    print(f'\nlatency (ms): p50={_pct(50):.2f}  p90={_pct(90):.2f}  '
          f'p99={_pct(99):.2f}  mean={mean_ms:.2f}  min={timings[0]:.2f}  max={timings[-1]:.2f}')
    print(f'effective fps (single-threaded forward) = {fps:.1f}')

    if config.dump_json:
        Path(str(config.dump_json)).parent.mkdir(parents=True, exist_ok=True)
        Path(str(config.dump_json)).write_text(json.dumps({
            'onnx': str(config.onnx),
            'image': str(config.image) if config.image else None,
            'provider': sess.get_providers()[0],
            'threads': int(config.threads),
            'iters': int(config.iters),
            'warmup': int(config.warmup),
            'timings_ms': timings,
            'mean_ms': mean_ms,
            'fps': fps,
        }, indent=2))


__cli__ = BenchCLI


if __name__ == '__main__':
    __cli__.main()
