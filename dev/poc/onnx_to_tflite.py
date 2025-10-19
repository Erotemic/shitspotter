#!/usr/bin/env python3
"""
Convert an ONNX model to a TensorFlow Lite flatbuffer with optional quantization.

Supports:
    - float32 (default)
    - float16
    - int8 (dynamic-range quantization)
    - full-int8 (integer quantization with representative dataset)
    - uint8 (same as full-int8 but with unsigned I/O)
    - float8 (placeholder; not yet supported by TensorFlow Lite)

Examples:

    python tools/convert_onnx_to_tflite.py \
        yolox_nano.onnx yolox_nano_fp16.tflite \
        --dtype float16

    python tools/convert_onnx_to_tflite.py \
        yolox_nano.onnx yolox_nano_int8_full.tflite \
        --dtype full-int8 \
        --rep-data data/kwcoco/train.kwcoco.json \
        --num-calib-samples 200

Requirements:
    # This requires older versions of onnx and tensorflow
    we pyenv3.11.9
    uv pip install "onnx==1.15.0" "onnx-tf==1.10.0" "tensorflow==2.15.*" "tensorflow_probability==0.23.0"

"""

from __future__ import annotations
import logging
import pathlib
import tempfile
import random

import numpy as np

import kwcoco
from PIL import Image
import scriptconfig as scfg

_LOGGER = logging.getLogger(__name__)


class ConvertConfig(scfg.DataConfig):
    """
    Configuration for ONNX → TFLite conversion.
    """

    input = scfg.Value(..., help="Path to the ONNX model.", required=True, position=1)
    output = scfg.Value(..., help="Destination path for the .tflite output.", required=True, position=2)

    dtype = scfg.Value(
        "float32",
        help=(
            "Quantization mode / target dtype. Options:\n"
            " - float32 (no quantization)\n"
            " - float16\n"
            " - int8 (dynamic-range)\n"
            " - full-int8 (integer quant with representative data)\n"
            " - uint8 (integer quant with unsigned I/O)\n"
            " - float8 (placeholder)\n"
        ),
        choices=["float32", "float16", "int8", "full-int8", "uint8", "float8"],
    )

    calibration_data = scfg.Value(None, help="Path to kwcoco dataset JSON for calibration.")
    num_calib_samples = scfg.Value(100, help="Number of calibration samples from the dataset.")
    input_shape = scfg.Value("1,416,416,3", type=str, help="Input tensor shape (NHWC). Default: 1,416,416,3.")
    normalize = scfg.Value(True, isflag=True, help="Normalize calibration images by dividing by 255.0.")
    optimize = scfg.Value(None, help="TFLite optimization hint.", choices=["default", "size", "latency"])


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_shape(shape_str: str):
    return tuple(int(s.strip()) for s in shape_str.split(","))


def _converter_optimisation(opt: str | None):
    import tensorflow as tf
    if opt is None:
        return None
    mapping = {
        "default": tf.lite.Optimize.DEFAULT,
        "size": tf.lite.Optimize.OPTIMIZE_FOR_SIZE,
        "latency": tf.lite.Optimize.OPTIMIZE_FOR_LATENCY,
    }
    return mapping[opt]


def _make_kwcoco_representative_dataset(
    json_path: pathlib.Path,
    num_samples: int,
    input_shape: tuple[int, int, int, int],
    normalize: bool,
):
    """
    Yields representative samples for full-int8 quantization from a kwcoco dataset.
    """
    import kwarray
    rng = kwarray.ensure_rng(2143432432, api='python')
    ds = kwcoco.CocoDataset(str(json_path))
    gids = list(ds.images())
    rng.shuffle(gids)
    gids = gids[:max(1, num_samples)]

    _, H, W, C = input_shape
    _LOGGER.info("Using %d kwcoco images for calibration", len(gids))

    def gen():
        for gid in gids:
            img_fpath = ds.coco_image(gid).primary_image_filepath()
            if not img_fpath.exists():
                continue
            img = Image.open(img_fpath).convert("RGB").resize((W, H), Image.BILINEAR)
            arr = np.asarray(img).astype(np.float32)

            if arr.shape != (H, W, C):
                arr = np.resize(arr, (H, W, C))

            if normalize:
                arr /= 255.0

            # Hack to handle nchw shape
            arr = np.transpose(arr, (2, 0, 1))   # (3, H, W)
            arr = np.expand_dims(arr, axis=0)
            print(f'arr.shape={arr.shape}')
            result = [arr]
            yield result

    return gen


# -----------------------------------------------------------------------------
# Core conversion logic
# -----------------------------------------------------------------------------

def convert_onnx_to_tflite(cfg: ConvertConfig):
    import tensorflow as tf
    import onnx
    import ubelt as ub
    np.object = object
    from onnx_tf.backend import prepare

    onnx_path = ub.Path(cfg.input)
    output_path = ub.Path(cfg.output)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    input_shape = _parse_shape(cfg.input_shape)
    _LOGGER.info(f"input_shape: {input_shape!r}")

    _LOGGER.info("Loading ONNX model: %s", onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx_input_shape = onnx_model.graph.input[0].type.tensor_type.shape
    _LOGGER.info(f'onnx_input_shape={onnx_input_shape}')

    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = pathlib.Path(tmpdir, "saved_model")

        _LOGGER.info("Exporting intermediate SavedModel to %s", export_dir)
        tf_rep = prepare(onnx_model)

        # import tf2onnx
        # onnx_model = onnx.load(onnx_path)
        # tf_rep, _ = tf2onnx.convert.from_onnx_model(onnx_model)
        # tf.saved_model.save(tf_rep, "saved_model_dir")

        tf_rep.export_graph(str(export_dir))

        _LOGGER.info("Converting SavedModel to TFLite (dtype=%s)...", cfg.dtype)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))

        opt = _converter_optimisation(cfg.optimize)
        if opt is not None:
            converter.optimizations = [opt]

        dtype = cfg.dtype.lower()

        if dtype == "float32":
            _LOGGER.info("No quantization (float32 model).")

        elif dtype == "float16":
            _LOGGER.info("Applying float16 quantization.")
            if not converter.optimizations:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        elif dtype == "int8":
            _LOGGER.info("Applying int8 dynamic-range quantization.")
            if not converter.optimizations:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

        elif dtype in {"full-int8", "uint8"}:
            if not cfg.calibration_data:
                raise ValueError("--dtype=full-int8 or uint8 requires --rep-data (kwcoco dataset).")
            _LOGGER.info("Applying full integer quantization using kwcoco data: %s", cfg.calibration_data)
            if not converter.optimizations:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            rep_gen = _make_kwcoco_representative_dataset(
                cfg.calibration_data,
                num_samples=cfg.num_calib_samples,
                input_shape=input_shape,
                normalize=cfg.normalize,
            )
            converter.representative_dataset = rep_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            if dtype == "uint8":
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            else:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

        elif dtype == "float8":
            _LOGGER.warning("Float8 quantization is not currently supported by TensorFlow Lite. Producing float32 model instead.")

        else:
            raise ValueError(f"Unsupported dtype mode: {dtype}")

        tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    _LOGGER.info("Wrote TFLite model to %s (%.2f MiB)", output_path, len(tflite_model) / (1024 * 1024))


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = ConvertConfig.cli(strict=True, verbose='auto')
    convert_onnx_to_tflite(cfg)


if __name__ == "__main__":
    main()
