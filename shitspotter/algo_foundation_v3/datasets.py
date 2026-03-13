"""
Higher-level dataset preparation for training and prediction workflows.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shitspotter.algo_foundation_v3.coco_adapter import export_training_splits
from shitspotter.algo_foundation_v3.kwcoco_adapter import resolve_prediction_paths


@dataclass
class PreparedTrainingData:
    output_dpath: Path
    train_coco_fpath: Path
    vali_coco_fpath: Path
    test_coco_fpath: Optional[Path]


def prepare_detector_training_data(train_kwcoco, vali_kwcoco, output_dpath, test_kwcoco=None):
    exports = export_training_splits(
        train_kwcoco=train_kwcoco,
        vali_kwcoco=vali_kwcoco,
        test_kwcoco=test_kwcoco,
        output_dpath=output_dpath,
        include_segmentations=False,
    )
    return PreparedTrainingData(
        output_dpath=Path(output_dpath),
        train_coco_fpath=exports['train'],
        vali_coco_fpath=exports['vali'],
        test_coco_fpath=exports.get('test', None),
    )


def prepare_maskdino_training_data(train_kwcoco, vali_kwcoco, output_dpath, test_kwcoco=None):
    exports = export_training_splits(
        train_kwcoco=train_kwcoco,
        vali_kwcoco=vali_kwcoco,
        test_kwcoco=test_kwcoco,
        output_dpath=output_dpath,
        include_segmentations=True,
    )
    return PreparedTrainingData(
        output_dpath=Path(output_dpath),
        train_coco_fpath=exports['train'],
        vali_coco_fpath=exports['vali'],
        test_coco_fpath=exports.get('test', None),
    )


def prepare_prediction_io(src, dst, package_name):
    return resolve_prediction_paths(src=src, dst=dst, package_name=package_name)
