"""
Higher-level dataset preparation for training and prediction workflows.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shitspotter.algo_foundation_v3.coco_adapter import export_training_splits
from shitspotter.algo_foundation_v3.kwcoco_adapter import resolve_prediction_paths
from shitspotter.algo_foundation_v3.segmenter_sam2 import export_sam2_training_splits


@dataclass
class PreparedTrainingData:
    output_dpath: Path
    train_coco_fpath: Path
    vali_coco_fpath: Path
    test_coco_fpath: Optional[Path]


@dataclass
class PreparedSAM2TrainingData:
    output_dpath: Path
    train_image_dpath: Path
    train_gt_dpath: Path
    train_file_list_fpath: Path
    train_metadata_fpath: Path
    vali_image_dpath: Path
    vali_gt_dpath: Path
    vali_file_list_fpath: Path
    vali_metadata_fpath: Path


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


def prepare_segmenter_training_data(
    train_kwcoco,
    vali_kwcoco,
    output_dpath,
    category_names=None,
):
    exports = export_sam2_training_splits(
        train_kwcoco=train_kwcoco,
        vali_kwcoco=vali_kwcoco,
        output_dpath=output_dpath,
        category_names=category_names,
    )
    train = exports['train']
    vali = exports['vali']
    return PreparedSAM2TrainingData(
        output_dpath=Path(output_dpath),
        train_image_dpath=train['image_dpath'],
        train_gt_dpath=train['gt_dpath'],
        train_file_list_fpath=train['file_list_fpath'],
        train_metadata_fpath=train['metadata_fpath'],
        vali_image_dpath=vali['image_dpath'],
        vali_gt_dpath=vali['gt_dpath'],
        vali_file_list_fpath=vali['file_list_fpath'],
        vali_metadata_fpath=vali['metadata_fpath'],
    )


def prepare_prediction_io(src, dst, package_name):
    return resolve_prediction_paths(src=src, dst=dst, package_name=package_name)
