"""
Training CLI for the foundation v3 pipeline.
"""

from pathlib import Path

import scriptconfig as scfg
import ubelt as ub
import yaml

from shitspotter.algo_foundation_v3 import model_registry
from shitspotter.algo_foundation_v3.baseline_maskdino import train_maskdino
from shitspotter.algo_foundation_v3.detector_deimv2 import train_detector
from shitspotter.algo_foundation_v3.packaging import build_package, dump_package
from shitspotter.algo_foundation_v3.segmenter_sam2 import (
    train_segmenter,
    validate_segmenter_assets,
)


def _coerce_yaml_overrides(text):
    if text in [None, '', {}]:
        return {}
    if isinstance(text, dict):
        return text
    return yaml.safe_load(text) or {}


def _coerce_yaml_value(text):
    if text in [None, '']:
        return None
    return yaml.safe_load(text)


class AlgoTrainCLI(scfg.ModalCLI):
    """
    Train or prepare assets for foundation_detseg_v3.
    """


@AlgoTrainCLI.register
class detector(scfg.DataConfig):
    train_kwcoco = scfg.Value(None, help='training kwcoco path')
    vali_kwcoco = scfg.Value(None, help='validation kwcoco path')
    test_kwcoco = scfg.Value(None, help='optional test kwcoco path')
    train_coco_json = scfg.Value(None, help='training COCO / MSCOCO json path')
    vali_coco_json = scfg.Value(None, help='validation COCO / MSCOCO json path')
    test_coco_json = scfg.Value(None, help='optional test COCO / MSCOCO json path')
    workdir = scfg.Value('./runs/foundation_detseg_v3/deimv2', help='work directory')
    variant = scfg.Value('deimv2_m', help='detector preset', choices=['deimv2_m', 'deimv2_s'])
    init_checkpoint_fpath = scfg.Value(None, help='optional checkpoint to fine-tune from')
    device = scfg.Value(None, help='torch device passed to DEIMv2 train.py')
    use_amp = scfg.Value(False, help='if True pass --use-amp to train.py')
    config_overrides = scfg.Value(None, help='YAML fragment merged into generated DEIMv2 train config')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        have_kwcoco = config.train_kwcoco is not None or config.vali_kwcoco is not None
        have_coco = config.train_coco_json is not None or config.vali_coco_json is not None
        if have_kwcoco and have_coco:
            raise ValueError('Specify kwcoco inputs or coco_json inputs, not both')
        if have_coco:
            if config.train_coco_json is None or config.vali_coco_json is None:
                raise ValueError('train_coco_json and vali_coco_json are required together')
        else:
            if config.train_kwcoco is None or config.vali_kwcoco is None:
                raise ValueError('train_kwcoco and vali_kwcoco are required when coco_json inputs are not provided')
        detector_cfg = model_registry.resolve_detector_preset(config.variant)
        train_config_fpath = train_detector(
            train_kwcoco=config.train_kwcoco,
            vali_kwcoco=config.vali_kwcoco,
            test_kwcoco=config.test_kwcoco,
            train_coco_json=config.train_coco_json,
            vali_coco_json=config.vali_coco_json,
            test_coco_json=config.test_coco_json,
            workdir=config.workdir,
            detector_cfg=detector_cfg,
            init_checkpoint_fpath=config.init_checkpoint_fpath,
            device=config.device,
            use_amp=config.use_amp,
            config_overrides=_coerce_yaml_overrides(config.config_overrides),
        )
        print(f'Generated detector training config at {train_config_fpath}')


@AlgoTrainCLI.register
class segmenter(scfg.DataConfig):
    train_kwcoco = scfg.Value(None, help='optional training kwcoco path; if set, launch SAM2 fine-tuning')
    vali_kwcoco = scfg.Value(None, help='optional validation kwcoco path used for bundle export metadata')
    workdir = scfg.Value('./runs/foundation_detseg_v3/sam2', help='segmenter work directory')
    variant = scfg.Value('sam2.1_hiera_base_plus', help='segmenter preset')
    checkpoint_fpath = scfg.Value(None, help='SAM2 checkpoint path; required for fine-tuning, optional for asset validation')
    config_fpath = scfg.Value(None, help='optional SAM2 config path')
    package_out = scfg.Value(None, help='optional output package yaml')
    metadata_name = scfg.Value(None, help='optional package name')
    detector_checkpoint_fpath = scfg.Value(None, help='optional detector checkpoint path used to build a runnable deimv2_sam2 package after SAM2 fine-tuning')
    resolution = scfg.Value(1024, help='training resolution for SAM2 fine-tuning')
    train_batch_size = scfg.Value(1, help='training batch size for SAM2 fine-tuning')
    num_train_workers = scfg.Value(8, help='training dataloader workers for SAM2 fine-tuning')
    num_epochs = scfg.Value(20, help='number of epochs for SAM2 fine-tuning')
    num_gpus = scfg.Value(1, help='gpus-per-node passed to the SAM2 trainer')
    base_lr = scfg.Value(5e-6, help='base learning rate for SAM2 fine-tuning')
    vision_lr = scfg.Value(3e-6, help='image encoder learning rate for SAM2 fine-tuning')
    max_num_objects = scfg.Value(8, help='maximum number of objects sampled per image for SAM2 fine-tuning')
    multiplier = scfg.Value(1, help='repeat-factor multiplier for the training split')
    checkpoint_save_freq = scfg.Value(1, help='checkpoint save frequency in epochs')
    category_names = scfg.Value(None, help='optional YAML list of category names to keep for SAM2 fine-tuning')
    config_overrides = scfg.Value(None, help='optional YAML fragment merged into the generated SAM2 training config')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        segmenter_cfg = model_registry.resolve_segmenter_preset(config.variant)
        if config.checkpoint_fpath is not None:
            segmenter_cfg['checkpoint_fpath'] = str(Path(config.checkpoint_fpath).expanduser())
        if config.config_fpath is not None:
            segmenter_cfg['config_fpath'] = str(Path(config.config_fpath).expanduser())
        if config.train_kwcoco:
            if not config.vali_kwcoco:
                raise ValueError('vali_kwcoco is required when launching SAM2 fine-tuning')
            result = train_segmenter(
                train_kwcoco=config.train_kwcoco,
                vali_kwcoco=config.vali_kwcoco,
                workdir=config.workdir,
                segmenter_cfg=segmenter_cfg,
                init_checkpoint_fpath=config.checkpoint_fpath,
                package_out=config.package_out,
                metadata_name=config.metadata_name,
                train_kwargs={
                    'detector_checkpoint_fpath': config.detector_checkpoint_fpath,
                    'resolution': config.resolution,
                    'train_batch_size': config.train_batch_size,
                    'num_train_workers': config.num_train_workers,
                    'num_epochs': config.num_epochs,
                    'num_gpus': config.num_gpus,
                    'base_lr': config.base_lr,
                    'vision_lr': config.vision_lr,
                    'max_num_objects': config.max_num_objects,
                    'multiplier': config.multiplier,
                    'checkpoint_save_freq': config.checkpoint_save_freq,
                    'category_names': _coerce_yaml_value(config.category_names),
                    'config_overrides': _coerce_yaml_overrides(config.config_overrides),
                },
            )
            print(ub.urepr(result, nl=1))
        else:
            validate_segmenter_assets(segmenter_cfg)
            package = build_package(
                backend='deimv2_sam2',
                segmenter_preset=config.variant,
                segmenter_checkpoint_fpath=config.checkpoint_fpath,
                metadata_name=config.metadata_name,
            )
            if config.config_fpath is not None:
                package['segmenter']['config_fpath'] = str(Path(config.config_fpath).expanduser())
            if config.package_out is not None:
                dump_package(package, config.package_out)
            print(ub.urepr(package, nl=1))


@AlgoTrainCLI.register
class baseline_maskdino(scfg.DataConfig):
    __command__ = 'baseline-maskdino'
    train_kwcoco = scfg.Value(None, help='training kwcoco path', required=True)
    vali_kwcoco = scfg.Value(None, help='validation kwcoco path', required=True)
    test_kwcoco = scfg.Value(None, help='optional test kwcoco path')
    workdir = scfg.Value('./runs/foundation_detseg_v3/maskdino', help='work directory')
    variant = scfg.Value('maskdino_r50', help='baseline preset')
    init_checkpoint_fpath = scfg.Value(None, help='optional checkpoint to fine-tune from')
    ims_per_batch = scfg.Value(None, help='optional solver batch size override')
    base_lr = scfg.Value(None, help='optional learning rate override')
    max_iter = scfg.Value(None, help='optional max iter override')
    num_workers = scfg.Value(None, help='optional dataloader worker override')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        baseline_cfg = model_registry.resolve_baseline_preset(config.variant)
        out_dpath = train_maskdino(
            train_kwcoco=config.train_kwcoco,
            vali_kwcoco=config.vali_kwcoco,
            test_kwcoco=config.test_kwcoco,
            workdir=config.workdir,
            baseline_cfg=baseline_cfg,
            init_checkpoint_fpath=config.init_checkpoint_fpath,
            ims_per_batch=config.ims_per_batch,
            base_lr=config.base_lr,
            max_iter=config.max_iter,
            num_workers=config.num_workers,
        )
        print(f'MaskDINO outputs written to {out_dpath}')


__cli__ = AlgoTrainCLI


if __name__ == '__main__':
    __cli__.main()
