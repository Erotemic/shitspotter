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
from shitspotter.algo_foundation_v3.segmenter_sam2 import validate_segmenter_assets


def _coerce_yaml_overrides(text):
    if text in [None, '', {}]:
        return {}
    if isinstance(text, dict):
        return text
    return yaml.safe_load(text) or {}


class AlgoTrainCLI(scfg.ModalCLI):
    """
    Train or prepare assets for foundation_detseg_v3.
    """


@AlgoTrainCLI.register
class detector(scfg.DataConfig):
    train_kwcoco = scfg.Value(None, help='training kwcoco path', required=True)
    vali_kwcoco = scfg.Value(None, help='validation kwcoco path', required=True)
    test_kwcoco = scfg.Value(None, help='optional test kwcoco path')
    workdir = scfg.Value('./runs/foundation_detseg_v3/deimv2', help='work directory')
    variant = scfg.Value('deimv2_m', help='detector preset', choices=['deimv2_m', 'deimv2_s'])
    init_checkpoint_fpath = scfg.Value(None, help='optional checkpoint to fine-tune from')
    device = scfg.Value(None, help='torch device passed to DEIMv2 train.py')
    use_amp = scfg.Value(False, help='if True pass --use-amp to train.py')
    config_overrides = scfg.Value(None, help='YAML fragment merged into generated DEIMv2 train config')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        detector_cfg = model_registry.resolve_detector_preset(config.variant)
        train_config_fpath = train_detector(
            train_kwcoco=config.train_kwcoco,
            vali_kwcoco=config.vali_kwcoco,
            test_kwcoco=config.test_kwcoco,
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
    variant = scfg.Value('sam2.1_hiera_base_plus', help='segmenter preset')
    checkpoint_fpath = scfg.Value(None, help='optional SAM2 checkpoint path')
    config_fpath = scfg.Value(None, help='optional SAM2 config path')
    package_out = scfg.Value(None, help='optional output package yaml')
    metadata_name = scfg.Value(None, help='optional package name')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        segmenter_cfg = model_registry.resolve_segmenter_preset(config.variant)
        if config.checkpoint_fpath is not None:
            segmenter_cfg['checkpoint_fpath'] = str(Path(config.checkpoint_fpath).expanduser())
        if config.config_fpath is not None:
            segmenter_cfg['config_fpath'] = str(Path(config.config_fpath).expanduser())
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
