"""
CLI for building and validating packaged model configs.
"""

import scriptconfig as scfg
import ubelt as ub

from shitspotter.algo_foundation_v3.packaging import (
    build_package,
    dump_package,
    load_package,
    resolve_package,
    validate_package,
)


class AlgoPackageCLI(scfg.ModalCLI):
    """
    Build, inspect, and validate foundation v3 model packages.
    """


@AlgoPackageCLI.register
class build(scfg.DataConfig):
    backend = scfg.Value('deimv2_sam2', help='backend preset to build')
    dst = scfg.Value(None, position=1, help='path to write the package yaml')
    detector_preset = scfg.Value(None, help='override detector preset')
    segmenter_preset = scfg.Value(None, help='override segmenter preset')
    baseline_preset = scfg.Value(None, help='override baseline preset')
    detector_checkpoint_fpath = scfg.Value(None, help='detector checkpoint path')
    detector_config_fpath = scfg.Value(None, help='detector config path (required for opengroundingdino)')
    segmenter_checkpoint_fpath = scfg.Value(None, help='segmenter checkpoint path')
    baseline_checkpoint_fpath = scfg.Value(None, help='baseline checkpoint path')
    metadata_name = scfg.Value(None, help='optional package name')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        package = build_package(
            backend=config.backend,
            detector_preset=config.detector_preset,
            segmenter_preset=config.segmenter_preset,
            baseline_preset=config.baseline_preset,
            detector_checkpoint_fpath=config.detector_checkpoint_fpath,
            detector_config_fpath=config.detector_config_fpath,
            segmenter_checkpoint_fpath=config.segmenter_checkpoint_fpath,
            baseline_checkpoint_fpath=config.baseline_checkpoint_fpath,
            metadata_name=config.metadata_name,
        )
        if config.dst is not None:
            dump_package(package, config.dst)
        print(ub.urepr(package, nl=1))


@AlgoPackageCLI.register
class inspect(scfg.DataConfig):
    package_fpath = scfg.Value(None, position=1, help='package yaml to inspect')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        package = resolve_package(package_fpath=config.package_fpath)
        print(ub.urepr(package, nl=1))


@AlgoPackageCLI.register
class validate(scfg.DataConfig):
    package_fpath = scfg.Value(None, position=1, help='package yaml to validate')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        package = load_package(config.package_fpath)
        validate_package(package)
        resolve_package(package_fpath=config.package_fpath)
        print(f'Validated {config.package_fpath}')


__cli__ = AlgoPackageCLI


if __name__ == '__main__':
    __cli__.main()
