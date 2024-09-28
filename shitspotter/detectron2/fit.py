#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class DetectronFitCLI(scfg.DataConfig):
    """
    Wrapper around detectron2 trainers
    """
    train_fpath = scfg.Value(None, help='param1')
    vali_fpath = scfg.Value(None, help='param1')
    expt_name = scfg.Value(None, help='param1')
    default_root_dir = scfg.Value('./out')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from shitspotter.detectron2.fit import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = DetectronFitCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        detectron_fit(config)


def detectron_fit(config):
    """
    References:
        https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

    Ignore:
        from shitspotter.detectron2.fit import *  # NOQA
        cmdline = 0
        import kwutil
        kwargs = kwutil.Yaml.coerce(
            '''
            default_root_dir: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v3
            expt_name: train_baseline_maskrcnn_v3
            train_fpath: /home/joncrall/data/dvc-repos/shitspotter_dvc/train_imgs5747_1e73d54f.mscoco.json
            vali_fpath: /home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json
            ''')
        cls = DetectronFitCLI
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

    """
    import os
    import ubelt as ub
    import detectron2  # NOQA
    from detectron2.data.datasets import register_coco_instances
    from detectron2 import model_zoo
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    # full_fpath = ub.Path(shitspotter.util.find_shit_coco_fpath())
    # bundle_dpath = full_fpath.parent
    # vali_fpath = bundle_dpath / 'train_imgs5747_1e73d54f.mscoco.json'

    import kwutil
    proc_context = kwutil.ProcessContext(
        name='shitspotter.detectron2.fit',
        config=kwutil.Json.ensure_serializable(dict(config)),
        track_emissions=True,
    )
    proc_context.start()
    print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')

    dataset_paths = {
        'vali': ub.Path(config.vali_fpath),
        'train': ub.Path(config.train_fpath),
    }

    dataset_infos = {}
    for key, fpath in dataset_paths.items():
        assert fpath.exists()
        row = {'path': fpath}
        row['name'] = fpath.name.split('.', 1)[0]
        dataset_infos[key] = row
    for key, row in dataset_infos.items():
        register_coco_instances(row['name'], {}, row['path'], row['path'].parent)

    # It would be nice if detectron had a resource path, but oh well...
    # detectron_repo_dpath = ub.Path(detectron2.__file__).parent.parent
    # shitspotter_repo_dpath = ub.Path(shitspotter.__file__).parent.parent

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = (dataset_infos['train']['name'],)
    # cfg.DATASETS.TEST = (dataset_infos['vali']['name'],)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2   # This is the real 'batch size' commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025   # pick a good LR
    cfg.SOLVER.MAX_ITER = 120_000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []          # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The 'RoIHead batch size'. 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.OUTPUT_DIR = None  # hack: null out for the initial
    hashid = ub.hash_data(cfg)[0:8]
    # expt_name = f'{config.expt_name}_{hashid}'
    # train_prefix = ub.Path('~/data/dvc-repos/shitspotter_expt_dvc/training').expand()
    # output_dpath = (train_prefix / platform.node() / os.environ['USER'] / 'ShitSpotter' / 'runs' / expt_name)
    # print(hashid)
    output_dpath = ub.Path(config.default_root_dir) / f'v_{hashid}'
    output_dpath.ensuredir()
    cfg.OUTPUT_DIR = os.fspath(output_dpath)
    print(ub.urepr(cfg, nl=-1))

    telemetry_fpath1 = output_dpath / 'initial_telemetry.json'
    telemetry_fpath1.write_text(kwutil.Json.dumps(proc_context.obj))

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    proc_context.stop()
    print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')

    telemetry_fpath2 = output_dpath / 'final_telemetry.json'
    telemetry_fpath2.write_text(kwutil.Json.dumps(proc_context.obj))

__cli__ = DetectronFitCLI

if __name__ == '__main__':
    """

    CommandLine:
        python -m shitspotter.detectron2.fit
    """
    __cli__.main()
