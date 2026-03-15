"""
Thin wrappers around SAM2 image prediction and fine-tuning.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from shitspotter.algo_foundation_v3.config_utils import deep_update, ensuredir


def resolve_repo_dpath(segmenter_cfg):
    envvar = segmenter_cfg.get('repo_envvar', 'SHITSPOTTER_SAM2_REPO_DPATH')
    repo = segmenter_cfg.get('repo_dpath', os.environ.get(envvar, None))
    if repo is None:
        return None
    return Path(repo).expanduser().resolve()


def _ensure_repo_on_path(repo_dpath):
    repo_str = str(repo_dpath)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _ensure_pycocotools():
    try:
        from pycocotools import mask as mask_utils  # noqa: F401
    except Exception as ex:
        raise ImportError(
            'pycocotools is required for SAM2 fine-tuning data export.'
        ) from ex


def _usable_category_names(src_dset, category_names=None):
    if category_names is not None:
        return list(category_names)
    ann_cids = {ann.get('category_id', None) for ann in src_dset.annots().objs}
    names = []
    for cid in sorted(c for c in ann_cids if c is not None):
        cat = src_dset.cats.get(cid, None)
        if cat is not None:
            names.append(cat['name'])
    return names


def _safe_link_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        dst.write_bytes(src.read_bytes())


def _segmentation_to_rle(segmentation, dims):
    import kwimage
    import numpy as np
    from pycocotools import mask as mask_utils

    h, w = map(int, dims)
    mask = kwimage.Segmentation.coerce(segmentation, dims=(h, w)).to_mask(dims=(h, w))
    data = np.asfortranarray(mask.data.astype('uint8'))
    rle = mask_utils.encode(data[:, :, None])[0]
    counts = rle.get('counts', None)
    if isinstance(counts, bytes):
        rle['counts'] = counts.decode('ascii')
    return rle


def _export_sam2_split(src, split_dpath, split_name, category_names=None):
    import kwcoco

    src_dset = kwcoco.CocoDataset.coerce(src)
    category_names = set(_usable_category_names(src_dset, category_names))
    image_dpath = ensuredir(split_dpath / 'images')
    gt_dpath = ensuredir(split_dpath / 'annotations')
    file_list = []
    records = []

    for img in src_dset.images().objs:
        gid = img['id']
        anns = src_dset.annots(gid=gid).objs
        exported_anns = []
        for ann in anns:
            cid = ann.get('category_id', None)
            if cid is None:
                continue
            cat = src_dset.cats.get(cid, None)
            if cat is None or cat['name'] not in category_names:
                continue
            seg = ann.get('segmentation', None)
            if seg is None:
                continue
            height = img.get('height', None)
            width = img.get('width', None)
            if height is None or width is None:
                img_fpath = src_dset.get_image_fpath(gid)
                import kwimage
                shape = kwimage.load_image_shape(img_fpath)
                height, width = shape[0:2]
            rle = _segmentation_to_rle(seg, (height, width))
            area = float(ann.get('area', 0.0))
            if not area:
                bbox = ann.get('bbox', None)
                if bbox is not None:
                    area = float(bbox[2] * bbox[3])
            exported_anns.append({
                'segmentation': rle,
                'area': area,
                'category_name': cat['name'],
                'source_annotation_id': ann.get('id', None),
            })
        if not exported_anns:
            continue
        stem = f'sa_{int(gid):08d}'
        src_img_fpath = src_dset.get_image_fpath(gid)
        dst_img_fpath = image_dpath / f'{stem}.jpg'
        _safe_link_or_copy(src_img_fpath, dst_img_fpath)
        ann_fpath = gt_dpath / f'{stem}.json'
        ann_fpath.write_text(json.dumps({'annotations': exported_anns}))
        file_list.append(stem)
        records.append({
            'gid': gid,
            'stem': stem,
            'source_image_fpath': str(Path(src_img_fpath).resolve()),
            'num_annotations': len(exported_anns),
        })

    file_list_fpath = split_dpath / f'{split_name}.txt'
    file_list_fpath.write_text(''.join(f'{stem}\n' for stem in file_list))
    metadata = {
        'split': split_name,
        'num_images': len(file_list),
        'category_names': sorted(category_names),
        'records': records,
    }
    metadata_fpath = split_dpath / 'metadata.json'
    metadata_fpath.write_text(json.dumps(metadata, indent=2))
    return {
        'image_dpath': image_dpath,
        'gt_dpath': gt_dpath,
        'file_list_fpath': file_list_fpath,
        'metadata_fpath': metadata_fpath,
    }


def export_sam2_training_splits(train_kwcoco, vali_kwcoco, output_dpath, category_names=None):
    _ensure_pycocotools()
    output_dpath = ensuredir(output_dpath)
    exports = {
        'train': _export_sam2_split(
            train_kwcoco,
            output_dpath / 'train',
            split_name='train',
            category_names=category_names,
        ),
        'vali': _export_sam2_split(
            vali_kwcoco,
            output_dpath / 'vali',
            split_name='vali',
            category_names=category_names,
        ),
    }
    return exports


def _find_training_template(segmenter_cfg):
    repo_dpath = resolve_repo_dpath(segmenter_cfg)
    if repo_dpath is None:
        raise FileNotFoundError('Set SHITSPOTTER_SAM2_REPO_DPATH or repo_dpath to fine-tune SAM2')
    training_template_fpath = segmenter_cfg.get('training_template_fpath', None)
    if training_template_fpath is not None:
        return Path(training_template_fpath).expanduser().resolve()
    relpath = segmenter_cfg.get('training_template_relpath', None)
    if relpath is None:
        raise NotImplementedError(
            f'No training template registered for segmenter variant={segmenter_cfg.get("variant", None)!r}'
        )
    return (repo_dpath / relpath).resolve()


def _resolve_inference_config_name(segmenter_cfg):
    repo_dpath = resolve_repo_dpath(segmenter_cfg)
    config_name = segmenter_cfg.get('hydra_config_name', None)
    if config_name is not None:
        return str(config_name).replace('\\', '/')

    config_relpath = segmenter_cfg.get('config_relpath', None)
    if config_relpath is not None:
        config_relpath = str(config_relpath).replace('\\', '/')
        if config_relpath.startswith('sam2/'):
            return config_relpath[len('sam2/'):]
        return config_relpath

    config_fpath = segmenter_cfg.get('config_fpath', None)
    if config_fpath is not None and repo_dpath is not None:
        relpath = Path(config_fpath).expanduser().resolve().relative_to(repo_dpath)
        relpath = str(relpath).replace('\\', '/')
        if relpath.startswith('sam2/'):
            return relpath[len('sam2/'):]
        return relpath

    return None


def _dump_hydra_global_yaml(data, fpath):
    text = '# @package _global_\n\n' + yaml.safe_dump(data, sort_keys=False)
    Path(fpath).write_text(text)


def build_sam2_training_config(
    segmenter_cfg,
    prepared,
    workdir,
    init_checkpoint_fpath,
    train_kwargs=None,
):
    train_kwargs = train_kwargs or {}
    repo_dpath = resolve_repo_dpath(segmenter_cfg)
    template_fpath = _find_training_template(segmenter_cfg)
    template = yaml.safe_load(template_fpath.read_text())
    workdir = Path(workdir).expanduser().resolve()
    generated_dpath = ensuredir(workdir / 'generated_configs')

    resolution = int(train_kwargs.get('resolution', 1024))
    train_batch_size = int(train_kwargs.get('train_batch_size', 1))
    num_train_workers = int(train_kwargs.get('num_train_workers', 8))
    num_epochs = int(train_kwargs.get('num_epochs', 20))
    base_lr = float(train_kwargs.get('base_lr', 5e-6))
    vision_lr = float(train_kwargs.get('vision_lr', 3e-6))
    max_num_objects = int(train_kwargs.get('max_num_objects', 8))
    multiplier = int(train_kwargs.get('multiplier', 1))
    checkpoint_save_freq = int(train_kwargs.get('checkpoint_save_freq', 1))
    num_gpus = int(train_kwargs.get('num_gpus', 1))
    config_overrides = train_kwargs.get('config_overrides', {}) or {}

    template['scratch']['resolution'] = resolution
    template['scratch']['train_batch_size'] = train_batch_size
    template['scratch']['num_train_workers'] = num_train_workers
    template['scratch']['num_frames'] = 1
    template['scratch']['max_num_objects'] = max_num_objects
    template['scratch']['base_lr'] = base_lr
    template['scratch']['vision_lr'] = vision_lr
    template['scratch']['num_epochs'] = num_epochs

    template['dataset']['img_folder'] = str(prepared.train_image_dpath)
    template['dataset']['gt_folder'] = str(prepared.train_gt_dpath)
    template['dataset']['file_list_txt'] = str(prepared.train_file_list_fpath)
    template['dataset']['multiplier'] = multiplier

    video_dataset = template['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0]['video_dataset']
    video_dataset['_target_'] = 'training.dataset.vos_raw_dataset.SA1BRawDataset'
    video_dataset['img_folder'] = '${dataset.img_folder}'
    video_dataset['gt_folder'] = '${dataset.gt_folder}'
    video_dataset['file_list_txt'] = '${dataset.file_list_txt}'
    video_dataset['num_frames'] = 1
    video_dataset.pop('is_palette', None)
    video_dataset.pop('single_object_mode', None)
    video_dataset.pop('truncate_video', None)
    video_dataset.pop('frames_sampling_mult', None)
    video_dataset.setdefault('mask_area_frac_thresh', 1.1)
    video_dataset.setdefault('uncertain_iou', -1)

    sampler = template['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0]['sampler']
    sampler['num_frames'] = 1
    sampler['max_num_objects'] = max_num_objects

    model_cfg = template['trainer']['model']
    model_cfg['prob_to_use_pt_input_for_train'] = float(train_kwargs.get('prob_to_use_pt_input_for_train', 1.0))
    model_cfg['prob_to_use_box_input_for_train'] = float(train_kwargs.get('prob_to_use_box_input_for_train', 1.0))
    model_cfg['prob_to_sample_from_gt_for_train'] = float(train_kwargs.get('prob_to_sample_from_gt_for_train', 0.5))
    model_cfg['num_frames_to_correct_for_train'] = 1
    model_cfg['rand_frames_to_correct_for_train'] = False
    model_cfg['num_init_cond_frames_for_train'] = 1
    model_cfg['rand_init_cond_frames_for_train'] = False
    model_cfg['add_all_frames_to_correct_as_cond'] = True
    model_cfg['num_frames_to_correct_for_eval'] = 1
    model_cfg['num_init_cond_frames_for_eval'] = 1

    template['trainer']['checkpoint']['save_dir'] = '${launcher.experiment_log_dir}/checkpoints'
    template['trainer']['checkpoint']['save_freq'] = checkpoint_save_freq
    template['trainer']['checkpoint']['model_weight_initializer']['state_dict']['checkpoint_path'] = str(
        Path(init_checkpoint_fpath).expanduser().resolve()
    )

    template['launcher']['num_nodes'] = int(train_kwargs.get('num_nodes', 1))
    template['launcher']['gpus_per_node'] = num_gpus
    template['launcher']['experiment_log_dir'] = str(workdir)
    template['submitit']['use_cluster'] = bool(train_kwargs.get('use_cluster', False))
    if train_kwargs.get('submitit_cpus_per_task', None) is not None:
        template['submitit']['cpus_per_task'] = int(train_kwargs['submitit_cpus_per_task'])

    resolved = deep_update(template, config_overrides)

    logical_name = f'foundation_v3_{segmenter_cfg.get("variant", "sam2").replace(".", "_")}_{workdir.name}'
    repo_config_relpath = Path('sam2/configs/shitspotter_training') / f'{logical_name}.yaml'
    hydra_config_name = Path('configs/shitspotter_training') / f'{logical_name}.yaml'
    repo_config_fpath = repo_dpath / repo_config_relpath
    repo_config_fpath.parent.mkdir(parents=True, exist_ok=True)
    _dump_hydra_global_yaml(resolved, repo_config_fpath)

    workdir_config_fpath = generated_dpath / 'train_sam2.yaml'
    _dump_hydra_global_yaml(resolved, workdir_config_fpath)

    metadata = {
        'repo_config_relpath': str(repo_config_relpath).replace('\\', '/'),
        'hydra_config_name': str(hydra_config_name).replace('\\', '/'),
        'repo_config_fpath': str(repo_config_fpath),
        'workdir_config_fpath': str(workdir_config_fpath),
        'expected_checkpoint_fpath': str(workdir / 'checkpoints' / 'checkpoint.pt'),
        'prepared_train_metadata_fpath': str(prepared.train_metadata_fpath),
        'prepared_vali_metadata_fpath': str(prepared.vali_metadata_fpath),
    }
    metadata_fpath = generated_dpath / 'train_sam2_metadata.json'
    metadata_fpath.write_text(json.dumps(metadata, indent=2))
    return metadata


def train_segmenter(
    train_kwcoco,
    vali_kwcoco,
    workdir,
    segmenter_cfg,
    init_checkpoint_fpath,
    package_out=None,
    metadata_name=None,
    train_kwargs=None,
):
    from shitspotter.algo_foundation_v3.datasets import prepare_segmenter_training_data
    from shitspotter.algo_foundation_v3.packaging import build_package, dump_package

    if init_checkpoint_fpath is None:
        raise ValueError('SAM2 fine-tuning requires init_checkpoint_fpath')
    validate_segmenter_assets({**segmenter_cfg, 'checkpoint_fpath': init_checkpoint_fpath})

    workdir = Path(workdir).expanduser().resolve()
    prepared = prepare_segmenter_training_data(
        train_kwcoco=train_kwcoco,
        vali_kwcoco=vali_kwcoco,
        output_dpath=workdir / 'prepared_data' / 'sam2',
        category_names=(train_kwargs or {}).get('category_names', None),
    )

    metadata = build_sam2_training_config(
        segmenter_cfg=segmenter_cfg,
        prepared=prepared,
        workdir=workdir,
        init_checkpoint_fpath=init_checkpoint_fpath,
        train_kwargs=train_kwargs,
    )
    repo_dpath = resolve_repo_dpath(segmenter_cfg)
    command = [
        sys.executable,
        str((repo_dpath / 'training' / 'train.py').resolve()),
        '-c',
        metadata['hydra_config_name'],
        '--num-gpus',
        str(int((train_kwargs or {}).get('num_gpus', 1))),
    ]
    if (train_kwargs or {}).get('use_cluster', False):
        command += ['--use-cluster', '1']
    subprocess.run(command, cwd=repo_dpath, check=True)

    package_fpath = None
    detector_checkpoint_fpath = (train_kwargs or {}).get('detector_checkpoint_fpath', None)
    if package_out is not None and detector_checkpoint_fpath is not None:
        train_meta = json.loads(Path(prepared.train_metadata_fpath).read_text())
        package = build_package(
            backend='deimv2_sam2',
            detector_checkpoint_fpath=detector_checkpoint_fpath,
            segmenter_preset=segmenter_cfg.get('variant', None),
            segmenter_checkpoint_fpath=metadata['expected_checkpoint_fpath'],
            metadata_name=metadata_name,
        )
        package.setdefault('segmenter', {})
        package['segmenter']['checkpoint_fpath'] = metadata['expected_checkpoint_fpath']
        package['segmenter']['training_repo_config_fpath'] = metadata['repo_config_fpath']
        package['segmenter']['training_workdir'] = str(workdir)
        package.setdefault('metadata', {})
        package['metadata']['segmenter_training_categories'] = train_meta.get('category_names', [])
        package_fpath = dump_package(package, package_out)

    return {
        'workdir': workdir,
        'expected_checkpoint_fpath': metadata['expected_checkpoint_fpath'],
        'repo_config_fpath': metadata['repo_config_fpath'],
        'workdir_config_fpath': metadata['workdir_config_fpath'],
        'prepared_train_metadata_fpath': str(prepared.train_metadata_fpath),
        'prepared_vali_metadata_fpath': str(prepared.vali_metadata_fpath),
        'package_fpath': None if package_fpath is None else str(package_fpath),
    }


class SAM2Segmenter:
    def __init__(self, segmenter_cfg):
        self.segmenter_cfg = segmenter_cfg
        self.predictor = None

    def _lazy_init(self):
        if self.predictor is not None:
            return
        repo_dpath = resolve_repo_dpath(self.segmenter_cfg)
        if repo_dpath is not None:
            _ensure_repo_on_path(repo_dpath)
        try:
            SAM2ImagePredictor = importlib.import_module('sam2.sam2_image_predictor').SAM2ImagePredictor
        except Exception as ex:
            raise ImportError(
                'Unable to import SAM2. Install it or set SHITSPOTTER_SAM2_REPO_DPATH.'
            ) from ex

        checkpoint_fpath = self.segmenter_cfg.get('checkpoint_fpath', None)
        config_name = _resolve_inference_config_name(self.segmenter_cfg)

        if checkpoint_fpath and config_name:
            build_sam2 = importlib.import_module('sam2.build_sam').build_sam2
            predictor = SAM2ImagePredictor(
                build_sam2(
                    str(config_name),
                    str(checkpoint_fpath),
                    device=self.segmenter_cfg.get('device', 'cuda:0'),
                ),
                mask_threshold=float(self.segmenter_cfg.get('mask_threshold', 0.0)),
            )
        else:
            hf_model_id = self.segmenter_cfg.get('hf_model_id', None)
            if hf_model_id is None:
                raise KeyError('Segmenter package requires checkpoint+config or hf_model_id')
            predictor = SAM2ImagePredictor.from_pretrained(
                hf_model_id,
                device=self.segmenter_cfg.get('device', 'cuda:0'),
            )
        self.predictor = predictor

    def predict_masks_for_boxes(self, image, boxes_xyxy):
        self._lazy_init()
        self.predictor.set_image(image)
        mask_infos = []
        for box in boxes_xyxy:
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
                return_logits=False,
                normalize_coords=False,
            )
            best_idx = int(scores.argmax()) if len(scores) else 0
            mask_infos.append({
                'mask': masks[best_idx],
                'score': float(scores[best_idx]) if len(scores) else 0.0,
            })
        return mask_infos


def validate_segmenter_assets(segmenter_cfg):
    repo_dpath = resolve_repo_dpath(segmenter_cfg)
    if repo_dpath is not None and not repo_dpath.exists():
        raise FileNotFoundError(repo_dpath)
    checkpoint_fpath = segmenter_cfg.get('checkpoint_fpath', None)
    if checkpoint_fpath is not None and not Path(checkpoint_fpath).expanduser().exists():
        raise FileNotFoundError(checkpoint_fpath)
    config_fpath = segmenter_cfg.get('config_fpath', None)
    if config_fpath is not None and not Path(config_fpath).expanduser().exists():
        raise FileNotFoundError(config_fpath)
