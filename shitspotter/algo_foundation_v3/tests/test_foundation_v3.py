import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import kwcoco
import kwimage
import numpy as np
import ubelt as ub

from shitspotter.algo_foundation_v3 import coco_adapter
from shitspotter.algo_foundation_v3 import kwcoco_adapter
from shitspotter.algo_foundation_v3 import packaging
from shitspotter.algo_foundation_v3 import cli_predict
from shitspotter.algo_foundation_v3 import cli_predict_gtboxes
from shitspotter.algo_foundation_v3 import model_registry
from shitspotter.algo_foundation_v3 import detector_deimv2
from shitspotter.algo_foundation_v3 import polygon_utils
from shitspotter.algo_foundation_v3 import segmenter_sam2
from shitspotter.algo_foundation_v3.datasets import prepare_segmenter_training_data


def _demo_dataset(tmp_path):
    img_fpath = tmp_path / 'demo.png'
    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    canvas[16:40, 20:44, 1] = 255
    kwimage.imwrite(img_fpath, canvas)

    poly = kwimage.Polygon.from_coco([20, 16, 44, 16, 44, 40, 20, 40])
    dset = kwcoco.CocoDataset()
    dset.add_category(name='poop', id=1)
    gid = dset.add_image(file_name=str(img_fpath), width=64, height=64, name='demo')
    dset.add_annotation(
        image_id=gid,
        category_id=1,
        bbox=poly.box().to_coco(),
        segmentation=poly.to_coco(style='new'),
        area=float(poly.area),
    )
    dset.fpath = tmp_path / 'demo.kwcoco.zip'
    dset.dump()
    return dset.fpath, img_fpath


def test_foundation_module_imports():
    modules = [
        'shitspotter.algo_foundation_v3.cli_train',
        'shitspotter.algo_foundation_v3.cli_predict',
        'shitspotter.algo_foundation_v3.cli_predict_boxes',
        'shitspotter.algo_foundation_v3.cli_predict_gtboxes',
        'shitspotter.algo_foundation_v3.cli_export_labelme',
        'shitspotter.algo_foundation_v3.cli_package',
        'shitspotter.algo_foundation_v3.packaging',
        'shitspotter.algo_foundation_v3.kwcoco_adapter',
        'shitspotter.algo_foundation_v3.coco_adapter',
        'shitspotter.algo_foundation_v3.detector_deimv2',
        'shitspotter.algo_foundation_v3.segmenter_sam2',
        'shitspotter.algo_foundation_v3.baseline_maskdino',
    ]
    for modname in modules:
        importlib.import_module(modname)


def test_cli_help_smoke():
    commands = [
        [sys.executable, '-m', 'shitspotter.algo_foundation_v3.cli_train', '--help'],
        [sys.executable, '-m', 'shitspotter.algo_foundation_v3.cli_train', 'detector', '--help'],
        [sys.executable, '-m', 'shitspotter.algo_foundation_v3.cli_train', 'segmenter', '--help'],
        [sys.executable, '-m', 'shitspotter.algo_foundation_v3.cli_train', 'baseline-maskdino', '--help'],
        [sys.executable, '-m', 'shitspotter.algo_foundation_v3.cli_predict', '--help'],
        [sys.executable, '-m', 'shitspotter.algo_foundation_v3.cli_predict_boxes', '--help'],
        [sys.executable, '-m', 'shitspotter.algo_foundation_v3.cli_predict_gtboxes', '--help'],
        [sys.executable, '-m', 'shitspotter.algo_foundation_v3.cli_export_labelme', '--help'],
        [sys.executable, '-m', 'shitspotter.algo_foundation_v3.cli_package', '--help'],
    ]
    for command in commands:
        proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.returncode == 0, proc.stderr


def test_package_build_and_resolve(tmp_path):
    package = packaging.build_package(
        backend='deimv2_sam2',
        detector_checkpoint_fpath='/tmp/detector.pth',
        segmenter_checkpoint_fpath='/tmp/segmenter.pt',
        metadata_name='demo-package',
    )
    package_fpath = tmp_path / 'package.yaml'
    packaging.dump_package(package, package_fpath)
    resolved = packaging.resolve_package(package_fpath=package_fpath)
    assert resolved['backend'] == 'deimv2_sam2'
    assert resolved['metadata']['name'] == 'demo-package'
    assert resolved['detector']['checkpoint_fpath'] == '/tmp/detector.pth'


def test_deimv2_infer_num_classes_from_state():
    class FakeWeight:
        def __init__(self, shape):
            self.shape = shape
            self.ndim = len(shape)

    state = {'decoder.enc_score_head.weight': FakeWeight((1, 256))}
    assert detector_deimv2._infer_num_classes_from_state(state) == 1

    state = {'decoder.denoising_class_embed.weight': FakeWeight((81, 256))}
    assert detector_deimv2._infer_num_classes_from_state(state) == 80


def test_sam2_inference_config_name_resolution():
    cfg = {'config_relpath': 'sam2/configs/sam2.1/sam2.1_hiera_b+.yaml'}
    assert segmenter_sam2._resolve_inference_config_name(cfg) == 'configs/sam2.1/sam2.1_hiera_b+.yaml'

    cfg = {'hydra_config_name': 'configs/shitspotter_training/demo.yaml'}
    assert segmenter_sam2._resolve_inference_config_name(cfg) == 'configs/shitspotter_training/demo.yaml'


def test_mask_to_multi_polygon_tolerates_degenerate_polygons(monkeypatch):
    class BadPoly:
        @property
        def area(self):
            raise ValueError('A linearring requires at least 4 coordinates.')

        def simplify(self, *_args, **_kwargs):
            return self

    class DummyMultiPolygon:
        def __init__(self, data):
            self.data = data

    class DummyMask:
        def to_multi_polygon(self):
            return DummyMultiPolygon([BadPoly()])

    class DummyMaskCoerce:
        @staticmethod
        def coerce(_mask):
            return DummyMask()

    monkeypatch.setattr('kwimage.Mask', DummyMaskCoerce)
    monkeypatch.setattr('kwimage.MultiPolygon', DummyMultiPolygon)
    result = polygon_utils.mask_to_multi_polygon(np.ones((1, 1), dtype=np.uint8))
    assert result.data == []


def test_image_dir_to_kwcoco_and_coco_export(tmp_path):
    dset_fpath, img_fpath = _demo_dataset(tmp_path)
    image_dpath = tmp_path / 'images'
    image_dpath.mkdir()
    copied = image_dpath / img_fpath.name
    copied.write_bytes(img_fpath.read_bytes())

    paths = kwcoco_adapter.resolve_prediction_paths(image_dpath, package_name='demo')
    src_fpath, was_generated = kwcoco_adapter.coerce_input_kwcoco(image_dpath, paths)
    assert was_generated
    assert ub.Path(src_fpath).exists()

    exports = coco_adapter.export_training_splits(
        train_kwcoco=dset_fpath,
        vali_kwcoco=dset_fpath,
        output_dpath=tmp_path / 'mscoco',
        include_segmentations=True,
    )
    train_data = json.loads(ub.Path(exports['train']).read_text())
    assert len(train_data['images']) == 1
    assert len(train_data['annotations']) == 1
    assert train_data['categories'][0]['name'] == 'poop'
    assert train_data['categories'][0]['id'] == 0
    assert train_data['annotations'][0]['category_id'] == 0


def test_coco_export_skips_null_and_non_target_categories(tmp_path):
    dset_fpath, _ = _demo_dataset(tmp_path)
    dset = kwcoco.CocoDataset(dset_fpath)
    dset.add_category(name='unknown', id=2)
    gid = next(iter(dset.imgs))
    dset.add_annotation(
        image_id=gid,
        category_id=2,
        bbox=[1, 1, 4, 4],
        area=16,
    )
    dset.dataset['annotations'].append({
        'id': max(dset.anns) + 1,
        'image_id': gid,
        'category_id': None,
        'bbox': [2, 2, 5, 5],
        'area': 25,
    })
    dset.dump(dset_fpath)

    exports = coco_adapter.export_training_splits(
        train_kwcoco=dset_fpath,
        vali_kwcoco=dset_fpath,
        output_dpath=tmp_path / 'mscoco_invalid',
        include_segmentations=True,
    )
    train_data = json.loads(ub.Path(exports['train']).read_text())
    assert len(train_data['annotations']) == 1
    assert train_data['annotations'][0]['category_id'] == 0


def test_predict_write_and_labelme_export_with_fake_backend(tmp_path, monkeypatch):
    dset_fpath, img_fpath = _demo_dataset(tmp_path)

    package = packaging.build_package(
        backend='deimv2_sam2',
        detector_checkpoint_fpath='/tmp/detector.pth',
        segmenter_checkpoint_fpath='/tmp/segmenter.pt',
        metadata_name='fake-deimv2-sam2',
    )
    package_fpath = tmp_path / 'package.yaml'
    packaging.dump_package(package, package_fpath)

    class FakeDetector:
        def __init__(self, detector_cfg):
            self.detector_cfg = detector_cfg

        def predict_image_records(self, image):
            return [{'label': 0, 'bbox_ltrb': [18, 14, 46, 42], 'score': 0.95}]

    class FakeSegmenter:
        def __init__(self, segmenter_cfg):
            self.segmenter_cfg = segmenter_cfg

        def predict_masks_for_boxes(self, image, boxes_xyxy):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[16:40, 20:44] = 1
            return [{'mask': mask, 'score': 0.99} for _ in boxes_xyxy]

    monkeypatch.setattr(cli_predict, 'DEIMv2Predictor', FakeDetector)
    monkeypatch.setattr(cli_predict, 'SAM2Segmenter', FakeSegmenter)

    pred_fpath = tmp_path / 'pred.kwcoco.zip'
    cli_predict.AlgoPredictCLI.main(
        argv=0,
        src=dset_fpath,
        dst=pred_fpath,
        package_fpath=package_fpath,
        create_labelme=False,
    )

    pred_dset = kwcoco.CocoDataset(pred_fpath)
    assert len(pred_dset.annots()) == 1
    ann = pred_dset.annots().objs[0]
    assert ann['role'] == 'prediction'
    assert ann['foundation_backend'] == 'deimv2_sam2'
    assert ann.get('segmentation', None) is not None
    assert ann['foundation_prompt_source'] == 'detector_box'
    assert ann['detector_bbox'] == [18.0, 14.0, 28.0, 28.0]
    assert ann['prompt_bbox'] == [0.0, 0.0, 64.0, 64.0]

    labelme_fpath = img_fpath.with_suffix('.json')
    if labelme_fpath.exists():
        labelme_fpath.unlink()
    written = kwcoco_adapter.export_predictions_to_labelme(pred_fpath, only_missing=True)
    assert labelme_fpath in written
    written_again = kwcoco_adapter.export_predictions_to_labelme(pred_fpath, only_missing=True)
    assert written_again == []

    copy_dst = tmp_path / 'labelme_copy'
    if labelme_fpath.exists():
        labelme_fpath.unlink()
    copied_written = kwcoco_adapter.export_predictions_to_labelme(
        pred_fpath,
        only_missing=True,
        copy_dst=copy_dst,
    )
    assert copied_written
    copied_sidecar = copied_written[0]
    assert copied_sidecar.parent == copy_dst
    copied_image = copy_dst / f'{next(iter(pred_dset.imgs)):08d}_{img_fpath.name}'
    assert copied_image.exists()
    assert not labelme_fpath.exists()


def test_predict_gtboxes_records_prompt_metadata(tmp_path, monkeypatch):
    dset_fpath, _img_fpath = _demo_dataset(tmp_path)

    package = packaging.build_package(
        backend='deimv2_sam2',
        detector_checkpoint_fpath='/tmp/detector.pth',
        segmenter_checkpoint_fpath='/tmp/segmenter.pt',
        metadata_name='fake-deimv2-sam2',
    )
    package_fpath = tmp_path / 'package_gtboxes.yaml'
    packaging.dump_package(package, package_fpath)

    class FakeSegmenter:
        def __init__(self, segmenter_cfg):
            self.segmenter_cfg = segmenter_cfg

        def predict_masks_for_boxes(self, image, boxes_xyxy):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[16:40, 20:44] = 1
            return [{'mask': mask, 'score': 0.99} for _ in boxes_xyxy]

    monkeypatch.setattr(cli_predict_gtboxes, 'SAM2Segmenter', FakeSegmenter)

    pred_fpath = tmp_path / 'pred_gtboxes.kwcoco.zip'
    cli_predict_gtboxes.AlgoPredictGTBoxesCLI.main(
        argv=0,
        src=dset_fpath,
        dst=pred_fpath,
        package_fpath=package_fpath,
        create_labelme=False,
        crop_padding=0,
        polygon_simplify=0,
        min_component_area=0,
        keep_largest_component=False,
    )

    pred_dset = kwcoco.CocoDataset(pred_fpath)
    ann = pred_dset.annots().objs[0]
    assert ann['foundation_backend'] == 'sam2_gtboxes'
    assert ann['foundation_prompt_source'] == 'truth_box'
    assert ann['source_gt_ann_id'] is not None
    assert ann['source_gt_bbox'] == [20.0, 16.0, 24.0, 24.0]
    assert ann['prompt_bbox'] == [20.0, 16.0, 24.0, 24.0]


def test_sam2_training_bundle_and_config_generation(tmp_path):
    dset_fpath, _ = _demo_dataset(tmp_path)
    prepared = prepare_segmenter_training_data(
        train_kwcoco=dset_fpath,
        vali_kwcoco=dset_fpath,
        output_dpath=tmp_path / 'sam2_bundle',
    )
    train_meta = json.loads(prepared.train_metadata_fpath.read_text())
    assert train_meta['num_images'] == 1
    assert train_meta['category_names'] == ['poop']
    train_stem = train_meta['records'][0]['stem']
    train_ann = json.loads((prepared.train_gt_dpath / f'{train_stem}.json').read_text())
    assert len(train_ann['annotations']) == 1
    assert 'segmentation' in train_ann['annotations'][0]
    assert 'counts' in train_ann['annotations'][0]['segmentation']

    fake_ckpt = tmp_path / 'sam2_init.pt'
    fake_ckpt.write_bytes(b'fake')
    segmenter_cfg = model_registry.resolve_segmenter_preset('sam2.1_hiera_base_plus')
    repo_root = Path(__file__).resolve().parents[3]
    segmenter_cfg['repo_dpath'] = str((repo_root / 'tpl/segment-anything-2').resolve())
    meta = segmenter_sam2.build_sam2_training_config(
        segmenter_cfg=segmenter_cfg,
        prepared=prepared,
        workdir=tmp_path / 'sam2_workdir',
        init_checkpoint_fpath=fake_ckpt,
        train_kwargs={
            'resolution': 512,
            'train_batch_size': 1,
            'num_train_workers': 2,
            'num_epochs': 3,
            'num_gpus': 1,
            'category_names': ['poop'],
        },
    )
    cfg_text = Path(meta['workdir_config_fpath']).read_text()
    assert meta['hydra_config_name'].startswith('configs/shitspotter_training/')
    assert cfg_text.startswith('# @package _global_')
    assert 'training.dataset.vos_raw_dataset.SA1BRawDataset' in cfg_text
    assert str(prepared.train_image_dpath) in cfg_text
    assert str(fake_ckpt) in cfg_text


def test_foundation_pipeline_with_fake_geowatch(monkeypatch):
    fake_pipeline_nodes = types.ModuleType('geowatch.mlops.pipeline_nodes')

    class FakePort:
        def __init__(self, name):
            self.name = name
            self.connections = []

        def connect(self, other):
            self.connections.append(other)

    class FakeProcessNode:
        def __init__(self):
            self.inputs = {key: FakePort(key) for key in getattr(self, 'in_paths', {})}
            self.outputs = {key: FakePort(key) for key in getattr(self, 'out_paths', {})}

    class FakePipelineDAG:
        def __init__(self, nodes):
            self.nodes = nodes

        def build_nx_graphs(self):
            return None

    fake_pipeline_nodes.ProcessNode = FakeProcessNode
    fake_pipeline_nodes.PipelineDAG = FakePipelineDAG

    fake_mlops = types.ModuleType('geowatch.mlops')
    fake_mlops.pipeline_nodes = fake_pipeline_nodes
    fake_geowatch = types.ModuleType('geowatch')
    fake_geowatch.mlops = fake_mlops

    monkeypatch.setitem(sys.modules, 'geowatch', fake_geowatch)
    monkeypatch.setitem(sys.modules, 'geowatch.mlops', fake_mlops)
    monkeypatch.setitem(sys.modules, 'geowatch.mlops.pipeline_nodes', fake_pipeline_nodes)

    import shitspotter.algo_foundation_v3.mlops as foundation_mlops
    import shitspotter.pipelines as pipelines

    foundation_mlops = importlib.reload(foundation_mlops)
    pipelines = importlib.reload(pipelines)
    dag = pipelines.foundation_v3_evaluation_pipeline()
    assert 'foundation_v3_pred' in dag.nodes
    assert 'detection_evaluation' in dag.nodes
