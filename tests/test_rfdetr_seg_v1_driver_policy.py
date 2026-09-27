from pathlib import Path
import importlib.util


DRIVER_FPATH = (
    Path(__file__).resolve().parents[1]
    / 'experiments' / 'rfdetr_seg_v1' / 'driver.py'
)
SPEC = importlib.util.spec_from_file_location('rfdetr_seg_v1_driver', DRIVER_FPATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_annotated_distractor_policy_defaults_and_validation():
    policy = {'seed': 10}
    got = driver._annotated_distractor_policy(policy)
    assert got == {
        'fraction': 0.0,
        'seed': 1019,
        'min_annotation_coverage': 0.5,
        'max_per_source': 64,
    }

    policy = {
        'seed': 10,
        'annotated_distractors': {
            'fraction': 0.2,
            'seed': 99,
            'min_annotation_coverage': 0.75,
            'max_per_source': 8,
        },
    }
    got = driver._annotated_distractor_policy(policy)
    assert got['fraction'] == 0.2
    assert got['seed'] == 99
    assert got['min_annotation_coverage'] == 0.75
    assert got['max_per_source'] == 8


def test_candidate_distractor_categories_uses_annotation_coverage():
    boxes = {
        1: [
            {'xyxy': (10, 10, 20, 20), 'area': 100, 'category': 'leaf'},
            {'xyxy': (50, 50, 70, 70), 'area': 400, 'category': 'rock'},
        ]
    }
    row = {
        'tile_source_gid': 1,
        'tile_extent_xyxy_in_source': [0, 0, 15, 20],
    }
    assert driver._candidate_distractor_categories(row, boxes, 0.5) == ('leaf',)
    assert driver._candidate_distractor_categories(row, boxes, 0.51) == ()


class _FakeAnnots:
    def __init__(self, objs):
        self.objs = objs


class _FakeDset:
    def __init__(self, anns, cats):
        self._anns = anns
        self.cats = cats

    def annots(self):
        return _FakeAnnots(self._anns)


def test_reviewed_hardneg_ignore_categories_include_cleanup_marks():
    config = driver.load_config(
        Path(__file__).resolve().parents[1]
        / 'experiments' / 'rfdetr_seg_v1' / 'config.v6_reviewed_hardneg.yaml'
    )
    ignore_categories = config['truth_semantics']['ignore_categories']
    assert 'ignore' in ignore_categories
    assert 'unknown' in ignore_categories
    assert 'residual' in ignore_categories
    assert 'residue' in ignore_categories


def test_truth_hygiene_rejects_typo_and_uncategorized_spatial_truth():
    config = {
        'truth_hygiene': {
            'forbidden_category_names': ['unkown'],
            'require_categorized_annotations_splits': ['test'],
        }
    }
    dset = _FakeDset(
        anns=[{'id': 1, 'category_id': 1}],
        cats={1: {'id': 1, 'name': 'unkown'}},
    )
    try:
        driver._check_truth_hygiene(config, 'train', dset)
    except RuntimeError as ex:
        assert 'forbidden category' in str(ex)
    else:
        raise AssertionError('expected forbidden-category failure')

    dset = _FakeDset(
        anns=[{'id': 2, 'category_id': None, 'bbox': [0, 0, 10, 10]}],
        cats={},
    )
    try:
        driver._check_truth_hygiene(config, 'test', dset)
    except RuntimeError as ex:
        assert 'spatial annotations have no resolved category' in str(ex)
    else:
        raise AssertionError('expected uncategorized-spatial-annotation failure')


def test_truth_hygiene_allows_uncategorized_nonspatial_metadata():
    config = {
        'truth_hygiene': {
            'require_categorized_annotations_splits': ['train'],
        }
    }
    dset = _FakeDset(
        anns=[
            {
                'id': 10,
                'image_id': 1,
                'category_id': None,
                'caption': 'grass; downview',
                'iscrowd': False,
                'ignore': False,
            },
            {
                'id': 11,
                'image_id': 2,
                'category_id': None,
                'caption': None,
                'iscrowd': False,
                'ignore': False,
            },
        ],
        cats={},
    )
    report = driver._check_truth_hygiene(config, 'train', dset)
    assert report['uncategorized_spatial_annotations'] == 0
    assert report['uncategorized_nonspatial_metadata_annotations'] == 2
    assert report['uncategorized_nonspatial_metadata_examples'] == [10, 11]
    assert driver._uncategorized_nonspatial_metadata_ids(dset) == [10, 11]


def test_annotation_has_spatial_payload():
    assert not driver._annotation_has_spatial_payload({'category_id': None})
    assert not driver._annotation_has_spatial_payload({'bbox': None, 'segmentation': []})
    assert driver._annotation_has_spatial_payload({'bbox': [0, 0, 1, 1]})
    assert driver._annotation_has_spatial_payload({'segmentation': [[0, 0, 1, 0, 1, 1]]})
    assert driver._annotation_has_spatial_payload({'keypoints': [1, 2, 2]})
