import json
from pathlib import Path

import pytest

from shitspotter import labelme_review


def _demo_truth_and_pred(tmp_path, exif_ori=6, with_sidecar=True):
    import kwcoco
    import kwimage
    import numpy as np

    source = tmp_path / 'source.jpg'
    kwimage.imwrite(source, np.zeros((60, 100, 3), dtype=np.uint8))

    true = kwcoco.CocoDataset()
    gid = true.add_image(
        file_name=str(source), width=100, height=60, exif_ori=exif_ori,
    )
    poop_cid = true.add_category(name='poop')
    true.add_annotation(
        image_id=gid,
        category_id=poop_cid,
        bbox=[2, 2, 10, 10],
        segmentation={'exterior': [[2, 2], [12, 2], [12, 12], [2, 12]]},
    )
    true.fpath = tmp_path / 'truth.kwcoco.json'
    true.dump()

    if with_sidecar:
        # EXIF orientation 6 swaps the display dimensions.
        sidecar = {
            'version': '5.3.1',
            'flags': {},
            'imagePath': source.name,
            'imageHeight': 100 if exif_ori in {5, 6, 7, 8} else 60,
            'imageWidth': 60 if exif_ori in {5, 6, 7, 8} else 100,
            'imageData': None,
            'shapes': [{
                'label': '__metadata__',
                'points': [[1.0, 1.0]],
                'group_id': None,
                'shape_type': 'point',
                'flags': {},
                'description': 'keep me',
            }],
        }
        source.with_suffix('.json').write_text(json.dumps(sidecar))

    pred = kwcoco.CocoDataset()
    pred_gid = pred.add_image(file_name=str(source), width=100, height=60, exif_ori=exif_ori)
    assert pred_gid == gid
    pred_cid = pred.add_category(name='poop')
    pred_aid = pred.add_annotation(
        image_id=gid,
        category_id=pred_cid,
        bbox=[40, 20, 20, 15],
        segmentation={'exterior': [[40, 20], [60, 20], [60, 35], [40, 35]]},
        score=0.95,
    )
    pred.fpath = tmp_path / 'pred.kwcoco.json'
    pred.dump()
    return source, true, pred, gid, pred_aid


def _write_review(tmp_path, true, pred, gid, pred_aid):
    review = tmp_path / 'review'
    review.mkdir()
    queue = {
        'schema': 'kwcoco_detector_kit.prediction_review.v1',
        'true_kwcoco': str(true.fpath),
        'pred_kwcoco': str(pred.fpath),
        'items': [{
            'rank': 1,
            'score': 0.95,
            'classification': 'unexplained_prediction',
            'source_gid': gid,
            'source_image': true.imgs[gid]['file_name'],
            'source_fpath': str(Path(true.get_image_fpath(gid)).resolve()),
            'labelme_json': str(Path(true.get_image_fpath(gid)).with_suffix('.json').resolve()),
            'prediction_ann_id': pred_aid,
            'prediction_bbox_xyxy': [40.0, 20.0, 60.0, 35.0],
            'prediction_has_segmentation': True,
            'has_spatial_truth_overlap': False,
            'overlaps': [],
        }],
    }
    (review / 'review_queue.json').write_text(json.dumps(queue))
    return review


def test_prepare_requires_explicit_proposal_adjudication(tmp_path):
    source, true, pred, gid, pred_aid = _demo_truth_and_pred(tmp_path)
    review = _write_review(tmp_path, true, pred, gid, pred_aid)
    workspace = tmp_path / 'workspace'

    labelme_review.prepare_workspace(review, workspace, top_images=1)
    manifest = json.loads((workspace / 'manifest.json').read_text())
    item = manifest['items'][0]
    staged = Path(item['staged_labelme'])
    data = json.loads(staged.read_text())

    proposal_shapes = [
        s for s in data['shapes']
        if s['label'] == labelme_review.PROPOSAL_LABEL
    ]
    assert proposal_shapes
    assert data['imagePath'] == Path(item['staged_image']).name
    # Existing point-only metadata survives staging.
    assert any(s['shape_type'] == 'point' for s in data['shapes'])

    with pytest.raises(RuntimeError, match='unresolved'):
        labelme_review.build_apply_plan(workspace)

    # Accept the model proposal as a labeled hard negative.
    for shape in proposal_shapes:
        shape['label'] = 'leaf'
    staged.write_text(json.dumps(data))

    plan = labelme_review.build_apply_plan(workspace)
    assert len(plan['operations']) == 1
    # Dry run cannot mutate canonical truth.
    before = source.with_suffix('.json').read_bytes()
    labelme_review.apply_workspace(workspace, commit=False)
    assert source.with_suffix('.json').read_bytes() == before

    labelme_review.apply_workspace(workspace, commit=True)
    canonical = json.loads(source.with_suffix('.json').read_text())
    assert any(s['label'] == 'leaf' for s in canonical['shapes'])
    assert not any(s['label'] == labelme_review.PROPOSAL_LABEL for s in canonical['shapes'])
    accepted = [s for s in canonical['shapes'] if s['label'] == 'leaf']
    assert all(s.get('group_id') is None for s in accepted)
    assert all(
        not s.get('description', '').startswith(
            labelme_review.PROPOSAL_DESCRIPTION_PREFIX
        )
        for s in accepted
    )
    assert canonical['imagePath'] == source.name

    # Copy-back is idempotent. A second application recognizes its own receipt
    # rather than misdiagnosing the first write as an external conflict.
    second = labelme_review.apply_workspace(workspace, commit=True)
    assert second['complete'] is True
    assert len(second['operations']) == 1


def test_proposal_metadata_loss_is_safe(tmp_path):
    source, true, pred, gid, pred_aid = _demo_truth_and_pred(tmp_path)
    review = _write_review(tmp_path, true, pred, gid, pred_aid)
    workspace = tmp_path / 'workspace'
    labelme_review.prepare_workspace(review, workspace, top_images=1)

    manifest = json.loads((workspace / 'manifest.json').read_text())
    staged = Path(manifest['items'][0]['staged_labelme'])
    data = json.loads(staged.read_text())
    proposal = next(
        s for s in data['shapes']
        if s['label'] == labelme_review.PROPOSAL_LABEL
    )

    # Simulate an editor that discards group_id. The reserved label still
    # blocks apply.
    proposal['group_id'] = None
    staged.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='unresolved'):
        labelme_review.build_apply_plan(workspace)

    # If the user explicitly relabels the shape, the description marker is
    # enough to recognize it as an accepted proposal and strip review metadata.
    proposal['label'] = 'trash'
    staged.write_text(json.dumps(data))
    plan = labelme_review.build_apply_plan(workspace)
    accepted = [
        s for s in plan['operations'][0]['data']['shapes']
        if s['label'] == 'trash'
    ]
    assert accepted
    assert accepted[0].get('group_id') is None
    assert not accepted[0].get('description', '').startswith(
        labelme_review.PROPOSAL_DESCRIPTION_PREFIX
    )


def test_delete_proposal_is_rejection_and_noop(tmp_path):
    source, true, pred, gid, pred_aid = _demo_truth_and_pred(tmp_path)
    review = _write_review(tmp_path, true, pred, gid, pred_aid)
    workspace = tmp_path / 'workspace'
    labelme_review.prepare_workspace(review, workspace, top_images=1)

    manifest = json.loads((workspace / 'manifest.json').read_text())
    item = manifest['items'][0]
    staged = Path(item['staged_labelme'])
    data = json.loads(staged.read_text())
    proposal_groups = {p['group_id'] for p in item['proposals']}
    data['shapes'] = [
        s for s in data['shapes'] if s.get('group_id') not in proposal_groups
    ]
    staged.write_text(json.dumps(data))

    plan = labelme_review.build_apply_plan(workspace)
    assert plan['operations'] == []


def test_apply_fails_if_canonical_sidecar_changed(tmp_path):
    source, true, pred, gid, pred_aid = _demo_truth_and_pred(tmp_path)
    review = _write_review(tmp_path, true, pred, gid, pred_aid)
    workspace = tmp_path / 'workspace'
    labelme_review.prepare_workspace(review, workspace, top_images=1)

    manifest = json.loads((workspace / 'manifest.json').read_text())
    item = manifest['items'][0]
    staged = Path(item['staged_labelme'])
    data = json.loads(staged.read_text())
    for shape in data['shapes']:
        if shape['label'] == labelme_review.PROPOSAL_LABEL:
            shape['label'] = 'poop'
    staged.write_text(json.dumps(data))

    canonical = source.with_suffix('.json')
    concurrent = json.loads(canonical.read_text())
    concurrent['flags']['changed_elsewhere'] = True
    canonical.write_text(json.dumps(concurrent))

    with pytest.raises(RuntimeError, match='changed after workspace preparation'):
        labelme_review.build_apply_plan(workspace)


def test_raw_exif_proposal_roundtrip(tmp_path):
    import kwimage

    source, true, pred, gid, pred_aid = _demo_truth_and_pred(tmp_path, exif_ori=6)
    img = true.imgs[gid]
    exif_dsize = (60, 100)
    pred_ann = pred.anns[pred_aid]
    exif_polys = labelme_review._proposal_polygons_in_exif(pred_ann, img, exif_dsize)
    assert exif_polys

    raw_from_exif = labelme_review._raw_from_exif_transform(6, exif_dsize)
    reconstructed = exif_polys[0].warp(raw_from_exif)
    expected = kwimage.Segmentation.coerce(pred_ann['segmentation']).to_multi_polygon().data[0]
    reconstructed_shp = reconstructed.to_shapely()
    expected_shp = expected.to_shapely()
    delta_area = reconstructed_shp.symmetric_difference(expected_shp).area
    assert delta_area < 1e-8



def test_prepare_revalidates_against_current_truth(tmp_path):
    import kwcoco

    source, true, pred, gid, pred_aid = _demo_truth_and_pred(tmp_path)
    review = _write_review(tmp_path, true, pred, gid, pred_aid)

    # Simulate the user fixing canonical LabelMe truth after the review queue
    # was generated. The current KWCoco now contains a localized annotation
    # that intersects the formerly-unexplained prediction.
    current = kwcoco.CocoDataset.coerce(str(true.fpath))
    current.add_annotation(
        image_id=gid,
        category_id=current.index.name_to_cat['poop']['id'],
        bbox=[42, 22, 10, 8],
        segmentation={
            'exterior': [[42, 22], [52, 22], [52, 30], [42, 30]],
        },
    )
    current.fpath = tmp_path / 'current_truth.kwcoco.json'
    current.dump()

    workspace = tmp_path / 'workspace'
    with pytest.raises(RuntimeError, match='no zero-overlap unexplained predictions'):
        labelme_review.prepare_workspace(
            review,
            workspace,
            true=current.fpath,
            top_images=1,
        )
