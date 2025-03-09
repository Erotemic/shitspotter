#!/usr/bin/env python3
"""
"""
import scriptconfig as scfg
import ubelt as ub


class SimplifyKwcocoCLI(scfg.DataConfig):
    """
    """
    src = scfg.Value(None, help='input kwcoco path')
    dst = scfg.Value(None, help='output kwcoco path')
    minimum_instances = scfg.Value(100, help='only keep categories with at least this many instances')

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Ignore:
            >>> from shitspotter.cli.simplify_kwcoco import *  # NOQA
            >>> import shitspotter
            >>> argv = 0
            >>> cls = SimplifyKwcocoCLI
            >>> kwargs = {}

            >>> kwargs['src'] = '/home/joncrall/data/dvc-repos/shitspotter_dvc/train_imgs6917_05c90c75.kwcoco.zip'
            >>> kwargs['dst'] = '/home/joncrall/data/dvc-repos/shitspotter_dvc/simplified_train_imgs6917_05c90c75.kwcoco.zip'
            >>> kwargs['minimum_instances'] = 100
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)

            >>> kwargs['src'] = '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_4fb668db.kwcoco.zip'
            >>> kwargs['dst'] = '/home/joncrall/data/dvc-repos/shitspotter_dvc/simplified_vali_imgs1258_4fb668db.kwcoco.zip'
            >>> kwargs['minimum_instances'] = 100
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)

        Example:
            >>> # xdoctest: +SKIP
            >>> from shitspotter.cli.simplify_kwcoco import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict(src='vidshapes8')
            >>> cls = SimplifyKwcocoCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        if config.dst is None:
            raise ValueError('Must specify output path')

        import kwcoco
        import kwimage
        import numpy as np
        dset = kwcoco.CocoDataset.coerce(config.src)
        dset.reroot(absolute=True)
        dset._update_fpath(config.dst)

        category_hist = {c: 0 for c in dset.categories().lookup('name')} | ub.dict_hist(dset.annots().category_names)
        print(f'category_hist = {ub.urepr(category_hist, nl=1)}')
        remove_catnames = [k for k, f in category_hist.items() if f < config.minimum_instances]
        keep_catnames = set(category_hist) - set(remove_catnames)
        print(f'remove_catnames={remove_catnames}')
        print(f'keep_catnames={keep_catnames}')

        dset.remove_categories(remove_catnames)

        to_remove = []
        new_anns = []
        for coco_image in ub.ProgIter(dset.images().coco_images_iter(), total=dset.n_images, desc='simplify'):
            annots = coco_image.annots()
            # boxes: kwimage.Boxes = annots.boxes
            dets: kwimage.Detections = annots.detections
            if len(dets) == 0:
                continue

            dets.data['catnames'] = np.array(annots.category_names)
            dets.data['annot_ids'] = np.array(annots)
            import kwarray
            keep_flags = kwarray.isect_flags(dets.data['catnames'], keep_catnames)
            if not np.all(keep_flags):
                to_remove.extend(dets.compress(~keep_flags).data['annot_ids'])
                dets = dets.compress(keep_flags)

            if len(dets) == 0:
                continue

            from geowatch.utils.util_kwimage import find_low_overlap_covering_boxes
            from shapely.ops import unary_union
            min_dim = dets.boxes.to_xywh().data[..., 2:4].min()
            max_dim = dets.boxes.to_xywh().data[..., 2:4].max()
            scale = 1.5
            result = find_low_overlap_covering_boxes(dets.boxes.to_polygons(), scale, min_box_dim=min_dim, max_box_dim=max_dim * 1.5, verbose=0)

            new_boxes, assignment = result
            if len(dets) != len(new_boxes):
                for idxs in assignment:
                    if len(idxs) > 1:
                        if ub.allsame(dets.data['catnames'][idxs]):
                            old_aids = list(ub.take(annots, idxs))
                            new_poly = unary_union([p.to_shapely() for p in ub.take(dets.data['segmentations'], idxs)])
                            new_ann = ub.udict(annots.objs[idxs[0]]) - {'id', 'segmentation', 'bbox'}
                            new_poly = kwimage.MultiPolygon.coerce(new_poly)
                            new_ann['bbox'] = new_poly.box().to_coco()
                            new_ann['segmenation'] = new_poly.to_coco(style='new')
                            to_remove.extend(old_aids)
                            new_anns.append(new_ann)

        dset.remove_annotations(to_remove)
        for ann in new_anns:
            dset.add_annotation(**ann)

        empty_gids = [gid for gid, aids in dset.index.gid_to_aids.items() if len(aids) == 0]
        dset.remove_images(empty_gids)
        dset.dump()


def simulate_holes_with_seams(polygon):
    """
    Add a seam to convert holes into part of the exterior boundary.

    Args:
        polygon (shapely.geometry.Polygon | shapely.geometry.MultiPolygon)

    Example:
        >>> from shitspotter.cli.simplify_kwcoco import *  # NOQA
        >>> import kwimage
        >>> poly = kwimage.Polygon.random(n_holes=1, rng=3)
        >>> polygon = poly.to_shapely()
        >>> result = simulate_holes_with_seams(polygon)
        >>> new_poly = kwimage.MultiPolygon.from_shapely(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=True)
        >>> poly.draw(color='blue', edgecolor='black', setlim=1)
        >>> kwplot.figure(fnum=1, pnum=(1, 2, 2))
        >>> new_poly.draw(color='green', edgecolor='black', alpha=0.5)
    """
    from shapely.geometry import MultiPolygon, Polygon  # NOQA
    from shapely.geometry import Point
    from shapely.geometry import LineString  # NOQA
    from shapely.geometry import MultiLineString  # NOQA
    from shapely.ops import split  # NOQA
    from shapely import difference

    input_polygons = []
    if polygon.geom_type == 'MultiPolygon':
        input_polygons = polygon.geoms
    else:
        input_polygons = [polygon]

    new_polygons = []

    for polygon in input_polygons:
        if len(polygon.interiors):

            seams = []
            # Iterate through the holes
            for hole in polygon.interiors:

                # TODO: find the "best" point
                # Find a point on the hole and a point on the exterior
                hole_pt = Point(hole.coords[0])
                dist = polygon.exterior.project(hole_pt)
                exterior_pt = polygon.exterior.interpolate(dist)
                # Create a line connecting the hole to the exterior
                seam = LineString([hole_pt, exterior_pt])
                # Hack, can we do better here?
                seam = seam.buffer(0.001)
                seams.append(seam)

            # Split the polygon along the seams
            split_result = polygon
            for seam in seams:
                split_result = difference(split_result, seam)
                # split_result = split(split_result, seam)

            if polygon.geom_type == 'Polygon':
                new_polygons.append(split_result)
            else:
                new_polygons.extend(split_result.geoms)

        else:
            # No holes, no need to do anything
            new_polygons.extend(polygon)

    result = MultiPolygon(new_polygons)
    return result


def approximate_minimum_distance_seam(ring1, ring2):
    """
    Finds an approximately minimum distance seam between two LinearRings.

    The result is guaranteed to be no more than twice the minimum distance.

    Args:
        ring1 (shapely.geometry.LinearRing): The first linear ring of the exterior boundary.
        ring2 (shapely.geometry.LinearRing): The second linear ring of the interior boundary.

    Returns:
        shapely.geometry.LineString: The approximately minimum distance line.

    Ignore:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/shitspotter'))
        >>> from shitspotter.cli.simplify_kwcoco import *  # NOQA
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.autompl()
        >>> poly1 = kwimage.Polygon.random()
        >>> poly2 = poly1.scale(1.5, about='centroid').warp(kwimage.Affine.random(offset=0))
        >>> ring1 = poly1.to_shapely().exterior
        >>> ring2 = poly2.to_shapely().exterior
        >>> seam = approximate_minimum_distance_seam(ring1, ring2)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import kwplot.mpl_draw
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=True)
        >>> poly1.draw(color='blue', edgecolor='blue', facecolor='none', setlim='grow')
        >>> poly2.draw(color='blue', edgecolor='green', facecolor='none', setlim='grow')
        >>> kwplot.mpl_draw.draw_polyline(np.array(seam.coords.xy).T, edgecolor='black')

    """
    import numpy as np
    from shapely.geometry import LineString  # NOQA
    from shapely.geometry import Point  # NOQA
    # Convert coordinates to NumPy arrays for faster computation
    coords1 = np.array(ring1.coords)
    coords2 = np.array(ring2.coords)

    min_distance = float('inf')
    best = None

    # Iterate over all points in ring1 and find the closest point in ring2
    for idx1, p1 in enumerate(coords1):
        distances = np.linalg.norm(coords2 - p1, axis=1)
        idx2 = np.argmin(distances)
        min_dist = distances[idx2]

        if min_dist < min_distance:
            min_distance = min_dist
            best = (p1, coords2[idx2], idx1, idx2)

    pt1, pt2, idx1, idx2 = best

    search_idxs1 = np.array([idx1 - 1, idx1 + 1]) % len(coords1)
    search_idxs2 = np.array([idx2 - 1, idx2 + 1]) % len(coords1)

    ring1_search_pt1 = Point(coords1[search_idxs1[0]])
    ring1_search_pt2 = Point(coords1[search_idxs1[1]])

    ring2_search_pt1 = Point(coords2[search_idxs2[0]])
    ring2_search_pt2 = Point(coords2[search_idxs2[1]])

    ring1_d1 = ring1.project(ring1_search_pt1)
    ring1_d2 = ring1.project(ring1_search_pt2)

    ring2_d1 = ring1.project(ring2_search_pt1)
    ring2_d2 = ring1.project(ring2_search_pt2)

    search_distances1 = np.linspace(ring1_d1, ring1_d2, 7)
    search_distances2 = np.linspace(ring2_d1, ring2_d2, 5)

    # Given two vertices that are close, refine futher.
    candidate_points1 = np.array([p.xy for p in ring1.interpolate(search_distances1)])[..., 0]
    candidate_points2 = np.array([p.xy for p in ring2.interpolate(search_distances2)])[..., 0]
    candidate_points1.shape
    distances = np.linalg.norm(candidate_points1[:, None, :] - candidate_points2[None, :, :], axis=2)

    refined_idxs = np.unravel_index(distances.argmin(), distances.shape)
    refined_idx1 = refined_idxs[0]
    refined_idx2 = refined_idxs[1]

    pt1 = Point(candidate_points1[refined_idx1])
    pt2 = Point(candidate_points2[refined_idx2])

    seam = LineString([pt1, pt2])
    return seam

    # # Return the LineString connecting the best pair of points
    # pt1 = Point(best_pair[0])
    # pt2 = Point(best_pair[1])

    # return LineString([pt1, pt2])



__cli__ = SimplifyKwcocoCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/shitspotter/cli/simplify_kwcoco.py
        python -m shitspotter.cli.simplify_kwcoco
    """
    __cli__.main()
