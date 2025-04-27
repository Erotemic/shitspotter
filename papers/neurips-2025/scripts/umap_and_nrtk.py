# /// script
# dependencies = [
#   "geowatch",
#   "kwimage",
#   "ubelt",
#   "einops",
#   "nrtk_explorer",
#   "numpy",
#   "torch",
#   "kwutil",
#   "umap",
#   "scikit-learn",
#   "fast_tsp",
# ]
# requires-python = ">=3.11"
# ///
from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
import kwimage
import ubelt as ub
import kwutil
import einops
import torch
import shitspotter
import numpy as np
# import json
import base64
from nrtk_explorer.library.embeddings_extractor import EmbeddingsExtractor


class WrappedDataset(KWCocoVideoDataset):
    def getitem(self, index):
        OPTIMIZE = 1
        if OPTIMIZE:
            target = self._coerce_target(index)
            image_id = target['gids'][0]
            coco_img = self.sampler.dset.coco_image(image_id)
            fpath = coco_img.primary_image_filepath()
            # imdata_hwc = kwimage.imread(fpath, overview="coarsest", backend='gdal')
            imdata_hwc = kwimage.imread(fpath, overview=2, backend='gdal')
        else:
            item = super(KWCocoVideoDataset, self).getitem(index)
            # We should get the KWCocoVideoDataset to support this case natively
            imdata_chw = item.imdata_chw
            # old_shape = imdata_chw.shape
            imdata_hwc = einops.rearrange(imdata_chw, 'c h w -> h w c').numpy()
            target = item['target']
            image_id = target['gids'][0]

        # dsize = (224, 224)
        dsize = (512, 512)
        imdata_hwc = torch.from_numpy(kwimage.imresize(imdata_hwc, dsize=dsize))
        imdata_chw = einops.rearrange(imdata_hwc, 'h w c -> c h w')
        imdata_chw = imdata_chw.to(torch.float32)

        mean = torch.Tensor([0.485, 0.456, 0.406])[:, None, None]
        std  = torch.Tensor([0.229, 0.224, 0.225])[:, None, None]
        imdata_chw = ((imdata_chw / 255) - mean) / std

        imdata_chw.nan_to_num_()
        return imdata_chw, image_id


def our_collate(batch_tups):
    tostack = []
    imageids = []
    for imdata, image_id in batch_tups:
        tostack.append(imdata)
        imageids.append(image_id)
    batch = torch.stack(tostack, dim=0)
    imageids = np.array(imageids, dtype=int)
    return batch, imageids


def main():
    """
    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/shitspotter/papers/neurips-2025/scripts'))
        from umap_and_nrtk import *  # NOQA
    """
    dset = shitspotter.open_shit_coco()

    # See if removing extra information can help us iterate faster
    minified_dataset = dset.copy()
    minified_dataset.clear_annotations()
    passlist = {
        'file_name',
        # 'name',
        'width',
        'height',
        # 'channels',
        # 'sensor_coarse'
    }
    for img in minified_dataset.dataset['images']:
        to_remove = img.keys() - passlist
        for k in to_remove:
            img.pop(k)

    minify_factor = len(dset.dumps()) /  len(minified_dataset.dumps())
    print(f'minify_factor={minify_factor}')

    extractor = EmbeddingsExtractor()
    print(f'extractor.device={extractor.device}')
    # dset = kwcoco.CocoDataset.demo('vidshapes8')
    dataset = WrappedDataset(
        minified_dataset,
        mode='test',
        window_dims='full',
        time_dims=1,
        # TODO: allow input resolution to specify a fixed pixel size.
        input_resolution=0.09375,
        # dynamic_fixed_resolution={'fixed_dims': (224, 224)}, # This doesnt work yet
        # output_resolution=0.25,
        balance_options='sequential_without_replacement',
        # output_type='heterogeneous'
        # output_type='homogeneous'
        output_type='rgb',
        reduce_item_size=True,
    )
    dataset.disable_augmenter = True

    if 0:
        # Demo
        imdata, target = dataset[0]
        # item = super(KWCocoVideoDataset, dataset).getitem(0)
        # tensor = item.imdata_chw
        # print(f'item = {ub.urepr(item, nl=1)}')
        # print(f'tensor.shape={tensor.shape}')
        print(f'imdata.shape={imdata.shape}')

    loader = dataset.make_loader(batch_size=64, collate_fn=our_collate, num_workers=8)
    pman = kwutil.ProgressManager()
    with pman:
        # features = extractor.extract_from_loader(loader, pman=pman)
        # def extract_from_loader(self, loader, pman=None):
        _iter = pman.progiter(loader, desc='extract descriptors')
        device = extractor.device

        with torch.set_grad_enabled(False):
            results = []
            for pairs in _iter:
                batch, imageids = pairs
                # Copy image to device if using device
                batch = batch.to(device)
                # batch.nan_to_num_()
                result = extractor.model(batch)
                np_result = result.cpu().numpy()
                # be robust if an image returns more than one descriptor
                # should not happen though.
                results.append({
                    'imageids': imageids,
                    'np_result': np_result,
                })

        for result in pman.progiter(results, desc='Add results to coco'):
            imageids = result['imageids']
            np_result = result['np_result']
            # todo: Buffer writes to cache intermediate process results.
            for gid, desc in zip(imageids, np_result):
                dset.imgs[gid].setdefault('cache', {})
                dset.imgs[gid]['cache']['descriptor'] = pack_numpy_array(desc)

    print('dumping')
    dset.fpath = dset.fpath.augment(stemsuffix='.with-descriptors')
    dset.dump()


def pack_numpy_array(arr, use_base64=True):
    """
    Packs a numpy array into a JSON-serializable structure.

    Written by ChatGPT

    Args:
        arr (numpy.ndarray): The NumPy array to serialize.
        use_base64 (bool): Whether to use base64 encoding for binary data (default: True).

    Returns:
        dict: A dictionary containing the array's shape, dtype, and data.

    Example:
        >>> arr = np.random.rand(3, 3)
        >>> packed = pack_numpy_array(arr, use_base64=True)
        >>> unpacked = unpack_numpy_array(packed, use_base64=True)

    Benchmark:
        >>> import timerit
        >>> import json
        >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
        >>> for timer in ti.reset('unpacked'):
        >>>     with timer:
        >>>         json.dumps(arr.tolist())
        >>> for timer in ti.reset('packed'):
        >>>     with timer:
        >>>         json.dumps(packed)
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    # Get array metadata
    array_info = {
        'type': 'ndarray',
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'data': None
    }

    # Pack data
    if use_base64:
        # More efficient for large arrays, encodes binary data as a base64 string
        array_info['data'] = base64.b64encode(arr.tobytes()).decode('utf-8')
    else:
        # Convert to list (works but less efficient for large arrays)
        array_info['data'] = arr.tolist()

    return array_info


def unpack_numpy_array(packed_arr, use_base64=True):
    """
    Unpacks a JSON-serialized structure into a NumPy array.

    Written by ChatGPT

    Args:
        packed_arr (dict): The packed array structure to deserialize.
        use_base64 (bool): Whether the data was base64 encoded (default: True).

    Returns:
        numpy.ndarray: The deserialized NumPy array.
    """
    shape = packed_arr['shape']
    dtype = np.dtype(packed_arr['dtype'])

    if use_base64:
        # Decode base64 back into binary data
        data = base64.b64decode(packed_arr['data'].encode('utf-8'))
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    else:
        # Convert list back to array
        return np.array(packed_arr['data'], dtype=dtype)


def play_with_descriptors():
    """
    pip install fast-tsp
    """
    import kwcoco
    import umap
    import kwarray
    import sklearn
    import sklearn.cluster
    from scipy.spatial.distance import pdist, squareform
    import fast_tsp
    import kwplot
    fpath = '/home/joncrall/data/dvc-repos/shitspotter_dvc/data.kwcoco.with-descriptors.json'
    dset = kwcoco.CocoDataset(fpath)

    image_ids = []
    descriptors = []

    for img in dset.imgs.values():
        desc = unpack_numpy_array(img['cache']['descriptor'])
        image_ids.append(img['id'])
        descriptors.append(desc)

    image_ids = np.array(image_ids)
    descriptors = np.array(descriptors)

    rng = kwarray.ensure_rng(0)
    kwargs = {}
    kwargs['random_state'] = rng
    kwargs['n_neighbors'] = 15
    umap = umap.UMAP(n_components=2, **kwargs)
    reduced = umap.fit_transform(descriptors)

    num_neighbs = 3
    n_clusters = 13
    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbs * 4, algorithm='brute')
    neighbors.fit(reduced)

    # query_idx = 550
    # dist, idxs = neighbors.kneighbors(reduced[query_idx:query_idx + 1])

    labels = sklearn.cluster.KMeans(n_clusters=n_clusters).fit_predict(reduced)
    unique_labels, groupxs = kwarray.group_indices(labels)
    colors = kwimage.Color.distinct(len(unique_labels))

    kwplot.autompl()

    rng2 = kwarray.ensure_rng(1)

    tasks = []
    for cluster_idx, (color, gxs) in enumerate(zip(colors, groupxs), start=1):
        # ax.plot(reduced[gxs, 0], reduced[gxs, 1], 'o', color=color)
        # ax.plot(reduced[gxs, 0], reduced[gxs, 1], 'o', color=dot_color)
        n_annots = 0
        while n_annots == 0:
            chosen_subidx = rng2.randint(len(gxs))
            query_idx = gxs[chosen_subidx]
            query_gid = image_ids[query_idx]
            n_annots = len(dset.coco_image(query_gid).annots())
        print(f'n_annots={n_annots}')
        dist, idxs = neighbors.kneighbors(reduced[query_idx:query_idx + 1])
        candidates = []
        for idx, dist in zip(idxs.ravel(), dist.ravel()):
            gid = image_ids[idx]
            n_annots = len(dset.coco_image(gid).annots())
            candidates.append({
                'gid': gid,
                'dist': dist,
                'n_annots': n_annots,
                'idx': idx,
            })
        # Choose neighbors with annotations in them
        candidates = sorted(candidates, key=lambda x: (x['n_annots'] == 0, dist))
        idxs = np.array([c['idx'] for c in candidates[0:num_neighbs]])

        # print(f'dist={dist}')
        # idxs = idxs[0:num_neighbs]
        x, y = reduced[idxs].mean(axis=0).ravel()
        gids = image_ids[idxs].ravel()
        tasks.append({
            'xy': (x, y),
            'idxs': idxs,
            'gids': gids,
            'color': color,
        })

    # Order points for readability
    points = np.array([t['xy'] for t in tasks])
    # Make it harder to move in the y direction
    points[:, 1] *= 2
    dist_matrix = squareform(pdist(points))

    # Choose the leftmost point at the start
    start_idx = points[:, 0].argmin()

    int_dist_matrix = (dist_matrix * 100).astype(int)

    # Force the distance to/from the "start point" to always be huge, which
    # means the tour wont optimize to return to the start and we get a tsp path
    # intead.
    int_dist_matrix[start_idx, :] = np.iinfo(np.int32).max
    int_dist_matrix[:, start_idx] = np.iinfo(np.int32).max
    int_dist_matrix[start_idx, start_idx] = 0

    # tour_indexes = fast_tsp.greedy_nearest_neighbor(int_dist_matrix)
    # tour_indexes = fast_tsp.solve_tsp_exact(int_dist_matrix)
    tour_indexes = fast_tsp.find_tour(int_dist_matrix)
    tour_indexes = np.array(tour_indexes)

    start_pos = np.where(tour_indexes == start_idx)[0][0]
    tour_indexes = np.roll(tour_indexes, -start_pos)  # Rotate the path so that it starts with start_idx

    for tour_pos, tour_index in enumerate(tour_indexes):
        task = tasks[tour_index]
        task['tour_pos'] = tour_pos

    # First render the umap embedding
    umap_fig = kwplot.figure(fnum=2)
    umap_fig.clf()
    ax = umap_fig.gca()
    ax.cla()
    dot_color = kwimage.Color.coerce('kitware_gray').adjust(lighten=-0.3).as01()
    ax.plot(reduced[:, 0], reduced[:, 1], 'o', color=dot_color)
    tasks = sorted(tasks, key=lambda x: x['tour_pos'])
    for task in tasks:
        tour_pos = task['tour_pos']
        idxs = task['idxs']
        x, y = task['xy']
        # ax.plot(reduced[idxs, 0], reduced[idxs, 1], 'o', color='orange')
        ax.text(x, y, str(tour_pos + 1), size=20,
                ha="center", va="center",
                bbox=dict(boxstyle="circle", color="orange"))
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    vert_parts = []
    for task in tasks:
        tour_pos = task['tour_pos']
        idxs = task['idxs']
        x, y = task['xy']
        gids = task['gids']
        coco_imgs = dset.images(gids).coco_images
        to_stack = []
        for coco_img in coco_imgs:
            scale = 0.125
            dets = coco_img.annots().detections.scale(scale)
            imdata = coco_img.imdelay().scale(scale).finalize()
            imdata = dets.boxes.draw_on(imdata, color='kitware_blue', thickness=6)
            # imdata = dset.draw_image(coco_img['id'])
            kwimage.imresize(imdata, max_dim=256)
            to_stack.append(imdata)
        stack = kwimage.stack_images(to_stack, axis=0, resize='larger', pad=20, bg_value='white')
        stack = kwimage.imresize(stack, dsize=(None, 1024))
        stack = kwimage.draw_header_text(text=str(tour_pos + 1), image=stack, bg_color='white', color='black')
        vert_parts.append(stack)

    image_canvas = kwimage.stack_images(vert_parts, axis=1, pad=20, bg_value='white')
    height, width = image_canvas.shape[0:2]

    ar = height / width
    w, h = umap_fig.get_size_inches()
    ar = 0.13
    w = 35
    umap_fig.set_size_inches((w, w * ar))
    umap_canvas = kwplot.render_figure_to_image(umap_fig)
    print(f'umap_canvas.shape={umap_canvas.shape}')
    print(f'image_canvas.shape={image_canvas.shape}')

    canvas = kwimage.stack_images([umap_canvas, image_canvas], axis=0, pad=20, bg_value='white', resize='larger')

    fpath = ub.Path('~/code/shitspotter/papers/neurips-2025/figures/umap-v3.jpg').expand()

    kwimage.imwrite(fpath, canvas)
    ub.cmd(f'eog {fpath}')

    kwplot.imshow(canvas, fnum=3)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/papers/neurips-2025/scripts/umap_and_nrtk.py
    """
    main()
