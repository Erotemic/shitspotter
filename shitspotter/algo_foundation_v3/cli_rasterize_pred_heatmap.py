"""
Rasterize polygon-segmentation predictions into per-pixel salient heatmaps.

kwcoco.metrics.segmentation_metrics requires a ``salient`` channel on each
predicted image.  Our SAM2 predictions store segmentation as polygon
annotations with a ``score`` attribute.  This script converts those into
per-pixel float32 heatmaps where each pixel's value is the maximum ``score``
of any polygon that covers it.

The output is a copy of the input kwcoco that adds an auxiliary ``salient``
asset (uint8 PNG, values 0-255 representing score 0.0-1.0) for every image
that has at least one prediction.  Images with no predictions get a zero
heatmap so that they contribute true-negatives to the metric.

The original prediction kwcoco is not modified.

CommandLine:
    python -m shitspotter.algo_foundation_v3.cli_rasterize_pred_heatmap --help
"""

import scriptconfig as scfg
import ubelt as ub


class RasterizePredHeatmapCLI(scfg.DataConfig):
    """
    Rasterize polygon predictions into a per-pixel salient heatmap channel.
    """
    src = scfg.Value(None, position=1, help='input kwcoco path with polygon predictions')
    dst = scfg.Value(None, help='output kwcoco path (will include salient aux channel)')
    salient_channel = scfg.Value('salient', help='channel name to write the heatmap under')
    score_field = scfg.Value('score', help='annotation field to use as the pixel value')
    workers = scfg.Value(0, help='number of parallel worker processes')

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco
        import kwimage
        import numpy as np
        from pathlib import Path

        config = cls.cli(argv=argv, data=kwargs, strict=True)

        src_dset = kwcoco.CocoDataset.coerce(config.src)
        src_dset.reroot(absolute=True)

        dst_fpath = Path(config.dst)
        dst_fpath.parent.mkdir(parents=True, exist_ok=True)

        # Asset directory: sibling folder named after the dst stem
        asset_dpath = dst_fpath.parent / (dst_fpath.stem.split('.')[0] + '_salient_assets')
        asset_dpath.mkdir(parents=True, exist_ok=True)

        dst_dset = src_dset.copy()
        dst_dset.fpath = str(dst_fpath)
        dst_dset.reroot(absolute=True)

        salient_channel = config.salient_channel
        score_field = config.score_field

        for coco_img in ub.ProgIter(
            dst_dset.images().coco_images,
            total=dst_dset.n_images,
            desc='rasterize salient heatmaps',
        ):
            img = coco_img.img
            w = img.get('width')
            h = img.get('height')
            if w is None or h is None:
                # Fall back to loading the image to get dimensions
                raw = coco_img.imdelay().finalize()
                h, w = raw.shape[:2]
                img['width'] = w
                img['height'] = h

            heatmap = np.zeros((h, w), dtype=np.float32)

            aids = dst_dset.index.gid_to_aids.get(img['id'], [])
            for ann in dst_dset.annots(list(aids)).objs:
                score = ann.get(score_field, 1.0)
                if score is None:
                    score = 1.0
                score = float(score)
                seg_data = ann.get('segmentation', None)
                if seg_data is None:
                    # Fall back to bbox if no segmentation
                    bbox = ann.get('bbox', None)
                    if bbox is None:
                        continue
                    x, y, bw, bh = bbox
                    sseg = kwimage.Boxes([[x, y, bw, bh]], 'xywh').to_polygons()[0]
                else:
                    sseg = kwimage.Segmentation.coerce(seg_data).to_multi_polygon()

                # Paint the max score: use np.maximum after filling
                layer = np.zeros((h, w), dtype=np.float32)
                layer = sseg.fill(layer, value=score)
                np.maximum(heatmap, layer, out=heatmap)

            # Encode as uint8 PNG (0-255 maps to score 0.0-1.0)
            heatmap_u8 = (heatmap * 255).clip(0, 255).astype(np.uint8)

            img_name = Path(img['file_name']).stem
            asset_fname = f'{img_name}_{img["id"]:06d}_salient.png'
            asset_fpath = asset_dpath / asset_fname

            kwimage.imwrite(str(asset_fpath), heatmap_u8)

            # Register the auxiliary asset on the image
            auxiliary = img.setdefault('auxiliary', [])
            # Remove any existing entry for the same channel
            auxiliary[:] = [a for a in auxiliary if a.get('channels') != salient_channel]
            auxiliary.append({
                'file_name': str(asset_fpath),
                'channels': salient_channel,
                'width': w,
                'height': h,
            })

        dst_dset.dump()
        print(f'Wrote heatmap kwcoco to {dst_dset.fpath}')


__cli__ = RasterizePredHeatmapCLI

if __name__ == '__main__':
    __cli__.main()
