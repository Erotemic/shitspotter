#!/usr/bin/env python


def make_splits():
    import shitspotter
    import kwcoco
    coco_fpath = shitspotter.util.find_shit_coco_fpath()
    dset = kwcoco.CocoDataset(coco_fpath)

    gids_with_annots = [gid for gid, aids in dset.index.gid_to_aids.items() if len(aids) > 0]
    images_with_annots = dset.images(gids_with_annots)

    import ubelt as ub
    from kwutil import util_time
    datetimes = list(map(util_time.coerce_datetime, images_with_annots.lookup('datetime', None)))
    year_to_gids = ub.group_items(images_with_annots, [d.year for d in datetimes])

    # Group images into videos (do this with the image pairs)
    for year, gids in year_to_gids.items():

        video_name = f'video_{year}'
        if video_name not in dset.index.name_to_video:
            video_id = dset.add_video(name=video_name)
        else:
            video_id = dset.index.name_to_video[video_name]['id']

        video = dset.index.videos[video_id]

        video_images = dset.images(gids)

        for idx, img in enumerate(video_images.objs):
            img['frame_index'] = idx
            img['video_id'] = video_id
            img['sensor_coarse'] = 'phone'
            img['datetime_captured'] = img['datetime']
            img['channels'] = 'red|green|blue'

            # hack
            video['width'] = img['width']
            video['height'] = img['height']

    dset._build_index()
    dset.conform()

    vali_gids = []
    train_gids = []

    for year, gids in year_to_gids.items():
        if year <= 2020:
            vali_gids.extend(gids)
        else:
            train_gids.extend(gids)

    groups = [g for k, g in sorted(year_to_gids.items())]
    train_gids = list(ub.flatten(groups[1:]))
    vali_gids = list(ub.flatten(groups[:1]))

    train_split = dset.subset(train_gids)
    vali_split = dset.subset(vali_gids)

    def build_code(coco_dset):
        hashid = coco_dset._build_hashid()[0:8]
        return f'imgs{coco_dset.n_images}_{hashid}'

    # coco_dset = vali_split
    fname = ('vali_' + build_code(vali_split) + '.kwcoco.zip')
    bundle_dpath = ub.Path(dset.fpath).parent
    vali_split.fpath = bundle_dpath / fname

    fname = ('train_' + build_code(train_split) + '.kwcoco.zip')
    fname = ('train_' + build_code(train_split) + '.kwcoco.zip')
    train_split.fpath = bundle_dpath / fname
    print(f'vali_split.fpath={vali_split.fpath}')
    print(f'train_split.fpath={train_split.fpath}')

    train_split.conform()
    vali_split.conform()

    vali_split.dump()
    train_split.dump()

    ub.symlink(train_split.fpath, link_path=train_split.fpath.parent / 'train.kwcoco.zip', overwrite=True, verbose=3)
    ub.symlink(vali_split.fpath, link_path=vali_split.fpath.parent / 'vali.kwcoco.zip', overwrite=True, verbose=3)

    # See ~/code/ndsampler/train.sh


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/make_splits.py
    """
    make_splits()
