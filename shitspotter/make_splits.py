#!/usr/bin/env python


def make_splits():
    import shitspotter
    import kwcoco
    import ubelt as ub
    import numpy as np
    import kwutil
    coco_fpath = shitspotter.util.find_shit_coco_fpath()
    dset = kwcoco.CocoDataset(coco_fpath)

    # Check on the automatic protocol
    change_point = kwutil.util_time.datetime.coerce('2021-05-11T120000')

    # Data from these years will belong to the validation dataset
    validation_years = {2020, 2024}

    # Group images by cohort, and determine train / val split
    cohort_to_imgs = ub.group_items(dset.images().coco_images, key=lambda g: g['cohort'])
    cohort_to_imgs = {cohort: sorted(imgs, key=lambda g: g['datetime']) for cohort, imgs in cohort_to_imgs.items()}
    for cohort, coco_imgs in cohort_to_imgs.items():

        cohort_start = coco_imgs[0].datetime

        if cohort.startswith('poop-'):
            has_annots = np.array([len(coco_img.annots()) > 0 for coco_img in coco_imgs]).astype(np.uint8)
            keep_flags = has_annots.copy()
            if cohort_start <= change_point:
                keep_flags + np.roll(has_annots, 1) + np.roll(has_annots, 2)
            else:
                keep_flags + np.roll(has_annots, 1)
            keep_imgs = ub.compress(coco_imgs, keep_flags)

            cohort_year = cohort_start.date().year
            is_validation = cohort_year in validation_years

            for coco_img in keep_imgs:
                if is_validation:
                    coco_img.img['split'] = 'vali'
                else:
                    coco_img.img['split'] = 'train'
        else:
            raise NotImplementedError

    for img in dset.dataset['images']:
        img['sensor_coarse'] = 'phone'
        img['datetime_captured'] = img['datetime']
        img['channels'] = 'red|green|blue'

    # gids_with_annots = [gid for gid, aids in dset.index.gid_to_aids.items() if len(aids) > 0]
    # images_with_annots = dset.images(gids_with_annots)
    # import ubelt as ub
    # from kwutil import util_time
    # datetimes = list(map(util_time.coerce_datetime, images_with_annots.lookup('datetime', None)))
    # year_to_gids = ub.group_items(images_with_annots, [d.year for d in datetimes])
    # # Group images into videos (do this with the image pairs)
    # if 0:
    #     for year, gids in year_to_gids.items():
    #         video_name = f'video_{year}'
    #         video_id = dset.ensure_video(name=video_name)
    #         video = dset.index.videos[video_id]
    #         video_images = dset.images(gids)
    #         for idx, img in enumerate(video_images.objs):
    #             img['frame_index'] = idx
    #             img['video_id'] = video_id
    #             img['sensor_coarse'] = 'phone'
    #             img['datetime_captured'] = img['datetime']
    #             img['channels'] = 'red|green|blue'
    #             # hack
    #             video['width'] = img['width']
    #             video['height'] = img['height']

    dset._build_index()
    dset.conform()

    images = dset.images()
    split_to_gids = ub.group_items(images, images.lookup('split', default=None))
    train_split = dset.subset(split_to_gids['train'])
    vali_split = dset.subset(split_to_gids['vali'])

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

    vali_stats = vali_split.basic_stats()
    train_stats = train_split.basic_stats()
    print(f'vali_stats = {ub.urepr(vali_stats, nl=1)}')
    print(f'train_stats = {ub.urepr(train_stats, nl=1)}')

    ub.symlink(train_split.fpath, link_path=train_split.fpath.parent / 'train.kwcoco.zip', overwrite=True, verbose=3)
    ub.symlink(vali_split.fpath, link_path=vali_split.fpath.parent / 'vali.kwcoco.zip', overwrite=True, verbose=3)

    # See ~/code/ndsampler/train.sh


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/make_splits.py
    """
    make_splits()
