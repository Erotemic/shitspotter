#!/usr/bin/env python
"""
This script defines how train / validation splits are created.
"""


def make_splits():
    import shitspotter
    import kwcoco
    import ubelt as ub
    import numpy as np
    import kwutil
    coco_fpath = shitspotter.util.find_shit_coco_fpath()
    dset = kwcoco.CocoDataset(coco_fpath)

    # This date represents the point that the protocol was changed from the 2
    # image before / after method to the 3 image before / after / negative.
    change_point = kwutil.util_time.datetime.coerce('2021-05-11T120000')

    # Data from these years will belong to the validation dataset
    # mapping specifies number of groups for each validation group
    partial_validation_years = {
        2020: 0,
        2024: 2,  # Cannot change this without putting images from train into vali
    }

    # Group images by cohort, and determine train / val split
    cohort_to_imgs = ub.group_items(dset.images().coco_images, key=lambda g: g['cohort'])
    cohort_to_imgs = {cohort: sorted(imgs, key=lambda g: g['datetime']) for cohort, imgs in cohort_to_imgs.items()}
    for cohort, coco_imgs in cohort_to_imgs.items():

        # Ensure the cohort is sorted chronologically
        coco_imgs = sorted(coco_imgs, key=lambda g: g['datetime'])
        cohort_start = coco_imgs[0].datetime

        if cohort.startswith('poop-'):
            has_annots = np.array([len(coco_img.annots()) > 0 for coco_img in coco_imgs]).astype(np.uint8)

            # We want to be careful to exclude any images that could have an
            # unannotated poop in them. To do this we will use the knowledge of
            # the data gathering protocol.
            keep_flags = has_annots.copy()
            protocol_version = '2img' if cohort_start <= change_point else '3img'
            if protocol_version == '2img':
                # This cohort belongs to the 2 image protocol, if an image is
                # annotated, we can infer that the image after it is likely a
                # negative and include it in the split.
                is_after_image = np.roll(has_annots, 1)
                old_keep_flags = has_annots + is_after_image

                groups, ungrouped = protocol_2img_organize(coco_imgs)
                keep_idxs = list(ub.flatten(groups))
                keep_flags = np.array(ub.boolmask(keep_idxs, len(coco_imgs))).astype(int)
                assert (old_keep_flags > 0).sum() == keep_flags.sum()

            elif protocol_version == '3img':
                # This cohort belongs to the 3 image protocol, if an image is
                # annotated, we can infer that two images after are likely a
                # negative and include it in the split.
                is_after_image = np.roll(has_annots, 1)
                is_negative_image = np.roll(has_annots, 2)
                old_keep_flags = has_annots + is_after_image + is_negative_image

                groups, ungrouped = protocol_3img_organize(coco_imgs)
                keep_idxs = list(ub.flatten(groups))
                keep_flags = np.array(ub.boolmask(keep_idxs, len(coco_imgs))).astype(int)

                assert (old_keep_flags > 0).sum() == keep_flags.sum()
            else:
                raise KeyError(protocol_version)
            keep_imgs = list(ub.compress(coco_imgs, keep_flags))

            # Determine which images go into train or validation
            cohort_year = cohort_start.date().year
            if cohort_year in partial_validation_years:
                train_per_vali = partial_validation_years[cohort_year]
                if train_per_vali == 0:
                    for coco_img in keep_imgs:
                        coco_img.img['split'] = 'vali'
                else:
                    # Split by ordinal day number, so any changes to the groups
                    # dont impact validation membership.
                    group_date = [coco_imgs[g[0]].datetime.date().toordinal() for g in groups]
                    modulus = (train_per_vali + 1)
                    is_vali_group = (np.array(group_date) % modulus) == 0
                    is_train_group = ~is_vali_group
                    for group in ub.compress(groups, is_vali_group):
                        for idx in group:
                            coco_imgs[idx].img['split'] = 'vali'
                    for group in ub.compress(groups, is_train_group):
                        for idx in group:
                            coco_imgs[idx].img['split'] = 'train'
            else:
                for coco_img in keep_imgs:
                    coco_img.img['split'] = 'train'
        else:
            raise NotImplementedError

    # Hack
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


def protocol_2img_organize(coco_imgs):
    """
    Args:
        coco_imgs (List[CocoImage]): sorted images in a corhot
    """
    # TODO: rectify this logic with the matching heuristic
    # stuff that also looks at dates.
    groups = []
    ungrouped = []
    group = None
    idx = 0
    try:
        while idx < len(coco_imgs):
            if len(coco_imgs[idx].annots()):
                # Start a new group if we find an image with annotations
                group = []
                # Check to see if the group is complete
                group.append(idx)
                idx += 1
                if not len(coco_imgs[idx].annots()):
                    # Next image should not have annots to be in the groups
                    group.append(idx)
                    idx += 1
                # Group is done
                groups.append(group)
            else:
                # cant assign this image to a group
                ungrouped.append(idx)
                idx += 1
    except IndexError:
        if group:
            groups.append(group)
    return groups, ungrouped


def protocol_3img_organize(coco_imgs):
    """
    Args:
        coco_imgs (List[CocoImage]): sorted images in a corhot
    """
    # TODO: rectify this logic with the matching heuristic
    # stuff that also looks at dates.
    groups = []
    ungrouped = []
    idx = 0
    group = None
    try:
        while idx < len(coco_imgs):
            if len(coco_imgs[idx].annots()):
                # Start a new group if we find an image with annotations
                group = []
                # Check to see if the group is complete
                group.append(idx)
                idx += 1
                if not len(coco_imgs[idx].annots()):
                    # Next image should not have annots to be in the groups
                    group.append(idx)
                    idx += 1
                    if not len(coco_imgs[idx].annots()):
                        # Next image should not have annots to be in the groups
                        group.append(idx)
                        idx += 1
                # Group is done
                groups.append(group)
            else:
                # cant assign this image to a group
                ungrouped.append(idx)
                idx += 1
    except IndexError:
        if group:
            groups.append(group)
    return groups, ungrouped


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/make_splits.py
    """
    make_splits()
