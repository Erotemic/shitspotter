"""
Helper to remove shit pictures from my phone
"""


def main():
    import shitspotter
    import kwcoco
    import ubelt as ub

    # https://askubuntu.com/questions/342319/where-are-mtp-mounted-devices-located-in-the-filesystem
    # phone_dpath = '/run/user/$UID/gvfs/mtp*'
    android_mount_base = ub.Path('/run/user')
    android_mount_dpath = list(android_mount_base.glob('*/gvfs/mtp:host=Google_Pixel_5_*'))[0]
    phone_dpath = android_mount_dpath / 'Internal shared storage/DCIM/Camera'
    phone_fpaths = list(ub.ProgIter(phone_dpath.glob('*'), desc='listing DCIM'))

    coco_fpath = shitspotter.util.find_shit_coco_fpath()
    dset = kwcoco.CocoDataset(coco_fpath)
    dvc_fpaths = [
        ub.Path(coco_img.primary_image_filepath())
        for coco_img in dset.images().coco_images
    ]

    dvc_name_to_fpath = {p.name: p for p in dvc_fpaths}
    phone_name_to_fpath = {p.name: p for p in phone_fpaths}

    common = set(dvc_name_to_fpath) & set(phone_name_to_fpath)

    print(f'{len(common)=}')
    print(f'{len(dvc_name_to_fpath)=}')
    print(f'{len(phone_name_to_fpath)=}')

    to_delete = []
    for key in ub.ProgIter(common, desc='checking files are probably the same'):
        phone_fpath = phone_name_to_fpath[key]
        dvc_fpath = dvc_name_to_fpath[key]
        # Minimal checking that these the same file
        dvc_stat = dvc_fpath.stat()
        phone_stat = phone_fpath.stat()
        assert phone_stat.st_size == dvc_stat.st_size
        assert phone_stat.st_mtime == dvc_stat.st_mtime

        to_delete.append(phone_fpath)

    for p in ub.ProgIter(to_delete, desc='deleting file'):
        p.unlink(missing_ok=True)

        # ub.hash_file(dvc_fpath)
        # ub.hash_file(phone_fpath)
