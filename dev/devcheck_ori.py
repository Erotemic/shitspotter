

def check_exif_orientation(coco_dset):
    """
    Notes on orientation:
        https://jdhao.github.io/2019/07/31/image_rotation_exif_info/

        1: Upright
        8: Rot 90 clockwise
        3: Rot 180
        6: Rot 270 clockwise

        2: Flip + Upright
        7: Flip + Rot 90 clockwise
        4: Flip + Rot 180
        5: Flip + Rot 270 clockwise
    """
    import kwplot
    import xdev
    import kwimage
    kwplot.autompl()
    gids = list(coco_dset.index.imgs.keys())
    giditer = xdev.InteractiveIter(gids)
    for gid in giditer:
        # for gid in gids:
        fpath = coco_dset.get_image_fpath(gid)
        exif = extract_exif_metadata(fpath)
        exif_ori = exif.get('Orientation', None)
        print('exif_ori = {!r}'.format(exif_ori))
        # 'ExifImageHeight': 3024,
        # 'ExifImageWidth': 4032,
        # 'ImageLength': 3024,
        # 'ImageWidth': 4032,
        # Reading with GDAL/cv2 will NOT apply any exif orientation
        # but reading with skimage will
        imdata = kwimage.imread(fpath, backend='gdal', overview=-1)

        kwplot.imshow(imdata)
        xdev.InteractiveIter.draw()
