"""
Image utilities
"""


def imread_with_exif(fpath, overview=None):
    import kwimage
    import numpy as np
    exif = extract_exif_metadata(fpath)
    exif_ori = exif.get('Orientation')
    imdata = kwimage.imread(fpath, backend='gdal', overview=overview)
    if exif_ori is not None:
        if exif_ori == 1:
            pass
        elif exif_ori == 6:
            imdata = np.rot90(imdata, k=-1)
        elif exif_ori == 8:
            imdata = np.rot90(imdata, k=1)
        elif exif_ori == 3:
            imdata = np.rot90(imdata, k=2)
        else:
            raise NotImplementedError(exif_ori)
    return imdata


def extract_exif_metadata(fpath):
    from PIL import Image, ExifTags
    from PIL.ExifTags import GPSTAGS
    import ubelt as ub

    img = Image.open(fpath)
    exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items()
            if k in ExifTags.TAGS}
    if 'GPSInfo' in exif:
        # TODO: get raw rationals?
        exif['GPSInfo'] = ub.map_keys(GPSTAGS, exif['GPSInfo'])
    return exif
