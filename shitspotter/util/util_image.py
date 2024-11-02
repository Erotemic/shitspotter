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


def scrub_exif_metadata(fpath, new_fpath):
    """
    Test that we can modify EXIF data (e.g. remove GPS or timestamp)
    without modifying the image.

    Example:
        >>> # xdoctest: +REQUIRES(module:PIL)
        >>> # xdoctest: +REQUIRES(module:piexif)
        >>> from shitspotter.util.util_image import *  # NOQA
        >>> from shitspotter.util.util_image import *  # NOQA
        >>> fpath = _dummy_exif_image()
        >>> new_fpath = fpath.augment(stemsuffix='scrubed')
        >>> scrub_exif_metadata(fpath, new_fpath)
        >>> import kwimage
        >>> data1 = kwimage.imread(fpath)
        >>> data2 = kwimage.imread(new_fpath)
        >>> from kwimage.im_io import _imread_exif
        >>> meta1 = _imread_exif(fpath)
        >>> meta2 = _imread_exif(new_fpath)
        >>> assert sum(len(v) for v in meta1.values() if v is not None) > 10
        >>> assert sum(len(v) for v in meta2.values() if v is not None) == 0
        >>> assert (data2 == data1).all()

    Ignore:
        ub.cmd(f'xxd {fpath} > b1.hex', shell=True)
        ub.cmd(f'xxd {new_fpath} > b2.hex', shell=True)
        _ = ub.cmd('diff -y b1.hex b2.hex', verbose=3, shell=1)
    """
    import piexif
    import os
    new_exif = piexif.dump({})
    raw_bytes = fpath.read_bytes()
    # old_exif = piexif.load(raw_bytes)
    piexif.insert(new_exif, raw_bytes, new_file=os.fspath(new_fpath))


def _dummy_exif_image():
    import kwimage
    from PIL import Image
    import piexif
    import ubelt as ub
    dpath = ub.Path.appdir('shitspotter/tests/exif').ensuredir()
    fpath = dpath / 'test_1.jpg'
    # Create a basic EXIF dictionary with dummy camera data
    # Reference: https://piexif.readthedocs.io/en/latest/functions.html#piexif.dump
    raw_exif = {
        "0th": {
            piexif.ImageIFD.Make: b"FakeMake",
            piexif.ImageIFD.Model: b"FakeModel",
            piexif.ImageIFD.Software: b"FakeSoftware 1.0",
            piexif.ImageIFD.ImageDescription: b"Sample image with dummy EXIF data",
            piexif.ImageIFD.DateTime: b"2345:01:23 12:34:56",  # YYYY:MM:DD HH:MM:SS format
        },
        "Exif": {
            piexif.ExifIFD.ExposureTime: (1, 60),  # Exposure time (1/60 second)
            piexif.ExifIFD.FNumber: (28, 10),  # F/2.8 aperture
            piexif.ExifIFD.ISOSpeedRatings: 100,  # ISO speed
            piexif.ExifIFD.ShutterSpeedValue: (600, 100),  # Shutter speed in APEX value
            piexif.ExifIFD.ApertureValue: (280, 100),  # Aperture in APEX value
            piexif.ExifIFD.BrightnessValue: (5, 1),  # Brightness level
            piexif.ExifIFD.FocalLength: (50, 1),  # Focal length (50mm)
            piexif.ExifIFD.LensMake: b"FakeLensMake",  # Lens manufacturer
            piexif.ExifIFD.LensModel: b"FakeLensModel",  # Lens model
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLatitude: [(40, 1), (0, 1), (0, 1)],  # 40°0'0" N
            piexif.GPSIFD.GPSLongitudeRef: b"W",
            piexif.GPSIFD.GPSLongitude: [(74, 1), (0, 1), (0, 1)],  # 74°0'0" W
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: (100, 1),  # 100 meters above sea level
        },
        # Writing 1st doesnot seem to work with piexif, need to understand why
        "1st": {},
        "thumbnail": None,
        #"1st": {
        #    piexif.ImageIFD.Make: "Canon",
        #    piexif.ImageIFD.XResolution: (40, 1),
        #    piexif.ImageIFD.YResolution: (40, 1),
        #    piexif.ImageIFD.Software: "piexif"
        #},
        #"thumbnail": None,
        "Interop": {
            piexif.InteropIFD.InteroperabilityIndex: b"R98",  # Interoperability index
        }
    }
    exif_bytes = piexif.dump(raw_exif)
    imdata = kwimage.grab_test_image()
    pil_img = Image.fromarray(imdata)
    pil_img.save(fpath, exif=exif_bytes)
    return fpath
