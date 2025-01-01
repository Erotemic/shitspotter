"""
Image utilities
"""
import ubelt as ub


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

    img = Image.open(fpath)
    exif_data = img._getexif()
    if exif_data is not None:
        exif = {ExifTags.TAGS[k]: v for k, v in exif_data.items()
                if k in ExifTags.TAGS}
        if 'GPSInfo' in exif:
            # TODO: get raw rationals?
            exif['GPSInfo'] = ub.map_keys(GPSTAGS, exif['GPSInfo'])
    else:
        exif = {}
    return exif


def scrub_exif_metadata(fpath, scrubbed_fpath, delta_fpath=None, remove_gps=True, remove_tags=True):
    """
    Test that we can modify EXIF data (e.g. remove GPS or timestamp)
    without modifying the image.

    Args:
        fpath (str): Path to the original image file with EXIF metadata.

        scrubbed_fpath (str):
            Path where the scrubbed image (without EXIF metadata) will be
            saved.

        delta_fpath (str | None):
            Path to save a delta file that encodes the differences between the
            original image and the scrubbed image. If None, no delta is saved.

        remove_gps (bool): if True, remove all GPS info.

        remove_tags (bool):
            if True remove all other info.  TODO: allow for a list of EXIF tags
            to keep/remove (e.g., ["Orientation", "DateTime", "ExposureTime"]).

    Example:
        >>> # xdoctest: +REQUIRES(module:PIL)
        >>> # xdoctest: +REQUIRES(module:piexif)
        >>> # Test without the delta path
        >>> from shitspotter.util.util_image import *  # NOQA
        >>> fpath = dummy_exif_image()
        >>> scrubbed_fpath = fpath.augment(stemsuffix='.scrubed')
        >>> scrub_exif_metadata(fpath, scrubbed_fpath)
        >>> import kwimage
        >>> data1 = kwimage.imread(fpath)
        >>> data2 = kwimage.imread(scrubbed_fpath)
        >>> from kwimage.im_io import _imread_exif
        >>> meta1 = _imread_exif(fpath)
        >>> meta2 = _imread_exif(scrubbed_fpath)
        >>> assert sum(len(v) for v in meta1.values() if v is not None) > 10
        >>> assert sum(len(v) for v in meta2.values() if v is not None) == 0
        >>> assert (data2 == data1).all()

    Example:
        >>> # xdoctest: +SKIP(fixme: requires xdelta3)
        >>> # xdoctest: +REQUIRES(module:PIL)
        >>> # xdoctest: +REQUIRES(module:piexif)
        >>> # Test with delta and reconstruction
        >>> from shitspotter.util.util_image import *  # NOQA
        >>> fpath = dummy_exif_image()
        >>> scrubbed_fpath = fpath.augment(stemsuffix='.scrubed')
        >>> delta_fpath = fpath.augment(stemsuffix='.delta')
        >>> # Test the ability to reconstruct
        >>> restored_fpath = fpath.augment(stemsuffix='.recon')
        >>> scrub_exif_metadata(fpath, scrubbed_fpath, delta_fpath)
        >>> import kwimage
        >>> data1 = kwimage.imread(fpath)
        >>> data2 = kwimage.imread(scrubbed_fpath)
        >>> from kwimage.im_io import _imread_exif
        >>> meta1 = _imread_exif(fpath)
        >>> meta2 = _imread_exif(scrubbed_fpath)
        >>> assert sum(len(v) for v in meta1.values() if v is not None) > 10
        >>> assert sum(len(v) for v in meta2.values() if v is not None) == 0
        >>> assert (data2 == data1).all()
        >>> unscrub_exif_metadata(scrubbed_fpath, delta_fpath, restored_fpath)
        >>> assert restored_fpath.exists()
        >>> assert ub.hash_file(restored_fpath) == ub.hash_file(fpath)
        >>> delta_size = delta_fpath.stat().st_size
        >>> scrub_size = scrubbed_fpath.stat().st_size
        >>> orig_size = fpath.stat().st_size
        >>> print(f'delta_size = {ub.urepr(delta_size, nl=1)}')
        >>> print(f'scrub_size = {ub.urepr(scrub_size, nl=1)}')
        >>> print(f'orig_size = {ub.urepr(orig_size, nl=1)}')

    Example:
        >>> # xdoctest: +SKIP(fixme: requires xdelta3)
        >>> # xdoctest: +REQUIRES(module:PIL)
        >>> # xdoctest: +REQUIRES(module:piexif)
        >>> # Test with only some tags removed
        >>> from shitspotter.util.util_image import *  # NOQA
        >>> fpath = dummy_exif_image()
        >>> scrubbed_fpath = fpath.augment(stemsuffix='.scrubed')
        >>> delta_fpath = fpath.augment(stemsuffix='.delta')
        >>> # Test the ability to reconstruct
        >>> restored_fpath = fpath.augment(stemsuffix='.recon')
        >>> scrub_exif_metadata(fpath, scrubbed_fpath, delta_fpath, remove_tags=False)
        >>> import kwimage
        >>> data1 = kwimage.imread(fpath)
        >>> data2 = kwimage.imread(scrubbed_fpath)
        >>> from kwimage.im_io import _imread_exif
        >>> meta1 = _imread_exif(fpath)
        >>> meta2 = _imread_exif(scrubbed_fpath)
        >>> assert sum(len(v) for v in meta1.values() if v is not None) > 20
        >>> assert sum(len(v) for v in meta2.values() if v is not None) < 20
        >>> assert (data2 == data1).all()
        >>> unscrub_exif_metadata(scrubbed_fpath, delta_fpath, restored_fpath)
        >>> assert restored_fpath.exists()
        >>> assert ub.hash_file(restored_fpath) == ub.hash_file(fpath)
        >>> delta_size = delta_fpath.stat().st_size
        >>> scrub_size = scrubbed_fpath.stat().st_size
        >>> orig_size = fpath.stat().st_size
        >>> print(f'delta_size = {ub.urepr(delta_size, nl=1)}')
        >>> print(f'scrub_size = {ub.urepr(scrub_size, nl=1)}')
        >>> print(f'orig_size = {ub.urepr(orig_size, nl=1)}')
    """
    import piexif
    import os
    raw_bytes = fpath.read_bytes()

    exif_dict = piexif.load(raw_bytes)

    # Define the EXIF fields to remove
    if remove_gps:
        # Specifically remove GPS data
        if "GPS" in exif_dict:
            exif_dict["GPS"] = {}  # Remove all GPS data

    if remove_tags:
        # Filter out other EXIF fields while keeping specified tags
        keep_tags = []
        for ifd, ifd_data in exif_dict.items():
            if isinstance(ifd_data, dict):  # Skip non-dictionary segments
                # idf_names = [
                #     piexif.TAGS[ifd].get(tag, {}).get("name")
                #     for tag in ifd_data
                # ]
                keys_to_remove = [
                    tag for tag in ifd_data if piexif.TAGS[ifd].get(tag, {}).get("name") not in keep_tags
                ]
                for key in keys_to_remove:
                    del ifd_data[key]

    # new_exif = piexif.dump({})
    new_exif = piexif.dump(exif_dict)

    piexif.insert(new_exif, raw_bytes, new_file=os.fspath(scrubbed_fpath))
    if delta_fpath is not None:
        # https://pypi.org/project/xdelta3/
        if delta_fpath.exists():
            delta_fpath.delete()
        # TODO: bsdiff backend? regular diff backend?
        # diff --binary foo.scrubbed.jpg foo.jpg > foo.diff
        # patch foo.scrubbed.jpg < foo.diff
        info = ub.cmd(f'xdelta3 -e -s {scrubbed_fpath} {fpath} {delta_fpath}', verbose=0)
        if info.returncode != 0:
            print('Failed:', info['command'])
            print(info.stdout)
            print(info.stderr)
            print(f'info.stdout = {ub.urepr(info.stdout, nl=1)}')
            info.check_returncode()
            raise RuntimeError


def unscrub_exif_metadata(scrubbed_fpath, delta_fpath, restored_fpath):
    if restored_fpath.exists():
        restored_fpath.delete()
    info = ub.cmd(f'xdelta3 -d -s {scrubbed_fpath} {delta_fpath} {restored_fpath}', verbose=0)
    if info.returncode != 0:
        print('Failed:', info['command'])
        print(info.stdout)
        print(info.stderr)
        print(f'info.stdout = {ub.urepr(info.stdout, nl=1)}')
        info.check_returncode()
        raise RuntimeError


def dummy_exif_image():
    import kwimage
    from PIL import Image
    import piexif
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
