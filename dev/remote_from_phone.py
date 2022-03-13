"""
Helper to remove shit pictures from my phone
"""
import ubelt as ub
from dateutil import parser


def list_phone_image_paths():
    """
    Get a list of file paths / infos about image / videos on the pixel phone
    """
    # import datetime
    # https://askubuntu.com/questions/342319/where-are-mtp-mounted-devices-located-in-the-filesystem
    # phone_dpath = '/run/user/$UID/gvfs/mtp*'
    android_mount_base = ub.Path('/run/user')
    android_mount_dpath = list(android_mount_base.glob('*/gvfs/mtp:host=Google_Pixel_5_*'))[0]
    phone_dpath = android_mount_dpath / 'Internal shared storage/DCIM/Camera'
    phone_fpaths = list(ub.ProgIter(phone_dpath.glob('*'), desc='listing DCIM'))

    phone_path_infos = []
    for fpath in ub.ProgIter(phone_fpaths, desc='build image info'):
        if not fpath.name.startswith('.') and fpath.name != 'thumbnails':
            fname = fpath.name
            info = parse_android_filename(fname)
            if info is None:
                print('Not an known image file: {}'.format(fpath))
            info['fpath'] = fpath
            phone_path_infos.append(info)

            # stat = fpath.stat()
            # datetime_created = datetime.datetime.fromtimestamp(stat.st_ctime)
            datetime_captured = parser.parse(info['timestamp'])
            phone_path_infos.append({
                'fpath': fpath,
                'datetime_captured': datetime_captured,
                # 'datetime_created': datetime_created,
            })

    return phone_path_infos


def transfer_phone_pictures():
    """
    This step does the transfer of ALL pictures from my phone to my
    image archive. The shit needs to be sorted out manually.
    """
    phone_path_infos = list_phone_image_paths()

    #
    pic_dpath = ub.Path('/data/store/Pictures/')
    # Find most recent existing transfer
    transfer_infos = []
    for dpath in pic_dpath.glob('Phone-DCIM-*'):
        year, month, day = dpath.name.split('-')[2:5]
        transfer_timestamp = parser.parse(f'{year}{month}{day}T000000')
        transfer_infos.append({
            'dpath': dpath,
            'datetime_transfer': transfer_timestamp
        })
    most_recent_dpath_info = max(transfer_infos, key=lambda x: x['datetime_transfer'])
    prev_fpaths = most_recent_dpath_info['dpath'].glob('*')
    prev_infos = []
    for fpath in prev_fpaths:
        info = parse_android_filename(fpath.name)
        info['fpath'] = fpath
        prev_infos.append(info)

    # most_recent_pic = max([p['datetime_captured'] for p in prev_infos])
    most_recent_xfer = most_recent_dpath_info['datetime_transfer']

    # Find all the new images on the phone
    needs_transfer_infos = []
    for p in phone_path_infos:
        if p['datetime_captured'] > most_recent_xfer:
            needs_transfer_infos.append(p)

    # Create a new folder
    # oldest_time = min([p['datetime_captured'] for p in needs_transfer_infos])
    newst_time = max([p['datetime_captured'] for p in needs_transfer_infos])

    new_stamp = newst_time.strftime('%Y-%m-%d-T%H%M%S')
    new_dname = f'Phone-DCIM-{new_stamp}'

    new_dpath = pic_dpath / new_dname

    # First to transfer to a temp directory so we avoid race conditions
    tmp_dpath = new_dpath.augment(prefix='_tmp_').ensuredir()

    copy_jobs = []
    for p in needs_transfer_infos:
        copy_jobs.append({
            'src': p['fpath'],
            'dst': tmp_dpath / p['fpath'].name,
        })

    import shutil
    jobs = ub.JobPool(mode='thread', max_workers=8)
    for copy_job in ub.ProgIter(copy_jobs, desc='submit jobs'):
        jobs.submit(shutil.copy, copy_job['src'], copy_job['dst'])

    for job in jobs.as_completed(desc='copying'):
        job.result()


def main():
    import shitspotter
    import kwcoco

    phone_path_infos = list_phone_paths()
    phone_fpaths = [p['fpath'] for p in phone_path_infos]

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


def parse_android_filename(fname):
    """
    Generic parser over multiple android formats

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/shitspotter/dev'))
        >>> from remote_from_phone import *  # NOQA
        >>> fnames = [
        >>>     'PXL_20210528_144025399~2.jpg',
        >>>     'PXL_20210528_144025399.jpg',
        >>>     'PXL_20210528_143924158.MP.jpg',
        >>>     'PXL_20200820_143015285.NIGHT.jpg',
        >>>     'PXL_20200820_143019420.PORTRAIT-01.COVER.jpg',
        >>>     'PXL_20200820_142352990.mp4',
        >>>     'IMG_20201110_112513325_HDR.jpg',  # probably an older motorola format?
        >>>     'IMG_20201112_111836850_BURST001.jpg',
        >>>     'IMG_20201112_111836850_BURST000_COVER_TOP.jpg',
        >>> ]
        >>> for fname in fnames:
        >>>     info = parse_android_filename(fname)
        >>>     print('===========')
        >>>     print('fname = {!r}'.format(fname))
        >>>     print('info = {!r}'.format(info))
    """
    parsers = [
        parse_motomod_filename,
        parse_pixel_filename,
    ]
    info = None
    for parse_fn in parsers:
        info = parse_fn(fname)
        if info is not None:
            break

    if info is not None:
        info = _expand_YYYYMMDD_HHMMSSmmm(info)
        info['datetime_captured'] = parser.parse(info['timestamp'])
    return info


def parse_motomod_filename(fname):
    """
    Parse info out of a motorola android image / video filename.
    """
    import parse
    moto_pattern_v1 = parse.Parser('{prefix}_{YYYYMMDD:.8}_{HHMMSSmmm:.9}_{extra}.{modifiers_and_ext}')
    info = None
    if info is None:
        result = moto_pattern_v1.parse(fname)
        if result:
            info = result.named
    return info


def _expand_YYYYMMDD_HHMMSSmmm(info):
    date_part = info.pop('YYYYMMDD')
    time_part = info.pop('HHMMSSmmm')
    hour = time_part[0:2]
    minute = time_part[2:4]
    second = time_part[4:6]
    mili = time_part[6:]
    iso_timestamp = f'{date_part}T{hour}:{minute}:{second}.{mili}'
    info['timestamp'] = iso_timestamp
    return info


def parse_pixel_filename(fname):
    """
    Parse info out of a pixel android image / video filename.

    Given an image / video filename taken via a pixel android phone,
    parse out info from the filename. Return None if the filename
    does not match a known pattern.

    Args:
        fname : filename of the image / video

    Returns:
        None | Dict: info dict

    References:
        https://9to5google.com/2020/08/20/google-camera-7-5-pxl/
        https://support.google.com/photos/thread/11837254/what-do-google-file-names-mean?hl=en

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/shitspotter/dev'))
        >>> from remote_from_phone import *  # NOQA
        >>> fnames = [
        >>>     'PXL_20210528_144025399~2.jpg',
        >>>     'PXL_20210528_144025399.jpg',
        >>>     'PXL_20210528_143924158.MP.jpg',
        >>>     'PXL_20200820_143015285.NIGHT.jpg',
        >>>     'PXL_20200820_143019420.PORTRAIT-01.COVER.jpg',
        >>>     'PXL_20200820_142352990.mp4',
        >>>     'IMG_20201110_112513325_HDR.jpg',  # probably an older motorola format?
        >>> ]
        >>> for fname in fnames:
        >>>     info = parse_pixel_filename(fname)
        >>>     print('===========')
        >>>     print('fname = {!r}'.format(fname))
        >>>     print('info = {!r}'.format(info))
    """
    import parse
    parse.parse

    def parse_dupid(text):
        return text
    parse_dupid.pattern = r'(~\d+)?'
    # re.search(r'~\d+', '~33')

    android_11_pattern = parse.Parser('{prefix}_{YYYYMMDD:.8}_{HHMMSSmmm:.9}{dupid:Dupid}.{modifiers_and_ext}', extra_types=dict(Dupid=parse_dupid))

    def expand_modifiers_and_ext(info):
        *modifiers, ext = info.pop('modifiers_and_ext').split('.')
        info['ext'] = ext
        info['modifiers'] = modifiers
        return info

    info = None
    if not info:
        result = android_11_pattern.parse(fname)
        if result:
            info = expand_modifiers_and_ext(result.named)
    return info
