"""
Helper to remove shit pictures from my phone

CommandLine:
    python ~/code/shitspotter/dev/remote_from_phone.py

https://app.pinata.cloud/pinmanager
"""
import ubelt as ub
import rich
from dateutil import parser as dateparser


class AndroidConventions:

    @staticmethod
    def parse_image_filename(fname):
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
            >>>     info = AndroidConventions.parse_image_filename(fname)
            >>>     print('===========')
            >>>     print('fname = {!r}'.format(fname))
            >>>     print('info = {!r}'.format(info))
        """
        parsers = [
            AndroidConventions.parse_motomod_filename,
            AndroidConventions.parse_pixel_filename,
        ]
        info = None
        for parse_fn in parsers:
            info = parse_fn(fname)
            if info is not None:
                break

        if info is not None:
            info = AndroidConventions._expand_YYYYMMDD_HHMMSSmmm(info)
            info['datetime_captured'] = dateparser.parse(info['timestamp'])
        return info

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
            >>>     info = AndroidConventions.parse_pixel_filename(fname)
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


class GVFSAndroidConnection(ub.Path):
    """
    Example:
        connections = GVFSAndroidConnection.discover()
        phone = self = connections[0]
        phone_paths_infos = self.dcim_image_infos()
    """

    @classmethod
    def discover(cls, on_error='raise'):
        """
        Get a list of discovered devices

        References:
            https://askubuntu.com/questions/842409/what-is-run-user-1000-gvfs
            https://unix.stackexchange.com/questions/162900/what-is-this-folder-run-user-1000
        """
        print('Discovering MTP connections mounted via GVFS')
        # see also XDG_RUNTIME_DIR?
        android_mount_base = ub.Path('/run/user')
        connections = list(android_mount_base.glob('*/gvfs/mtp:host=*'))
        if not connections and on_error == 'raise':
            if not list(android_mount_base.glob('*')):
                raise IOError('No UIDs are specified in /run/usr')
            elif not list(android_mount_base.glob('*/gvfs')):
                raise IOError('No GVFS folders exist for any uid')
            else:
                raise IOError('No MTP hosts in GVFS folders')
        connections = list(map(cls, connections))
        print(f'Discovered {len(connections)} MTP connections mounted via GVFS')
        return connections

    def dcim_image_infos(self, verbose=1):
        camera_dpath = self / 'Internal shared storage/DCIM/Camera'
        if verbose:
            print('Note: GVFS is slow, listing items on the DCIM may take some time')
        phone_fpaths = list(ub.ProgIter(camera_dpath.glob('*'), desc='Listing DCIM', verbose=verbose))
        phone_image_infos = []
        for fpath in ub.ProgIter(phone_fpaths, desc='build image info'):
            if not fpath.name.startswith('.') and fpath.name != 'thumbnails':
                fname = fpath.name
                info = AndroidConventions.parse_image_filename(fname)
                if info is None:
                    print('Not an known image file: {}'.format(fpath))
                info['fpath'] = fpath
                info['datetime_captured'] = dateparser.parse(info['timestamp'])
                # stat = fpath.stat()
                # datetime_created = datetime.datetime.fromtimestamp(stat.st_ctime)
                # info['datetime_created'] = datetime_created
                phone_image_infos.append(info)
        return phone_image_infos


def transfer_phone_pictures():
    """
    This step does the transfer of ALL pictures from my phone to my
    image archive. The shit needs to be sorted out manually.

    I've needed to open nautilus to get the phone mounted before.
    This might be scriptable by running:

        nautilius mtp://Google_Pixel_5_0A141FDD40091U/
    """
    phone = GVFSAndroidConnection.discover()[0]
    phone_image_infos = phone.dcim_image_infos()
    print(f'Found {len(phone_image_infos)=} phone pictures')

    if not len(phone_image_infos):
        raise Exception('No items detected. Is USB preference set to "File Transfer?"')

    # This is my internal convention for storing pictures, it will not
    # generalize
    pic_dpath = ub.Path('/data/store/Pictures/')
    # Find most recent existing transfer
    transfer_infos = []
    phone_transfer_dpaths = sorted(pic_dpath.glob('Phone-DCIM-*'))

    import re
    timepat = re.compile(r'T\d\d\d\d\d\d')
    for dpath in phone_transfer_dpaths:
        year, month, day, *rest = dpath.name.split('-')[2:]
        if len(rest) == 1 and timepat.match(rest[0]):
            transfer_timestamp = dateparser.parse(f'{year}{month}{day}{rest[0]}')
        else:
            transfer_timestamp = dateparser.parse(f'{year}{month}{day}T000000')
        transfer_infos.append({
            'dpath': dpath,
            'datetime_transfer': transfer_timestamp
        })
    most_recent_dpath_info = max(transfer_infos, key=lambda x: x['datetime_transfer'])
    print('Most recent existing transfer info:')
    print('most_recent_dpath_info = {}'.format(ub.repr2(most_recent_dpath_info, nl=1)))

    prev_fpaths = list(most_recent_dpath_info['dpath'].glob('*'))
    prev_infos = []
    for fpath in prev_fpaths:
        info = AndroidConventions.parse_image_filename(fpath.name)
        info['fpath'] = fpath
        prev_infos.append(info)

    # most_recent_pic = max([p['datetime_captured'] for p in prev_infos])
    most_recent_xfer = most_recent_dpath_info['datetime_transfer']

    # Find all the new images on the phone
    needs_transfer_infos = []
    for p in phone_image_infos:
        if p['datetime_captured'] > most_recent_xfer:
            needs_transfer_infos.append(p)

    print(f'There are {len(needs_transfer_infos)} item that need transfer')

    # Create a new folder
    # oldest_time = min([p['datetime_captured'] for p in needs_transfer_infos])
    newst_time = max([p['datetime_captured'] for p in needs_transfer_infos])

    new_stamp = newst_time.strftime('%Y-%m-%d-T%H%M%S')
    new_dname = f'Phone-DCIM-{new_stamp}'

    new_dpath = pic_dpath / new_dname
    print('New Transfer Destination = {!r}'.format(new_dpath))

    # First to transfer to a temp directory so we avoid race conditions
    # tmp_dpath = new_dpath.augment(prefix='_tmp_').ensuredir()
    tmp_dpath = ub.Path(ub.augpath(new_dpath, prefix='_tmp_')).ensuredir()
    copy_jobs = []
    for p in needs_transfer_infos:
        copy_jobs.append({
            'src': p['fpath'],
            'dst': tmp_dpath / p['fpath'].name,
        })

    print(f'Start {len(needs_transfer_infos)} copy jobs to {tmp_dpath=}')

    class CopyManager:
        """
        TODO: wrap some super fast protocol like rsync.
        Progress bars like with dvc would be neat.
        """
        pass

    def safe_copy(src, dst):
        if not dst.exists():
            return shutil.copy2(src, dst)

    import shutil
    jobs = ub.JobPool(mode='thread', max_workers=8)
    eager_copy_jobs = [d for d in copy_jobs if not d['dst'].exists()]
    print(f'# Needs Copy {len(eager_copy_jobs)} / {len(copy_jobs)}')

    for copy_job in ub.ProgIter(copy_jobs, desc='submit jobs'):
        src, dst = copy_job['src'], copy_job['dst']
        job = jobs.submit(safe_copy, src, dst)
        job.copy_job = copy_job

    for job in jobs.as_completed(desc='copying'):
        job.result()

    import os
    os.rename(tmp_dpath, new_dpath)

    finalize_transfer(new_dpath)


def finalize_transfer(new_dpath):
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/shitspotter/dev'))
    from remote_from_phone import *  # NOQA
    new_dpath = "/data/store/Pictures/Phone-DCIM-2022-05-26-T173650/"
    """
    # Finalize transfer by moving new folder into the right name
    print(f'Finalize transfer to {new_dpath}')
    new_dpath = ub.Path(new_dpath)
    new_stamp = new_dpath.name.split('-', 2)[2]

    import shitspotter
    coco_fpath = shitspotter.util.find_shit_coco_fpath()
    asset_dpath = coco_fpath.parent / 'assets'

    new_shit_dpath = asset_dpath / f'poop-{new_stamp}'
    new_shit_dpath.ensuredir()

    print('Need to manually move shit images')

    print('from new_dpath = {!r}'.format(new_dpath))
    print('to new_shit_dpath = {!r}'.format(new_shit_dpath))

    import xdev
    xdev.startfile(new_dpath)
    xdev.startfile(new_shit_dpath)

    from rich.prompt import Confirm

    ans = Confirm.ask('Manually move images and then enter y to continue')
    while not ans:
        ans = Confirm.ask('Manually move images and then enter y to continue')

    print('The next step is to run...')
    print(ub.codeblock(
        '''
        # The gather script
        python -m shitspotter.gather

        # The matching script
        python -m shitspotter.matching autofind_pair_hueristic

        # The plots script
        python -m shitspotter.plots update_analysis_plots
        '''))

    # print('Next step is to run the gather script: `python -m shitspotter.gather`')
    # print('Next step is to run the matching script: `python -m shitspotter.matching autofind_pair_hueristic`')
    # print('Next step is to run the plots script: `python -m shitspotter.plots update_analysis_plots`')
    print('')
    print('Then repin the updated dataset to IPFS')

    shitspotter_dvc_dpath = coco_fpath.parent
    command = ub.codeblock(
        f'''
        # Pin the new folder directly.
        ipfs add --pin -r {new_shit_dpath} --progress --cid-version=1 --raw-leaves=false | tee "new_pin_job.log"
        NEW_FOLDER_CID=$(tail -n 1 new_pin_job.log | cut -d ' ' -f 2)
        echo "NEW_FOLDER_CID=$NEW_FOLDER_CID"

        echo "
        On MOJO run:

        NEW_FOLDER_CID=$NEW_FOLDER_CID
        ipfs pin add --progress "$NEW_FOLDER_CID"
        "

        # Then re-add the root, which gives us the new CID
        ipfs add --pin -r {shitspotter_dvc_dpath} --progress --cid-version=1 --raw-leaves=false | tee "root_pin_job.log"

        # Programatically grab the new CID:
        NEW_ROOT_CID=$(tail -n 1 root_pin_job.log | cut -d ' ' -f 2)
        echo "NEW_ROOT_CID=$NEW_ROOT_CID"

        # Add it to the CID revisions:
        echo "$NEW_ROOT_CID" >> $HOME/code/shitspotter/shitspotter/cid_revisions.txt

        echo "
        Then on MOJO run:

        NEW_ROOT_CID=$NEW_ROOT_CID
        DATE=$(date +"%Y-%m-%d")
        ipfs pin add --progress "$NEW_ROOT_CID"

        # Add pin to web3 remote storage
        ipfs pin remote add --service=web3.storage.erotemic --name=shitspotter-dvc-$DATE $NEW_ROOT_CID --background

        # Query status of remote pin
        ipfs pin remote ls --service=web3.storage.erotemic --cid=$NEW_ROOT_CID --status=queued,pinning,pinned,failed
        "
        '''
    )
    print(command)

    # dpath = ub.Path(shitspotter.__file__).parent
    # cid_revisions_fpath = dpath / 'cid_revisions.txt'
    command = ub.codeblock(
        '''

        OLD EXAMPLES:

        echo "QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
        echo "QmaPPoPs7wXXkBgJeffVm49rd63ZtZw5GrhvQQbYrUbrYL" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
        echo "QmaSfRtzXDCiqyfmZuH6NEy2HBr7radiJNhmSjiETihoh6" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
        echo "QmPptXKFKi6oTJL3VeCNy5Apk8MJsHhCAAwVmegHhuRY83" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
        echo "QmfStoay5rjeHMEDiyuGsreXNHsyiS5kVaexSM2fov216j" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
        echo "bafybeihgex2fj4ethxnfmiivasw5wqsbt2pdjswu3yr554rewm6myrkq4a" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
        echo "bafybeihltrtb4xncqvfbipdwnlxsrxmeb4df7xmoqpjatg7jxrl3lqqk6y" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
        echo "bafybeihi7v7sgnxb2y57ie2dr7oobigsn5fqiwxwq56sdpmzo5on7a2xwe" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
        echo "bafybeiedk6bu2qpl4snlu3jmtri4b2sf476tgj5kdg2ztxtm7bd6ftzqyy" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt


        Then on mojo:

        NEW_ROOT_CID=bafybeihi7v7sgnxb2y57ie2dr7oobigsn5fqiwxwq56sdpmzo5on7a2xwe
        NEW_ROOT_CID=bafybeiedk6bu2qpl4snlu3jmtri4b2sf476tgj5kdg2ztxtm7bd6ftzqyy
        DATE=$(date +"%Y-%m-%d")

        ipfs pin add --progress "${NEW_ROOT_CID}"

        # Also, we should pin the CID on a pinning service
        ipfs pin remote add --service=web3.storage.erotemic --name=shitspotter-dvc-$DATE ${NEW_ROOT_CID} --background
        ipfs pin remote ls --service=web3.storage.erotemic --cid=${NEW_ROOT_CID} --status=queued,pinning,pinned,failed

        # e.g.
        ipfs pin add bafybeihgex2fj4ethxnfmiivasw5wqsbt2pdjswu3yr554rewm6myrkq4a --progress
        ipfs pin add QmfStoay5rjeHMEDiyuGsreXNHsyiS5kVaexSM2fov216j --progress
        ipfs pin remote add --service=web3.storage.erotemic --name=shitspotter-dvc-2022-06-08 bafybeihgex2fj4ethxnfmiivasw5wqsbt2pdjswu3yr554rewm6myrkq4a --background
        ipfs pin remote ls --service=web3.storage.erotemic --cid=QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq --status=queued,pinning,pinned,failed
        '''
    )
    print(command)

    # Try to programatically figure out what the CID was
    r"""
    import ubelt as ub
    out = ub.cmd('ipfs pin ls --type=recursive')['out']
    cids = []
    for line in out.strip().split('\n'):
        cid = line.split(' ')[0]
        cids.append(cid)

    for cid in cids:
        result = ub.cmd(f'ipfs ls {cid}')['out']
        for line in result.strip().split('\n'):
            if 'data.kwcoco.json' in line:
                print()
                print(line)
                print(cid)
                print()
                break
    """


def delete_shit_images_on_phone():
    """
    Looks for shit images that have been backed up in the shitspotter database
    and then removes them from the phone SD card.
    """
    import shitspotter
    import kwcoco
    phone = GVFSAndroidConnection.discover()[0]
    phone_image_infos = phone.dcim_image_infos()
    phone_fpaths = [p['fpath'] for p in phone_image_infos]

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
        # assert phone_stat.st_mtime == dvc_stat.st_mtime

        to_delete.append(phone_fpath)

    for p in ub.ProgIter(to_delete, desc='deleting file'):
        p.unlink(missing_ok=True)

        # ub.hash_file(dvc_fpath)
        # ub.hash_file(phone_fpath)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/dev/remote_from_phone.py
    """
    transfer_phone_pictures()
