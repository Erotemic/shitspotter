"""
Helper to remove shit pictures from my phone

CommandLine:
    python -m shitspotter.phone_manager

https://app.pinata.cloud/pinmanager

TODO:
    - [x] Move to a staging area first so images can be scrubbed if necessary.
"""
import ubelt as ub
import os
import re
import scriptconfig as scfg
import parse
from dateutil import parser as dateparser


class RemoteFromPhoneConfig(scfg.DataConfig):
    """
    This is an interactive script. It will:

        1. Discover an Android Device connected to a PC in USB file transfer mode.

        2. Copy all images it infers as new into a `pic_dpath` on your local PC

        3. Pop up the new folder and an empty new shitspotter folder.

        4. Ask the user to manually move all shit pictures into the new
           shitspotter folder.

        5. Ask the user to confirm when this process is done.

        6. Print instructions for subsequent commands to register the new
           images with the dataset / upload them to IPFS.

    Note:
        The :mod:`shitspotter` module is used to identify the dataset path.
    """
    pic_dpath = scfg.Value('/data/store/Pictures/', help=(
        'The location on the PC to transfer all new pictures to'))

    @classmethod
    def main(RemoteFromPhoneConfig, argv=True, **kwargs):
        """
        Execute main transfer logic

        Ignore:

            IPython interaction starting point.

            >>> import sys, ubelt
            >>> sys.path.append(ubelt.expandpath('~/code/shitspotter'))
            >>> from shitspotter.phone_manager import *  # NOQA
            >>> argv = False
            >>> kwargs = {}


        """
        # STEP 1:
        config = RemoteFromPhoneConfig.cli(argv=argv, strict=True, data=kwargs)
        import rich
        if config.verbose:
            rich.print('config = {}'.format(ub.urepr(config, nl=1)))
        # Import to make sure they are installed
        import shitspotter  # NOQA
        import xdev  # NOQA
        import pickle
        cache_dpath = ub.Path.appdir('shitspotter/transfer_session').ensuredir()
        lock_fpath = cache_dpath / 'transfering.lock'
        prepared_transfer_fpath = cache_dpath / 'prepared_transfer.pkl'
        if lock_fpath.exists():
            raise Exception(
                f'Previous transfer lockfile exists: {lock_fpath}. '
                'Needs to implement resume or cleanup dirty state')
        lock_fpath.touch()

        try:
            # STEP 2:
            new_dpath, needs_transfer_infos = prepare_phone_transfer(config)
        except Exception:
            print('FIXME: delete cache?')
            cache_dpath.delete()
            raise
        else:
            # STEP 3:
            # Write the intermediate state to a cache so we can resume if there
            # is an interruption. (TODO: resuming needs to be implemented)
            prepared_transfer_fpath.write_bytes(pickle.dumps({
                'new_dpath': new_dpath,
                'needs_transfer_infos': needs_transfer_infos,
            }))

            # STEP 4:
            # Move everything local
            transfer_phone_pictures(new_dpath, needs_transfer_infos)

            # STEP 5:
            finalize_transfer(new_dpath)
        finally:
            # STEP 6:
            cache_dpath.delete()


class AndroidConventions:
    """
    Parse information out of standard android file names.
    """

    @staticmethod
    def parse_image_filename(fname):
        """
        Generic parser over multiple android formats

        Example:
            >>> from shitspotter.phone_manager import *  # NOQA
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
            >>> from shitspotter.phone_manager import *  # NOQA
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
        def parse_dupid(text):
            return text
        parse_dupid.pattern = r'(~\d+)?'

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


class GVFSAndroidPath(ub.Path):
    """
    Discover connected android devices using a filesystem interface.

    Example:
        connections = GVFSAndroidPath.discover()
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


class SFTPAndroidConnection:
    """
    The SSH Server app can be used to start a sftp connection
    """

    def __init__(self, sftp):
        self.sftp = sftp

    @classmethod
    def from_hostname(cls):
        """
        Generated in part by ChatGPT
        """
        host = os.environ.get('PIXEL5_HOST', None)
        phone_pass = os.environ.get('PIXEL5_PASS', None)
        assert host
        if not phone_pass:
            raise AssertionError
        import paramiko
        from paramiko import Transport, SFTPClient
        ssh_config = paramiko.SSHConfig()
        with open(ub.Path("~/.ssh/config").expand()) as ssh_config_file:
            ssh_config.parse(ssh_config_file)
        # Look up the host configuration
        host_config = ssh_config.lookup(host)
        hostname = host_config["hostname"]
        username = host_config.get("user")
        port = int(host_config.get("port", 22))
        # key_path = host_config.get("identityfile", [None])[-1]
        # FIXME: public key auth does not seem to be working.
        # Using a password in the meantime.
        transport = Transport((hostname, port))
        transport.connect(username=username, password=phone_pass)
        sftp = SFTPClient.from_transport(transport)
        self = cls(sftp)
        return self

    def dcim_image_infos(self, verbose=1):
        self.sftp.chdir('/0/DCIM/Camera')
        file_names = self.sftp.listdir()

        phone_image_infos = []
        for fname in ub.ProgIter(file_names, desc='build image info'):
            if not fname.startswith('.') and fname != 'thumbnails':
                info = AndroidConventions.parse_image_filename(fname)
                if info is None:
                    print('Not an known image file: {}'.format(fname))
                info['fpath'] = f'/0/DCIM/Camera/{fname}'
                info['datetime_captured'] = dateparser.parse(info['timestamp'])
                # stat = fpath.stat()
                # datetime_created = datetime.datetime.fromtimestamp(stat.st_ctime)
                # info['datetime_created'] = datetime_created
                phone_image_infos.append(info)
        return phone_image_infos


def prepare_phone_transfer(config):
    """
    Gather information about the files that need to be transfered.
    """

    USE_GVFS = True
    if USE_GVFS:
        # TODO: get this working over sftp as well
        found = GVFSAndroidPath.discover()
        assert len(found) == 1, 'should only have 1'
        phone = found[0]
        phone_image_infos = phone.dcim_image_infos()
        print(f'Found {len(phone_image_infos)=} phone pictures')
    else:
        sftp_conn = SFTPAndroidConnection.from_hostname()
        phone_image_infos = sftp_conn.dcim_image_infos()

    if not len(phone_image_infos):
        raise Exception(ub.paragraph(
            '''
            No GVFS items detected.  Is the phone USB preference set to "File
            Transfer?". This property must be set each time the phone is
            connected to the PC. On an Android Pixel 5 unlock the phone, swipe
            down to access notifications. At the bottom select "Charging this
            USB device", which then expands and allows you to "Tap for more
            options". Selecting this opens USB Preferences. Change the "Use USB
            for" setting to "File transfer / Android Audio". Finally rerun the
            script.
            '''))

    # This is my internal convention for storing pictures, it will not
    # generalize
    # pic_dpath = ub.Path('/data/store/Pictures/')
    pic_dpath = ub.Path(config['pic_dpath'])
    # Find most recent existing transfer
    transfer_infos = []
    phone_transfer_dpaths = sorted(pic_dpath.glob('Phone-DCIM-*'))

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
        if info is None:
            continue
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
    return new_dpath, needs_transfer_infos


def transfer_phone_pictures(new_dpath, needs_transfer_infos):
    """
    This step does the transfer of ALL pictures from my phone to my
    image archive. The shit needs to be sorted out manually.

    I've needed to open nautilus to get the phone mounted before. But this
    doesn't always seem to be necessary.
    This might be scriptable by running:

        nautilius mtp://Google_Pixel_5_0A141FDD40091U/
    """
    import kwutil
    # First to transfer to a temp directory so we avoid race conditions
    # tmp_dpath = new_dpath.augment(prefix='_tmp_').ensuredir()
    tmp_dpath = new_dpath.augment(prefix='_tmp_').ensuredir()
    print(f'Start {len(needs_transfer_infos)} copy jobs to {tmp_dpath=}')

    USE_GVFS = True
    if USE_GVFS:
        copyman = kwutil.CopyManager(mode='thread', workers=8)
        for p in needs_transfer_infos:
            copyman.submit(
                src=p['fpath'],
                dst=tmp_dpath / p['fpath'].name,
                skip_existing=True,
            )

        # print(f'# Needs Copy {len(eager_copy_jobs)} / {len(copy_jobs)}')
        # copyman.report(sizes=False)
        copyman.run()
    else:
        sftp_conn = SFTPAndroidConnection.from_hostname()
        sftp = sftp_conn.sftp
        for p in ub.ProgIter(needs_transfer_infos, desc='sftp transfer'):
            dst = tmp_dpath / ub.Path(p['fpath']).name
            if not dst.exists():
                sftp.get(p['fpath'], dst)

    print('Copy finished, renaming...')
    os.rename(tmp_dpath, new_dpath)
    print('Rename finished')
    return new_dpath


def finalize_transfer(new_dpath):
    """
    TODO:
        This needs to be update to handle privacy scrubbing

    Ignore:
        from shitspotter.phone_manager import *  # NOQA
        new_dpath = "/data/store/Pictures/Phone-DCIM-2023-03-11-T165018"
    """
    import shitspotter
    import xdev
    from rich.prompt import Confirm
    # Finalize transfer by moving new folder into the right name
    print(f'Finalize transfer to {new_dpath}')
    new_dpath = ub.Path(new_dpath)
    new_stamp = new_dpath.name.split('-', 2)[2]
    new_name = f'poop-{new_stamp}'

    staging_dpath = shitspotter.util.util_data.find_staging_dpath()

    shitspotter_dvc_dpath = shitspotter.util.util_data.find_data_dpath()
    asset_dpath = shitspotter_dvc_dpath / 'assets'
    new_shit_dpath = asset_dpath / new_name

    staging_asset_dpath = staging_dpath / 'assets'
    staging_shit_dpath = staging_asset_dpath / new_name
    staging_shit_dpath.ensuredir()

    print('Need to manually move shit images')

    print('from new_dpath = {!r}'.format(new_dpath))
    print('to new_shit_dpath = {!r}'.format(staging_shit_dpath))

    xdev.startfile(new_dpath)
    xdev.startfile(staging_shit_dpath)

    ans = Confirm.ask('Manually move images and then enter y to continue')
    while not ans:
        ans = Confirm.ask('Manually move images and then enter y to continue')

    print(ub.codeblock(
        fr'''
        TODO: the next step is to scrub images of metadata depending on what
        the privacy policy is.

        This will require that we:

            * Have the transcrypt repo decrypted

            * From there you can call the script:
                source ~/code/shitspotter/secrets/secret_setup.sh
                mount_shit_secrets

              which will mount the encrypted file in rw mode, to write any of
              the secret metadata we want to hide, but not outright delete.

            * Call logic in shitspotter.gather_from_staging

                e.g.

                python -m shitspotter.gather_from_staging \
                    --staging_dpath '{staging_dpath}' \
                    --shitspotter_dvc_dpath '{shitspotter_dvc_dpath}' \
                    --staging_shit_dpath '{staging_shit_dpath}'

            * This should place a bunch of new images from the staging folder
              into the final folder, and some of the images may be scrubbed.

              Can now re-encrypt the secret metadata.

            source ~/code/shitspotter/secrets/secret_setup.sh
            dismount_shit_secrets

        We should seed predictions on the new data with

            python -m shitspotter.cli.predict

            python -m shitspotter.cli.predict \
                --src {new_shit_dpath} \
                --package_fpath ~/code/shitspotter/shitspotter_dvc/models/maskrcnn/train_baseline_maskrcnn_v3_v_966e49df_model_0014999.pth \
                --create_labelme True

            # Also cleanup the intermediate directories...
            # or fix the above script to do that.
            rm -rf {new_shit_dpath}-predict-output

        e.g.

            python -m shitspotter.cli.predict \
                --src /home/joncrall/code/shitspotter/shitspotter_dvc/assets/_contributions/sam-2025-03-07 \
                --package_fpath ~/code/shitspotter/shitspotter_dvc/models/maskrcnn/train_baseline_maskrcnn_v3_v_966e49df_model_0014999.pth \
                --create_labelme True

            python -m shitspotter.cli.predict \
                --src /home/joncrall/code/shitspotter/shitspotter_dvc/assets/poop-2025-03-08-T224918 \
                --package_fpath ~/code/shitspotter/shitspotter_dvc/models/maskrcnn/train_baseline_maskrcnn_v3_v_966e49df_model_0014999.pth \
                --create_labelme True

        At this point, we can do any annotation we wish, but whenever
        annotations change we need to rerun gather and make splits.

        cd {new_shit_dpath}
        labelme .
        '''))

    print('# Ensure IPFS is running')
    print('sudo systemctl start ipfs')

    print('The next step is to run...')
    print(ub.highlight_code(ub.codeblock(
        '''
        # The gather script
        python -m shitspotter.gather

        # The train/vali splits
        python -m shitspotter.make_splits

        # The matching script
        python -m shitspotter.matching autofind_pair_hueristic

        # The plots script
        python -m shitspotter.plots update_analysis_plots

        # Update the README based on the output of these scripts
        '''), 'bash'))

    print_pin_instructions(shitspotter_dvc_dpath, new_shit_dpath)


def print_pin_instructions(shitspotter_dvc_dpath, new_shit_dpath):
    """
    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/shitspotter'))
        from shitspotter.phone_manager import *  # NOQA
        import shitspotter
        shitspotter_dvc_dpath = shitspotter.util.util_data.find_data_dpath()
        new_shit_dpath = ub.Path('/home/joncrall/code/shitspotter/shitspotter_dvc/assets/poop-2024-12-30-T212347')
    """
    # print('Next step is to run the gather script: `python -m shitspotter.gather`')
    # print('Next step is to run the matching script: `python -m shitspotter.matching autofind_pair_hueristic`')
    # print('Next step is to run the plots script: `python -m shitspotter.plots update_analysis_plots`')
    print('')
    print('# Then repin the updated dataset to IPFS')

    import kwutil
    today = kwutil.util_time.datetime.coerce('now').date()
    today_iso = today.isoformat()

    new_assets_name = 'shitspotter-assets-' + new_shit_dpath.name
    new_dataset_name = f'shitspotter-{today_iso}'

    command = ub.codeblock(
        f'''
        # Pin the new folder directly.
        ipfs add --pin -r {new_shit_dpath} --progress --cid-version=1 --raw-leaves=false | tee "new_pin_job.log"
        NEW_ASSETS_CID=$(tail -n 1 new_pin_job.log | cut -d ' ' -f 2)
        echo "NEW_ASSETS_CID=$NEW_ASSETS_CID"

        # Add a name to the new pin on the local machine.
        ipfs pin add --name {new_assets_name} --progress -- "$NEW_ASSETS_CID"

        echo "
        On IPFS server run:

        ipfs pin add --name {new_assets_name} --progress -- $NEW_ASSETS_CID
        "

        # ---

        # Then re-add the root, which gives us the new CID
        ipfs add --pin -r {shitspotter_dvc_dpath} --progress --cid-version=1 --raw-leaves=false | tee "root_pin_job.log"
        # Programatically grab the new CID:
        NEW_ROOT_CID=$(tail -n 1 root_pin_job.log | cut -d ' ' -f 2)
        echo "NEW_ROOT_CID=$NEW_ROOT_CID"

        # Add a name to the new pin on the local machine.

        ipfs pin add --name {new_dataset_name} --progress -- "$NEW_ROOT_CID"

        # Add it to the CID revisions:
        echo "$NEW_ROOT_CID" >> "$HOME"/code/shitspotter/shitspotter/cid_revisions.txt

        echo "
        # Then on IPFS server run:

        ipfs pin add --name {new_dataset_name} --progress $NEW_ROOT_CID
        "

        # Also see: ~/code/shitspotter/dev/sync_shit.sh


        NEW:

        We also want to update the IPNS address

        NEW_ROOT_CID=$(tail -n 1 root_pin_job.log | cut -d ' ' -f 2)
        echo "NEW_ROOT_CID=$NEW_ROOT_CID"

        echo "
        # On IPFS server run:
        crontab -l | sed 's/bafybe[^ ]* /$NEW_ROOT_CID /' | crontab -

        # Check if the change worked
        crontab -l

        # Cron entry should look like this:
        IPFS_PATH=/flash/ipfs
        0 0 * * * /home/joncrall/.local/bin/ipfs name publish --key=shitspotter-key /ipfs/$NEW_ROOT_CID >> ~/ipfs_cron.log 2>&1
        "

        # If there is trouble fetching the nodes try to connect to known hosts
        ipfs swarm connect /p2p/12D3KooWPyQK2JEXnqK1QxiV9Y7bG3UsUQC5iQvDxn8bV1uqvsbi
        ipfs swarm connect /p2p/12D3KooWCFcfiBevjQD42aRAELMUZXAGScRiN2NcAthokF4dMnVU

        '''
    )
    # Note:
    # list named pins ipfs pin ls --type="recursive" --names
    # TODO: is there another pinning service that wont flake on us?
    # # Add pin to web3 remote storage
    # ipfs pin remote add --service=web3.storage.erotemic --name=shitspotter-dvc-$DATE $NEW_ROOT_CID --background
    # # Query status of remote pin
    # ipfs pin remote ls --service=web3.storage.erotemic --cid=$NEW_ROOT_CID --status=queued,pinning,pinned,failed
    # "
    print(ub.highlight_code(command, 'bash'))

    # dpath = ub.Path(shitspotter.__file__).parent
    # cid_revisions_fpath = dpath / 'cid_revisions.txt'
    # command = ub.codeblock(
    #     '''

    #     OLD EXAMPLES:
    #     echo "QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
    #     echo "QmaPPoPs7wXXkBgJeffVm49rd63ZtZw5GrhvQQbYrUbrYL" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
    #     echo "QmaSfRtzXDCiqyfmZuH6NEy2HBr7radiJNhmSjiETihoh6" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
    #     echo "QmPptXKFKi6oTJL3VeCNy5Apk8MJsHhCAAwVmegHhuRY83" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
    #     echo "QmfStoay5rjeHMEDiyuGsreXNHsyiS5kVaexSM2fov216j" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
    #     echo "bafybeihgex2fj4ethxnfmiivasw5wqsbt2pdjswu3yr554rewm6myrkq4a" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
    #     echo "bafybeihltrtb4xncqvfbipdwnlxsrxmeb4df7xmoqpjatg7jxrl3lqqk6y" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
    #     echo "bafybeihi7v7sgnxb2y57ie2dr7oobigsn5fqiwxwq56sdpmzo5on7a2xwe" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt
    #     echo "bafybeiedk6bu2qpl4snlu3jmtri4b2sf476tgj5kdg2ztxtm7bd6ftzqyy" >> /home/joncrall/code/shitspotter/shitspotter/cid_revisions.txt

    #     Then on mojo:

    #     # Note: define NEW_ROOT_CID=
    #     DATE=$(date +"%Y-%m-%d")

    #     ipfs pin add --progress "${NEW_ROOT_CID}"

    #     # Also, we should pin the CID on a pinning service
    #     ipfs pin remote add --service=web3.storage.erotemic --name="shitspotter-dvc-$DATE" "${NEW_ROOT_CID}" --background
    #     ipfs pin remote ls --service=web3.storage.erotemic --cid="${NEW_ROOT_CID}" --status=queued,pinning,pinned,failed

    #     # e.g.
    #     ipfs pin add bafybeihgex2fj4ethxnfmiivasw5wqsbt2pdjswu3yr554rewm6myrkq4a --progress
    #     ipfs pin add QmfStoay5rjeHMEDiyuGsreXNHsyiS5kVaexSM2fov216j --progress
    #     ipfs pin remote add --service=web3.storage.erotemic --name=shitspotter-dvc-2022-06-08 bafybeihgex2fj4ethxnfmiivasw5wqsbt2pdjswu3yr554rewm6myrkq4a --background
    #     ipfs pin remote ls --service=web3.storage.erotemic --cid=QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq --status=queued,pinning,pinned,failed

    #     Also see: ~/code/shitspotter/dev/sync_shit.sh
    #     '''
    # )
    # print(command)

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

    TODO:
        Currently I run this in IPython, it needs an entry point.

    Ignore:
        from shitspotter.phone_manager import *
    """
    import shitspotter
    import kwcoco
    phone = GVFSAndroidPath.discover()[0]
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

    DO_WE_WANT_SCRUBS = False
    if not DO_WE_WANT_SCRUBS:
        aliases = {k.replace('.scrubbed', ''): v for k, v in dvc_name_to_fpath.items() if '.scrubbed' in k}
        dvc_name_to_fpath.update(aliases)

    common = set(dvc_name_to_fpath) & set(phone_name_to_fpath)

    print(f'{len(common)=}')
    print(f'{len(dvc_name_to_fpath)=}')
    print(f'{len(phone_name_to_fpath)=}')

    to_delete = []
    num_bytes = 0
    for key in ub.ProgIter(common, desc='checking files are probably the same'):
        phone_fpath = phone_name_to_fpath[key]
        dvc_fpath = dvc_name_to_fpath[key]
        # Minimal checking that these the same file
        dvc_stat = dvc_fpath.stat()
        phone_stat = phone_fpath.stat()
        if 'scrubbed' in dvc_fpath.name:
            assert 0 <= (phone_stat.st_size - dvc_stat.st_size) <= 400
        else:
            assert phone_stat.st_size == dvc_stat.st_size
        to_delete.append({
            'path': phone_fpath,
            'st_size': phone_stat.st_size,
        })
        num_bytes += phone_stat.st_size

    to_delete = sorted(to_delete, key=lambda d: d['st_size'])

    from rich.prompt import Confirm
    import pint
    delete_size = (num_bytes * pint.Unit('byte')).to('gigabyte')
    print('to_delete = {}'.format(ub.urepr(to_delete, nl=1)))
    print(len(to_delete))
    print(f'delete_size = {ub.urepr(delete_size, nl=1)}')

    ans = Confirm.ask('Ready to delete?')
    if ans:
        for row in ub.ProgIter(to_delete, desc='deleting file'):
            row['path'].unlink(missing_ok=True)

        # ub.hash_file(dvc_fpath)
        # ub.hash_file(phone_fpath)


if __name__ == '__main__':
    """
    CommandLine:
        python -m shitspotter.phone_manager
    """
    RemoteFromPhoneConfig.main()
