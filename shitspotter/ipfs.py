#!/usr/bin/env python3
"""
A scriptconfig powered wrapper CLI for IPFS.

Ignore:
    python -m shitspotter.ipfs --help
"""
import os
import scriptconfig as scfg
import rich
import kwutil
import ubelt as ub


class IPFS(scfg.ModalCLI):
    ...


@IPFS.register
class IPFSPin(scfg.ModalCLI):
    """
    Extension of IPFS pin commands with dvc-like sidecar support
    """
    __command__ = 'pin'

    class add(scfg.DataConfig):
        """
        Ignore:
            # xdoctest: +REQUIRES(env:IPFS_TEST)
            from shitspotter.ipfs import *  # NOQA
            import ubelt as ub
            root_dpath = ub.Path.appdir('tests/ipfs').ensuredir()
            dpath = (root_dpath / 'dir1').ensuredir()
            fpath = (dpath / 'fpath1.txt')
            fpath.write_text('data')
            cls = IPFSAdd
            argv = 0
            kwargs = config = cls(path=dpath, name='scfg-ipfs-test-pin', only_hash=0)
            ipfs_add_argv = config._build_add_command()
            IPFSAdd.main(argv=argv, **config)
            ipfs_sidecar_fpath = dpath.augment(tail='.ipfs')
            kwutil.Yaml.coerce(ipfs_sidecar_fpath)['cid']
            IPFSPin.add.main(argv=argv, path=ipfs_sidecar_fpath, dry_run=1)
        """
        path = scfg.Value(None, help='path to a tracked .ipfs file or ipfs-object', position=1)
        recursive = scfg.Flag(True, help='Recursively pin the object linked to by the specified object(s). Default: true.')
        progress = scfg.Flag(True, short_alias=['p'], help='Stream progress data.')
        name = scfg.Value(None, help='An optional name for created pin(s).')
        dry_run = scfg.Flag(False, short_alias=['n'], help='Generate the command, but dont do work')

        @classmethod
        def main(cls, argv=1, **kwargs):
            argv = kwargs.pop('cmdline', argv)  # helper for cmdline->argv transition
            config = cls.cli(argv=argv, data=kwargs, strict=True)
            rich.print('config = ' + ub.urepr(config, nl=1))

            if config.path is None:
                raise Exception('Path must be specified')

            ipfs_sidecar_fpath = ub.Path(config.path)
            if ipfs_sidecar_fpath.exists():
                sidecar_metadata = kwutil.Yaml.load(ipfs_sidecar_fpath)
                root_cid = sidecar_metadata['cid']
                if config.name is None:
                    config.name = sidecar_metadata.get('add_config', {}).get('name', None)
            else:
                root_cid = config.path

            pin_argv = ['ipfs', 'pin', 'add']
            if config.name is not None:
                pin_argv += ['--name', config.name]
            if config.progress:
                pin_argv += ['--progress']
            if config.recursive:
                pin_argv += ['--recursive']
            pin_argv += [root_cid]

            if config.dry_run:
                print(argv_to_str(pin_argv))
            else:
                info = ub.cmd(pin_argv, verbose=3)
                info.check_returncode()


def argv_to_str(argv):
    import shlex
    command_parts = []
    # Allow the user to specify paths as part of the command
    for part in argv:
        if isinstance(part, os.PathLike):
            part = os.fspath(part)
        command_parts.append(part)
    command_tup = list(command_parts)
    command_text = ' '.join(list(map(shlex.quote, command_tup)))
    return command_text


@IPFS.register
class IPFSPull(scfg.DataConfig):
    """
    This works with the DVC-like sidecars we create with IPFS add.

    Add a folder to IPFS

    Modifies defaults and extends some functionality of the regular kubo API

    Extensions:
        * Writes a dvc-like sidecar file
        * Can specify a name for the initial pin in the add command

    Example:
        >>> # xdoctest: +REQUIRES(env:IPFS_TEST)
        >>> from shitspotter.ipfs import *  # NOQA
        >>> import ubelt as ub
        >>> root_dpath = ub.Path.appdir('tests/ipfs').ensuredir()
        >>> dpath = (root_dpath / 'dir1').ensuredir()
        >>> fpath = (dpath / 'fpath1.txt')
        >>> fpath.write_text('data')
        >>> cls = IPFSAdd
        >>> argv = 0
        >>> kwargs = config = cls(path=dpath, name='scfg-ipfs-test-pin', only_hash=0)
        >>> ipfs_add_argv = config._build_add_command()
        >>> IPFSAdd.main(argv=argv, **config)
        >>> ipfs_sidecar_fpath = dpath.augment(tail='.ipfs')
        >>> kwargs = dict(path=ipfs_sidecar_fpath, dry_run=1)
        >>> IPFSPull.main(argv=argv, **kwargs)
        >>> kwargs = dict(path=ipfs_sidecar_fpath, dry_run=0)
        >>> IPFSPull.main(argv=argv, **kwargs)
    """
    __command__ = 'pull'
    path = scfg.Value(None, help='path to a tracked .ipfs file', position=1)
    dry_run = scfg.Flag(False, short_alias=['n'], help='Only inspect what would be done and do basic error checking. No download / file modifiation')

    @classmethod
    def main(cls, argv=1, **kwargs):
        argv = kwargs.pop('cmdline', argv)  # helper for cmdline->argv transition
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        if config.path is None:
            raise Exception('Path must be specified')

        ipfs_sidecar_fpath = ub.Path(config.path)
        assert ipfs_sidecar_fpath.exists(), '#todo: simpledvc-like flexibility'

        sidecar_metadata = kwutil.Yaml.load(ipfs_sidecar_fpath)
        root_cid = sidecar_metadata['cid']
        dpath = ipfs_sidecar_fpath.parent
        rel_path = sidecar_metadata['rel_path']

        if not config.dry_run:
            sync_ipfs_pull(root_cid, dpath, rel_path)
        else:
            print('sidecar_metadata = {}'.format(ub.urepr(sidecar_metadata, nl=1)))


@IPFS.register
class IPFSAdd(scfg.DataConfig):
    """
    Add a folder to IPFS

    Modifies defaults and extends some functionality of the regular kubo API

    Extensions:
        * Writes a dvc-like sidecar file
        * Can specify a name for the initial pin in the add command

    Example:
        >>> # xdoctest: +REQUIRES(env:IPFS_TEST)
        >>> from shitspotter.ipfs import *  # NOQA
        >>> import ubelt as ub
        >>> root_dpath = ub.Path.appdir('tests/ipfs').ensuredir()
        >>> dpath = (root_dpath / 'dir1').ensuredir()
        >>> fpath = (dpath / 'fpath1.txt')
        >>> fpath.write_text('data')
        >>> cls = IPFSAdd
        >>> argv = 0
        >>> kwargs = config = cls(path=dpath, name='scfg-ipfs-test-pin', only_hash=0)
        >>> rich.print('config = ' + ub.urepr(config, nl=1))
        >>> ipfs_add_argv = config._build_add_command()
        >>> print('ipfs_add_argv = {}'.format(ub.urepr(ipfs_add_argv, nl=1)))
        >>> IPFSAdd.main(**config)
    """
    __command__ = 'add'
    path = scfg.Value(None, help='file or directory to add to IPFS', position=1)
    name = scfg.Value(None, help='An optional name for created pin(s).')
    recursive = scfg.Flag(True, help='Add directory paths recursively.')
    progress = scfg.Flag(True, short_alias=['p'], help='Stream progress data.')
    cid_version = scfg.Value(1, help='CID version.')
    raw_leaves = scfg.Flag(False, help='Use raw blocks for leaf nodes.')
    only_hash = scfg.Flag(False, short_alias=['n'], help='Only chunk and hash - do not write to disk.')
    pin = scfg.Flag(True, help='Pin locally to protect added files from garbage collection. Default: true.')
    sidecar = scfg.Flag(True, help='if True, write dvc-like metadata to a sidecar file')

    def _build_add_command(config):
        import kwutil
        import os
        # variables that are given as existant flags
        existant_flag_vars = [
            'pin', 'progress', 'recursive', 'only_hash',
        ]
        kv_vars = [
            'raw_leaves', 'cid_version'
        ]
        ipfs_add_argv = ['ipfs', 'add']
        for k in existant_flag_vars:
            if config[k]:
                k2 = k.replace('_', '-')
                ipfs_add_argv.append(f'--{k2}')
        for k in kv_vars:
            k2 = k.replace('_', '-')
            v2 = kwutil.Json.dumps(config[k])
            ipfs_add_argv.append(f'--{k2}={v2}')
        ipfs_add_argv.append(os.fspath(config.path))
        return ipfs_add_argv

    @classmethod
    def main(cls, argv=1, **kwargs):
        argv = kwargs.pop('cmdline', argv)  # helper for cmdline->argv transition
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        if config.path is None:
            raise Exception('Path must be specified')

        path = ub.Path(config.path)

        if config.sidecar:
            sidecar_fpath = config.sidecar
            if sidecar_fpath is True:
                sidecar_fpath = path.augment(tail='.ipfs')
            sidecar_fpath = ub.Path(sidecar_fpath)
            if sidecar_fpath.exists():
                print('Overwriting existing sidecar!')
                if sidecar_fpath.is_dir():
                    raise Exception('sidecar name conflicts with a directory')
        else:
            sidecar_fpath = None

        ipfs_add_argv = config._build_add_command()

        with ub.Timer() as timer:
            info = ub.cmd(ipfs_add_argv, verbose=3)

        info.check_returncode()

        if not config.only_hash:
            # Only do postprocessing steps for non-dry runs
            lines = info.stdout.strip().split('\n')
            parts = lines[-1].split(' ')
            assert len(parts) > 1
            cid = parts[1]

            USE_PROGBAR_META = 1
            if USE_PROGBAR_META:
                # not sure how stable the format is for this.
                progbar_line = info.stderr.strip().split('\n')[-1]
                assert '=' in progbar_line
                size_str = progbar_line.split('[')[0].split('/')[1].strip()
            else:
                size_str = None

            # fixme: only do in a git repo:
            ADD_TO_GIT_IGNORE = True
            ADD_SIDECAR_TO_GIT = True

            num_items = len(lines)  # number of new sub-cids
            if sidecar_fpath is not None:
                sidecar_dpath = sidecar_fpath.parent
                rel_path = path.relative_to(sidecar_dpath)

                sidecar_metadata = {
                    'type': 'ipfs-sidecar',
                    'cid': cid,
                    'rel_path': os.fspath(rel_path),
                    'size': size_str,
                    'num_items': num_items,
                    'add_config': dict(config),
                    'add_datetime': ub.timestamp(),
                    'add_duration': timer.elapsed,
                }
                sidecar_metadata = kwutil.Json.ensure_serializable(sidecar_metadata)
                sidecar_text = kwutil.Yaml.dumps(sidecar_metadata)
                print(f'write to: sidecar_fpath={sidecar_fpath}')
                sidecar_fpath.write_text(sidecar_text)

                if ADD_TO_GIT_IGNORE:
                    ignore_fpath = sidecar_dpath / '.gitignore'
                    rel_path_line = os.fspath(rel_path)
                    if ignore_fpath.exists():
                        ignore_lines = ignore_fpath.read_text().strip().split('\n')
                        needs_write = rel_path_line not in (p.strip() for p in ignore_lines)
                    else:
                        needs_write = True
                    if needs_write:
                        print('Update gitignore')
                        with open(ignore_fpath, 'a') as file:
                            file.write(os.fspath(rel_path_line) + '\n')
                    else:
                        print('gitignore does not need update')

                if ADD_SIDECAR_TO_GIT:
                    ub.cmd(f'git add {sidecar_fpath.name}', cwd=sidecar_fpath.parent, verbose=3)

                # Also handle .gitignore

            if config.name:
                pin_argv = ['ipfs', 'pin', 'add']
                pin_argv += ['--name', config.name]
                if config.progress:
                    pin_argv += ['--progress']
                pin_argv += [cid]
                info = ub.cmd(pin_argv, verbose=3)
                info.check_returncode()

            if sidecar_fpath is not None:
                print(sidecar_text)
                print(f'Wrote to: sidecar_fpath={sidecar_fpath}')


def sync_ipfs_pull(root_cid, dpath, rel_path):
    """
    Ignore:
        root_cid = 'bafybeicbm6nljzd6jdhnx7iha6764ntcjarktfltytbeymrflwfi6zuldm'
        dpath = '/home/joncrall/.cache/tests/ipfs'
        rel_path = 'dir1'
    """
    import os
    # import ipfsspec
    # import logging
    # import sys
    import ubelt as ub
    # avail = sorted(fsspec.available_protocols())
    # print('avail = {}'.format(ub.urepr(avail, nl=1)))
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    import uuid
    uuid = uuid.uuid4()

    dpath = ub.Path(dpath)
    tmp_fname = f'tmp-{uuid}'
    tmp_path = dpath / tmp_fname
    out_path = dpath / rel_path

    impl_version = 1
    if impl_version == 1:
        # Very rough initial implemention, force grab of everything followed by an
        # rsync. Should try to do sync more incrementally. Could also check that
        # hash of a directory is unchanged.
        ub.cmd(['ipfs', 'get', '--progress=true', f'--output={tmp_path}', root_cid], verbose=3, check=True)
        if out_path.exists():
            # if the output already exists we have to sync it
            # References: https://stackoverflow.com/questions/20300971/rsync-copy-directory-contents-but-not-directory-itself
            # rsync interprets a directory with no trailing slash as copy this
            # directory, and a directory with a trailing slash as copy the contents
            # of this directory.
            ub.cmd(['rsync', '-avprP', str(tmp_path) + '/', out_path], verbose=3, check=True)
            tmp_path.delete()
        else:
            # in the case where the output doesnt exist yet we can be more efficent
            os.rename(tmp_path, out_path)
    elif impl_version == 2:
        # FIXME
        import fsspec
        fs_cls = fsspec.get_filesystem_class('ipfs')
        fs = fs_cls(asynchronous=False)
        results = fs.ls(root_cid)
        print('results = {}'.format(ub.urepr(results, nl=1)))

        to_ensure = []
        to_copy = []
        for ipfs_root, dnames, fnames in ub.ProgIter(fs.walk(root_cid), desc='walking'):
            ...
            rel_root = os.path.relpath(ipfs_root, root_cid)
            local_root = dpath / rel_root
            if not local_root.exists():
                to_ensure.append(local_root)

            for fname in fnames:
                ipfs_fpath = os.path.join(ipfs_root, fname)
                local_fpath = local_root / fname
                if not local_fpath.exists():
                    to_copy.append({
                        'src': ipfs_fpath,
                        'dst': local_fpath,
                        'overwrite': False,
                    })
                else:
                    # if 'kwcoco' not in str(local_fpath) and '_cache' not in str(local_fpath):
                    ipfs_stat = fs.stat(ipfs_fpath)
                    local_stat = local_fpath.stat()
                    # TODO: need better mechanism to determine if files are the same
                    probably_same = (ipfs_stat['size'] == local_stat.st_size)
                    if not probably_same:
                        to_copy.append({
                            'src': ipfs_fpath,
                            'dst': local_fpath,
                            'overwrite': True,
                        })

        # Execute Copy

        for dpath in to_ensure:
            dpath.ensuredir()

        for task in ub.ProgIter(to_copy, desc='copy files'):
            if task['overwrite']:
                src = task['src']
                dst = task['dst']
                fs.get(src, str(dst))
            else:
                src = task['src']
                dst = task['dst']
                bak = dst.augment(suffix='.old')
                dst.move(bak)
                fs.get(src, str(dst))
                bak.delete()


if __name__ == '__main__':
    IPFS.main()
