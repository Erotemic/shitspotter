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
class IPFSAdd(scfg.DataConfig):
    """
    Add a folder to IPFS

    Modifies defaults and extends some functionality of the regular kubo API

    Extensions:
        * Writes a dvc-like sidecar file
        * Can specify a name for the initial pin in the add command

    Example:
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

            num_items = len(lines)  # number of new sub-cids
            if sidecar_fpath is not None:
                sidecar_metadata = {
                    'cid': cid,
                    'rel_path': os.fspath(path.relative_to(sidecar_fpath.parent)),
                    'size': size_str,
                    'num_items': num_items,
                    'add_config': dict(config),
                    'add_datetime': ub.timestamp(),
                    'add_duration': timer.elapsed,
                }
                sidecar_metadata = kwutil.Json.ensure_serializable(sidecar_metadata)
                sidecar_text = kwutil.Yaml.dumps(sidecar_metadata)
                sidecar_fpath.write_text(sidecar_text)

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


if __name__ == '__main__':
    IPFS.main()
