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


@IPFS.register
class export(scfg.DataConfig):
    """
    Scan for *.ipfs sidecars, extract CIDs (+ optional names), and print
    kubo commands you can run elsewhere.

    Examples:
        python -m shitspotter.ipfs pin export
        python -m shitspotter.ipfs pin export .
        python -m shitspotter.ipfs pin export data runs/**/assets
        python -m shitspotter.ipfs pin export "data/**/*.ipfs" --progress
        python -m shitspotter.ipfs pin export . --emit_bash > pin_all.sh
    """
    paths = scfg.Value([], position=1, nargs='*',
                       help='0+ paths (dirs / globs / .ipfs files). Default: "."')

    recurse = scfg.Flag(True, help='Recurse into directories when scanning')
    dedupe = scfg.Flag(True, help='Deduplicate by CID (keeps first encountered name)')
    sort = scfg.Flag(True, help='Sort output for stable scripts')

    # pin command knobs (kubo)
    name = scfg.Value(None, help='Override name for all pins (else use sidecar add_config.name if available)')
    prefer_sidecar_name = scfg.Flag(True, help='Use sidecar add_config.name when present')
    progress = scfg.Flag(False, short_alias=['p'], help='Include --progress in emitted pin commands')
    recursive = scfg.Flag(True, help='Include --recursive in emitted pin commands')

    emit_bash = scfg.Flag(False, help='Emit a small bash header')

    @classmethod
    def main(cls, argv=1, **kwargs):
        argv = kwargs.pop('cmdline', argv)
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        import glob

        paths = list(config.paths) if config.paths else ['.']

        # 1) collect sidecar files
        sidecar_fpaths = []
        for p in paths:
            p = ub.Path(p)
            p_str = os.fspath(p)

            is_glob = any(ch in p_str for ch in ['*', '?', '['])
            if is_glob:
                matches = [ub.Path(m) for m in glob.glob(p_str, recursive=True)]
                for m in matches:
                    if m.is_dir():
                        it = m.rglob('*.ipfs') if config.recurse else m.glob('*.ipfs')
                        sidecar_fpaths.extend(list(it))
                    elif m.is_file() and m.suffix == '.ipfs':
                        sidecar_fpaths.append(m)
            else:
                if p.is_dir():
                    it = p.rglob('*.ipfs') if config.recurse else p.glob('*.ipfs')
                    sidecar_fpaths.extend(list(it))
                elif p.is_file() and p.suffix == '.ipfs':
                    sidecar_fpaths.append(p)

        # normalize / filter
        sidecar_fpaths = [ub.Path(s) for s in sidecar_fpaths]
        sidecar_fpaths = [s for s in sidecar_fpaths if s.is_file() and s.suffix == '.ipfs']

        # 2) extract (cid, name)
        items = []
        for fpath in sidecar_fpaths:
            meta = kwutil.Yaml.load(fpath)
            cid = meta.get('cid', None)
            if not cid:
                continue

            pin_name = None
            if config.name is not None:
                pin_name = config.name
            elif config.prefer_sidecar_name:
                pin_name = meta.get('add_config', {}).get('name', None)

            items.append((str(cid), pin_name, fpath))

        # 3) dedupe by CID
        if config.dedupe:
            seen = {}
            for cid, name, fpath in items:
                if cid not in seen:
                    seen[cid] = (cid, name, fpath)
            items = list(seen.values())

        # 4) stable order
        if config.sort:
            items = sorted(items, key=lambda t: (t[0], str(t[1] or ''), os.fspath(t[2])))

        # 5) emit header
        if config.emit_bash:
            print('#!/usr/bin/env bash')
            print('set -euo pipefail')
            print('')

        # 6) print kubo commands
        for cid, name, _fpath in items:
            pin_argv = ['ipfs', 'pin', 'add']
            if config.progress:
                pin_argv += ['--progress']
            if config.recursive:
                pin_argv += ['--recursive']
            pin_argv += [cid]
            if name:
                pin_argv += [f'--name={name}']
            print(argv_to_str(pin_argv))


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
        sidecar_fpaths = list(kwutil.util_path.coerce_patterned_paths(ipfs_sidecar_fpath, expected_extension='.ipfs'))
        print(f'sidecar_fpaths = {ub.urepr(sidecar_fpaths, nl=1)}')

        for ipfs_sidecar_fpath in sidecar_fpaths:
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
    __alias__ = 'snapshot'  # TODO: choose a better name than add

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
                # TODO: maybe default the name to the filename or foldername?
                # TODO: maybe make a wrapped version of the pin that includes
                # the .ipfs sidecar file?
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


@IPFS.register
class IPFSStatus(scfg.DataConfig):
    """
    Check whether the local content tracked by .ipfs sidecars appears changed.

    Default: quick heuristic (size + mtime).
    Optional: full verification (recompute CID via `ipfs add --only-hash`).

    Examples:
        python -m shitspotter.ipfs status .
        python -m shitspotter.ipfs status "data/**/*.ipfs"
        python -m shitspotter.ipfs status path/to/file.ipfs --full
        python -m shitspotter.ipfs status . --write_baseline
    """
    __command__ = 'status'

    path = scfg.Value('.', help='Path / glob / directory containing .ipfs sidecars', position=1)

    # scanning
    recursive = scfg.Flag(True, help='If path is a directory, recurse to find *.ipfs')
    strict = scfg.Flag(False, help='If True, error on missing tracked paths; else mark missing')

    # checking mode
    full = scfg.Flag(False, help='If True, recompute CID using `ipfs add --only-hash` and compare to sidecar CID')

    # heuristic baseline handling
    write_baseline = scfg.Flag(False, help='If True, write/update quick baseline stats into sidecar files')
    baseline_key = scfg.Value('local_quickstat', help='Sidecar key to store baseline quick stats under')

    @classmethod
    def main(cls, argv=1, **kwargs):
        argv = kwargs.pop('cmdline', argv)
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        ipfs_sidecar_fpath = ub.Path(config.path)
        sidecar_fpaths = _find_sidecars(ipfs_sidecar_fpath, recursive=config.recursive)

        rows = []
        for sidecar_fpath in sidecar_fpaths:
            meta = kwutil.Yaml.load(sidecar_fpath)
            if meta.get('type', None) != 'ipfs-sidecar':
                # ignore unknown YAMLs
                continue

            root_cid = meta.get('cid', None)
            rel_path = meta.get('rel_path', None)
            if root_cid is None or rel_path is None:
                continue

            tracked_path = (sidecar_fpath.parent / rel_path)

            # quick stat (current)
            cur_quick = _compute_quickstat(tracked_path)

            # baseline quick stat (from sidecar, if present)
            base_quick = meta.get(config.baseline_key, None)

            # heuristic change detection
            if cur_quick is None:
                status = 'MISSING'
                changed_quick = None
            else:
                if base_quick is None:
                    status = 'NO_BASELINE'
                    changed_quick = None
                else:
                    changed_quick = (
                        (cur_quick.get('bytes') != base_quick.get('bytes')) or
                        (cur_quick.get('mtime') != base_quick.get('mtime'))
                    )
                    status = 'CHANGED' if changed_quick else 'OK'

            # optional full check by recomputing CID
            full_ok = None
            new_cid = None
            if config.full and cur_quick is not None:
                try:
                    new_cid = _ipfs_only_hash_cid(tracked_path, add_config=meta.get('add_config', {}))
                    full_ok = (new_cid == root_cid)
                    # full check overrides the main status if we can compute it
                    status = 'OK' if full_ok else 'CHANGED'
                except Exception as ex:
                    status = 'FULL_CHECK_ERROR'
                    full_ok = False

            # optionally write baseline
            if config.write_baseline and cur_quick is not None:
                meta2 = dict(meta)
                meta2[config.baseline_key] = cur_quick
                sidecar_fpath.write_text(kwutil.Yaml.dumps(meta2))

            rows.append({
                'sidecar': os.fspath(sidecar_fpath),
                'tracked': os.fspath(tracked_path),
                'status': status,
                'cid': root_cid,
                'cid_recomputed': new_cid,
                'bytes': None if cur_quick is None else cur_quick.get('bytes'),
                'mtime': None if cur_quick is None else cur_quick.get('mtime'),
                'baseline_bytes': None if base_quick is None else base_quick.get('bytes'),
                'baseline_mtime': None if base_quick is None else base_quick.get('mtime'),
            })

            if config.strict and status == 'MISSING':
                raise FileNotFoundError(f'Missing tracked path for sidecar={sidecar_fpath} tracked={tracked_path}')

        _print_status_table(rows)


def _find_sidecars(path: ub.Path, recursive: bool = True):
    """
    Resolve to a list of .ipfs sidecar file paths.
    Supports:
        - a single .ipfs file
        - a directory (recursive search for *.ipfs)
        - a glob / patterned path via kwutil util_path helper
    """
    # If the user gives a single file, keep it.
    if path.exists() and path.is_file() and path.suffix == '.ipfs':
        return [path]

    # If they gave a directory, find all *.ipfs.
    if path.exists() and path.is_dir():
        return list(path.rglob('*.ipfs') if recursive else path.glob('*.ipfs'))

    # Otherwise treat as a patterned path (glob)
    # (this matches what you do in pull)
    candidates = list(kwutil.util_path.coerce_patterned_paths(path, expected_extension='.ipfs'))
    return [ub.Path(p) for p in candidates]


def _compute_quickstat(tracked_path: ub.Path):
    """
    Quick heuristic:
        - files: size + mtime
        - dirs: total bytes + max mtime (walk)
    Returns a JSON-serializable dict, or None if missing.
    """
    tracked_path = ub.Path(tracked_path)
    if not tracked_path.exists():
        return None

    if tracked_path.is_file():
        st = tracked_path.stat()
        return {
            'kind': 'file',
            'bytes': int(st.st_size),
            'mtime': float(st.st_mtime),
        }

    # directory
    total = 0
    max_mtime = 0.0
    nfiles = 0
    for f in tracked_path.rglob('*'):
        if f.is_file():
            st = f.stat()
            nfiles += 1
            total += int(st.st_size)
            if st.st_mtime > max_mtime:
                max_mtime = float(st.st_mtime)

    return {
        'kind': 'dir',
        'bytes': int(total),
        'mtime': float(max_mtime),
        'nfiles': int(nfiles),
    }


def _ipfs_only_hash_cid(tracked_path: ub.Path, add_config=None):
    """
    Full check: recompute the CID using kubo's `ipfs add --only-hash`.

    We try to honor relevant options from the original add_config if present.
    """
    add_config = add_config or {}
    tracked_path = ub.Path(tracked_path)

    argv = ['ipfs', 'add', '--only-hash']

    # directory recursion
    if tracked_path.is_dir():
        argv += ['--recursive']

    # mirror a few knobs if they exist in stored add_config
    # (ignore unknown keys; be conservative)
    if 'cid_version' in add_config:
        argv += [f'--cid-version={kwutil.Json.dumps(add_config["cid_version"])}']
    if 'raw_leaves' in add_config:
        # kubo expects flag presence; scriptconfig stored bool
        if add_config['raw_leaves']:
            argv += ['--raw-leaves=true']
        else:
            argv += ['--raw-leaves=false']

    argv += [os.fspath(tracked_path)]

    info = ub.cmd(argv, verbose=0)
    info.check_returncode()

    # parse last line: "<added> <cid> <path>"
    lines = [ln for ln in info.stdout.strip().split('\n') if ln.strip()]
    last = lines[-1]
    parts = last.split()
    if len(parts) < 2:
        raise RuntimeError(f'Unexpected ipfs output: {last}')
    cid = parts[1]
    return cid


def _print_status_table(rows):
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title='IPFS Sidecar Status', show_lines=False)

    table.add_column('status', no_wrap=True)
    table.add_column('sidecar', overflow='fold')
    table.add_column('tracked', overflow='fold')
    table.add_column('bytes', justify='right')
    table.add_column('mtime', justify='right')
    table.add_column('cid', overflow='fold')
    table.add_column('cid_recomputed', overflow='fold')

    for r in rows:
        table.add_row(
            str(r['status']),
            str(r['sidecar']),
            str(r['tracked']),
            '' if r['bytes'] is None else str(r['bytes']),
            '' if r['mtime'] is None else f"{r['mtime']:.3f}",
            str(r['cid'] or ''),
            str(r['cid_recomputed'] or ''),
        )

    console.print(table)


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
