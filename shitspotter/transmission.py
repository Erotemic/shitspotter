#!/usr/bin/env python3
"""
A scriptconfig powered CLI for transmission.
"""
import scriptconfig as scfg
import ubelt as ub


class TransmissionModal(scfg.ModalCLI):
    ...


@TransmissionModal.register
class EnsureDaemon(scfg.ModalCLI):
    """
    Interact with the transmission daemon
    """
    __command__ = 'daemon'

    class start(scfg.DataConfig):
        """
        Use systemctl to start the transmission-daemon.service
        """
        @classmethod
        def main(cls, argv=True, **kwargs):
            ub.cmd('sudo systemctl start transmission-daemon.service', system=True)

    class status(scfg.DataConfig):
        """
        Query systemctl for the status of transmission-daemon.service
        """
        @classmethod
        def main(cls, argv=True, **kwargs):
            ub.cmd('systemctl status transmission-daemon.service --no-pager', system=True)


@TransmissionModal.register
class TransmissionList(scfg.DataConfig):
    """
    Lookup the id of a torrent by its name.
    """
    __command__ = 'list'
    auth = scfg.Value('transmission:transmission', help='auth argument')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        # This command may need to be modified
        ub.cmd(f'transmission-remote --auth {config.auth} --list', system=True)


@TransmissionModal.register
class TransmissionStart(scfg.DataConfig):
    """
    Start (unpause) a torrent.
    """
    __command__ = 'start'
    identifier = scfg.Value(None, position=1, help='name, hash, or id of the torrent')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(config.identifier, config.auth, verbose=config.verbose)
        if torrent_id is None:
            print('error')
            return 1
        else:
            out = ub.cmd(f'transmission-remote --auth {config.auth} --torrent {torrent_id} --start',
                         verbose=max(1, config.verbose))
            return out.returncode


@TransmissionModal.register
class TransmissionStop(scfg.DataConfig):
    """
    Stop (pause) a torrent.
    """
    __command__ = 'stop'
    identifier = scfg.Value(None, position=1, help='name, hash, or id of the torrent')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(config.identifier, config.auth, verbose=config.verbose)
        if torrent_id is None:
            print('error')
            return 1
        else:
            out = ub.cmd(f'transmission-remote --auth {config.auth} --torrent {torrent_id} --stop',
                         verbose=max(1, config.verbose))
            return out.returncode


@TransmissionModal.register
class TransmissionInfo(scfg.DataConfig):
    """
    Show information about a specific torrent.
    """
    __command__ = 'info'
    identifier = scfg.Value(None, position=1, help='name, hash, or id of the torrent')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(config.identifier, config.auth, verbose=config.verbose)
        if torrent_id is None:
            print('error')
            return 1
        else:
            out = ub.cmd(f'transmission-remote --auth {config.auth} --torrent {torrent_id} --info', verbose=max(1, config.verbose))
            return out.returncode


@TransmissionModal.register
class TransmissionAddTracker(scfg.DataConfig):
    """
    Add a tracker to an existing torrent.
    """
    __command__ = 'add_tracker'
    identifier = scfg.Value(None, position=1, help='name, hash, or id of the torrent')
    tracker_url = scfg.Value(None, position=2, help='url of the tracker to add')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(config.identifier, config.auth, verbose=config.verbose)
        if torrent_id is None:
            print('error')
            return 1
        else:
            out = ub.cmd(f'transmission-remote --auth {config.auth} --torrent {torrent_id} --tracker-add "{config.tracker_url}"',
                         verbose=max(1, config.verbose))
            return out.returncode


@TransmissionModal.register
class TransmissionReannounce(scfg.DataConfig):
    """
    Show information about a specific torrent.
    """
    __command__ = 'reannounce'
    identifier = scfg.Value(None, position=1, help='name, hash, or id of the torrent')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(config.identifier, config.auth, verbose=config.verbose)
        if torrent_id is None:
            print('error')
            return 1
        else:
            out = ub.cmd(f'transmission-remote --auth {config.auth} --torrent {torrent_id} --reannounce', verbose=max(1, config.verbose))
            return out.returncode


@TransmissionModal.register
class TransmissionFind(scfg.DataConfig):
    """
    Tell Transmission where to look for the current torrents' data.
    """
    __command__ = 'find'
    identifier = scfg.Value(None, position=1, help='name, hash, or id of the torrent')
    dpath = scfg.Value(None, position=2, help='path to look for the data')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(config.identifier, config.auth, verbose=config.verbose)
        if torrent_id is None:
            print('error')
            return 1
        else:
            out = ub.cmd(f'transmission-remote --auth {config.auth} --torrent {torrent_id} --find "{config.dpath}"',
                         verbose=max(1, config.verbose))
            return out.returncode


@TransmissionModal.register
class TransmissionMove(scfg.DataConfig):
    """
    Tell Transmission where to look for the current torrents' data.
    """
    __command__ = 'move'
    identifier = scfg.Value(None, position=1, help='name, hash, or id of the torrent')
    dpath = scfg.Value(None, position=2, help='path to move the data to')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(config.identifier, config.auth, verbose=config.verbose)
        if torrent_id is None:
            print('error')
            return 1
        else:
            out = ub.cmd(f'transmission-remote --auth {config.auth} --torrent {torrent_id} --move "{config.dpath}"',
                         verbose=max(1, config.verbose))
            return out.returncode


@TransmissionModal.register
class TransmissionVerify(scfg.DataConfig):
    """
    Verify the selected torrent.
    """
    __command__ = 'verify'
    identifier = scfg.Value(None, position=1, help='name or id of the torrent')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(config.identifier, config.auth, verbose=config.verbose)
        if torrent_id is None:
            print('error')
            return 1
        else:
            out = ub.cmd(f'transmission-remote --auth {config.auth} --torrent {torrent_id} --verify',
                         verbose=max(1, config.verbose))
            return out.returncode


@TransmissionModal.register
class TransmissionFiles(scfg.DataConfig):
    """
    Get a file list for the current torrent.
    """
    __command__ = 'files'
    identifier = scfg.Value(None, position=1, help='name or id of the torrent')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(config.identifier, config.auth, verbose=config.verbose)
        if torrent_id is None:
            print('error')
            return 1
        else:
            out = ub.cmd(f'transmission-remote --auth {config.auth} --torrent {torrent_id} --files',
                         verbose=max(1, config.verbose))
            return out.returncode


@TransmissionModal.register
class TransmissionLookupID(scfg.DataConfig):
    """
    Lookup the id of a torrent by its name for use with the raw
    transmission-remote tool.
    """
    __command__ = 'lookup_id'
    identifier = scfg.Value(None, position=1, help='name of the torrent')
    auth = scfg.Value('transmission:transmission', help='auth argument')
    verbose = scfg.Value(0, isflag=True, help='verbosity')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        torrent_id = lookup_torrent_id(**config)
        if torrent_id is None:
            print('error')
            return 1
        else:
            print(torrent_id)
            return 0


def lookup_torrent_id(identifier, auth, verbose=0):
    import re
    # If given as an integer, assume they are using the right id
    try:
        torrent_id = int(identifier)
    except Exception:
        ...
    else:
        return torrent_id

    # This command may need to be modified
    out = ub.cmd(
        f'transmission-remote --auth {auth} --list',
        shell=True, verbose=verbose, check=True)
    splitpat = re.compile('   *')
    for line in out.stdout.split(chr(10)):
        line_ = line.strip()
        if not line_ or line_.startswith(('Sum:', 'ID')):
            continue
        row_vals = splitpat.split(line_)
        name = row_vals[-1]
        torrent_id = row_vals[0].strip('*')
        if name == identifier:
            return torrent_id


if __name__ == '__main__':
    """
    CommandLine:
        python -m shitspotter.transmission --help
        python -m shitspotter.transmission ensure_deamon
        python -m shitspotter.transmission list
        python -m shitspotter.transmission lookup_id coco2014
    """
    TransmissionModal.main()
