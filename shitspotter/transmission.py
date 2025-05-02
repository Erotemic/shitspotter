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
        def main(cls, cmdline=1, **kwargs):
            ub.cmd('sudo systemctl start transmission-daemon.service', system=True)

    class status(scfg.DataConfig):
        """
        Query systemctl for the status of transmission-daemon.service
        """
        @classmethod
        def main(cls, cmdline=1, **kwargs):
            ub.cmd('systemctl status transmission-daemon.service --no-pager', system=True)


@TransmissionModal.register
class TransmissionList(scfg.DataConfig):
    """
    Lookup the id of a torrent by its name.
    """
    __command__ = 'list'
    auth = scfg.Value('transmission:transmission', help='auth argument')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        config = cls.cli(cmdline=cmdline, data=kwargs)
        # This command may need to be modified
        ub.cmd(f'transmission-remote --auth {config.auth} --list', system=True)


@TransmissionModal.register
class TransmissionLookupID(scfg.DataConfig):
    """
    Lookup the id of a torrent by its name.
    """
    __command__ = 'lookup_id'
    torrent_name = scfg.Value(None, position=1, help='name of the torrent')
    auth = scfg.Value('transmission:transmission', help='auth argument')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        config = cls.cli(cmdline=cmdline, data=kwargs)
        torrent_id = cls.lookup_torrent_id(**config)
        if torrent_id is None:
            print('error')
            return 1
        else:
            print(torrent_id)
            return 0

    @staticmethod
    def lookup_torrent_id(torrent_name, auth):
        import subprocess
        import sys
        import re
        # This command may need to be modified
        out = subprocess.check_output(
            f'transmission-remote --auth {auth} --list',
            shell=True, universal_newlines=True)
        splitpat = re.compile('   *')
        for line in out.split(chr(10)):
            line_ = line.strip()
            if not line_ or line_.startswith(('Sum:', 'ID')):
                continue
            row_vals = splitpat.split(line_)
            name = row_vals[-1]
            torrent_id = row_vals[0].strip('*')
            if name == sys.argv[1]:
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
