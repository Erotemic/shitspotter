#!/usr/bin/env python3
"""
A scriptconfig powered CLI for IPFS.
"""
import scriptconfig as scfg
import ubelt as ub


class IPFS(scfg.ModalCLI):
    ...


@IPFS.register
class IPFSAdd(scfg.DataConfig):
    """
    Add a folder to IPFS
    """
    __command__ = 'add'
    # auth = scfg.Value('transmission:transmission', help='auth argument')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        config = cls.cli(cmdline=cmdline, data=kwargs)
        # This command may need to be modified
        ub.cmd(f'..', system=True)
