#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CompressPdfCLI(scfg.DataConfig):
    pdf_fpath = scfg.Value(None, help='pdf_fpath', position=1)
    start_page = scfg.Value(0, help='The index of the first page to start from (zero indexed inclusive)')
    stop_page = scfg.Value(None, help='The index of the last page to start from (zero indexed exclusive)')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        compress_pdf(config)

__cli__ = CompressPdfCLI


def find_ghostscript_exe():
    if ub.WIN32:
        gs_exe = r'C:\Program Files (x86)\gs\gs9.16\bin\gswin32c.exe'
    else:
        gs_exe = 'gs'
    return gs_exe


def compress_pdf(config):
    """ uses ghostscript to write a pdf """
    suffix = '_' + ub.timestamp() + '_compressed'
    pdf_fpath = ub.Path(config.pdf_fpath)
    output_pdf_fpath = pdf_fpath.augment(stemsuffix=suffix)
    print(f'output_pdf_fpath={output_pdf_fpath}')
    gs_exe = ub.find_exe('gs')
    gs_options = [
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
    ]
    if config.start_page != 0:
        gs_options += [f'-dFirstPage={config.start_page + 1}']

    if config.stop_page is not None:
        gs_options += [f'-dLastPage={config.stop_page}']

    import os
    cmd_list = (
        gs_exe,
        *gs_options,
        '-sOutputFile=' + output_pdf_fpath,
        os.fspath(pdf_fpath)
    )
    ub.cmd(cmd_list)
    return output_pdf_fpath

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/papers/neurips-2025/scripts/compress_pdf.py ~/code/shitspotter/papers/neurips-2025/main.pdf
        python -m compress_pdf
    """
    __cli__.main()
