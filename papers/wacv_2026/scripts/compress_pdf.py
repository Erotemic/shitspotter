#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ubelt",
#   "scriptconfig",
#   "rich",
# ]
# ///
import scriptconfig as scfg
import ubelt as ub


class CompressPdfCLI(scfg.DataConfig):
    pdf_fpath = scfg.Value(None, help='pdf_fpath', position=1)
    start_page = scfg.Value(0, help='The index of the first page to start from (zero indexed inclusive)')
    stop_page = scfg.Value(None, help='The index of the last page to start from (zero indexed exclusive)')
    quality = scfg.Value('default', help=(
        'Compression quality level. '
        'Choices: "screen", "ebook", "printer", "prepress", "default". '
        '"screen" is highest compression, "prepress" is least.'),
        choices=['screen', 'ebook', 'printer', 'prepress', 'default'])

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

    # Apply compression quality
    quality_map = {
        'screen': '/screen',
        'ebook': '/ebook',
        'printer': '/printer',
        'prepress': '/prepress',
        'default': '/default',
    }
    gs_options += [f'-dPDFSETTINGS={quality_map[config.quality]}']

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
        python ~/code/shitspotter/papers/wacv_2026/scripts/compress_pdf.py main_part1.pdf --quality=printer
        python -m compress_pdf
    """
    __cli__.main()
