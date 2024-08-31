#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CompressPdfCLI(scfg.DataConfig):
    pdf_fpath = scfg.Value(None, help='pdf_fpath', position=1)

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        compress_pdf(config.pdf_fpath)

__cli__ = CompressPdfCLI


def find_ghostscript_exe():
    if ub.WIN32:
        gs_exe = r'C:\Program Files (x86)\gs\gs9.16\bin\gswin32c.exe'
    else:
        gs_exe = 'gs'
    return gs_exe


def compress_pdf(pdf_fpath):
    """ uses ghostscript to write a pdf """
    suffix = '_' + ub.timestamp() + '_compressed'
    output_pdf_fpath = ub.Path(pdf_fpath).augment(stemsuffix=suffix)
    print(f'output_pdf_fpath={output_pdf_fpath}')
    gs_exe = ub.find_exe('gs')
    cmd_list = (
        gs_exe,
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        '-sOutputFile=' + output_pdf_fpath,
        pdf_fpath
    )
    ub.cmd(cmd_list)
    return output_pdf_fpath

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/papers/application-2024/compress_pdf.py
        python -m compress_pdf
    """
    __cli__.main()
