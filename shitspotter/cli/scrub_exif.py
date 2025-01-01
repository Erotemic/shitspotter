#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class ScrubExifCLI(scfg.DataConfig):
    in_path = scfg.Value(None, help='param1')
    out_path = scfg.Value()

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from shitspotter.cli.scrub_exif import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict()
            >>> cls = ScrubExifCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        config['in_path'] = '/home/joncrall/Downloads/Unknown Residue'
        config['out_path'] = '/home/joncrall/Downloads/Unknown Residue-scrub'

        import kwimage
        import kwutil
        exts = ['*' + e for e in kwimage.im_io.JPG_EXTENSIONS]
        src_paths = kwutil.util_path.coerce_patterned_paths(config['in_path'], expected_extension=exts)

        out_path = ub.Path(config['out_path'])
        if out_path.endswith('.jpg'):
            assert len(src_paths) == 1
            tasks = [{'src': src_paths[0], 'dst': out_path}]
        else:
            out_path.ensuredir()
            tasks = [{'src': p, 'dst': out_path / p.name} for p in src_paths]

        import kwutil
        from shitspotter.util import util_image
        for task in tasks:
            util_image.scrub_exif_metadata(task['src'], task['dst'])

__cli__ = ScrubExifCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/shitspotter/cli/scrub_exif.py
        python -m shitspotter.cli.scrub_exif
    """
    __cli__.main()
