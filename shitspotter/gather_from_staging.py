#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class GatherFromStagingCLI(scfg.DataConfig):
    staging_dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_staging')
    shitspotter_dvc_dpath = scfg.Value(None)  # ub.Path('/home/joncrall/data/dvc-repos/shitspotter_dvc')
    staging_shit_dpath = scfg.Value(None)  # staging_dpath / 'assets/poop-2024-12-30-T212347/'

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Ignore:
            kwargs = dict(
                staging_dpath='/data/joncrall/dvc-repos/shitspotter_staging',
                shitspotter_dvc_dpath='/home/joncrall/data/dvc-repos/shitspotter_dvc',
                staging_shit_dpath='/data/joncrall/dvc-repos/shitspotter_staging/assets/poop-2025-04-20-T172113',
            )
            from shitspotter.gather_from_staging import *  # NOQA
            argv = 0
            cls = GatherFromStagingCLI
            config = cls(**kwargs)

        Example:
            >>> # xdoctest: +SKIP
            >>> from shitspotter.gather_from_staging import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict()
            >>> cls = GatherFromStagingCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        staging_shit_dpath = config.staging_shit_dpath
        staging_dpath = config.staging_dpath
        shitspotter_dvc_dpath = config.shitspotter_dvc_dpath
        gather_from_staging(staging_shit_dpath, staging_dpath, shitspotter_dvc_dpath)

__cli__ = GatherFromStagingCLI


def gather_from_staging(staging_shit_dpath, staging_dpath, shitspotter_dvc_dpath):
    """
    Given a path to raw staged data, scrub and copy the data into the DVC /
    version controlled data repo.

    Ignore:
        new_shit_dpath
        data_dpath
        staging_dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_staging')
        shitspotter_dvc_dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_dvc')
        staging_shit_dpath = staging_dpath / 'assets/poop-2024-12-30-T212347/'
    """
    from shitspotter.gather import _new_generic_gather_image_rows
    from shitspotter.gather import _new_generic_image_gdf_expand
    data_dpath = shitspotter_dvc_dpath
    image_rows = _new_generic_gather_image_rows(staging_shit_dpath)
    image_gdf = _new_generic_image_gdf_expand(image_rows)
    copy_to_repo(image_gdf, staging_dpath, data_dpath)
    # Now at this point, we can run gather like normal


def copy_to_repo(image_gdf, staging_dpath, data_dpath):
    import ubelt as ub
    HANDLE_PRIVACY = True
    if HANDLE_PRIVACY:
        """
        sudo apt-get install xdelta3
        """
        # Strip out privacy relevant data. Depending on the privacy policy,
        # either discard it entirely or save it into a secure encrypted form.
        # (in the latter case this allows the metadata to be released later
        # after it is no longer sensitive)
        from shitspotter.util.util_data import find_secret_dpath
        from shitspotter.util.util_data import is_probably_encrypted
        secret_dpath = find_secret_dpath()
        privacy_rules_fpath = secret_dpath / 'privacy_rules.py'
        if is_probably_encrypted(privacy_rules_fpath):
            raise EnvironmentError('The privacy rules file is still encrypted')
        privacy_rules = ub.import_module_from_path(privacy_rules_fpath)
        image_gdf = privacy_rules.apply_privacy_rules(image_gdf, staging_dpath, data_dpath)
    else:
        raise NotImplementedError(
            'todo: without privacy rules, we just copy from staging to the repo')


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/shitspotter/gather_from_staging.py
        python -m shitspotter.gather_from_staging \
            --staging_dpath '/data/joncrall/dvc-repos/shitspotter_staging' \
            --shitspotter_dvc_dpath '/home/joncrall/data/dvc-repos/shitspotter_dvc' \
            --staging_shit_dpath '/data/joncrall/dvc-repos/shitspotter_staging/assets/poop-2025-04-20-T172113' \
    """
    __cli__.main()
