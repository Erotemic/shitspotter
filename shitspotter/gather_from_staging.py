def gather_from_staging():
    """
    Given a path to raw staged data, scrub and copy the data into the DVC /
    version controlled data repo.
    """
    import ubelt as ub
    staging_dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_staging')
    data_dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_dvc')

    from shitspotter.gather import _new_generic_gather_image_rows
    from shitspotter.gather import _new_generic_image_gdf_expand

    image_dpath = staging_dpath / 'assets/poop-2024-12-30-T212347/'

    image_rows = _new_generic_gather_image_rows(image_dpath)
    image_gdf = _new_generic_image_gdf_expand(image_rows)
    copy_to_repo(image_gdf)

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
