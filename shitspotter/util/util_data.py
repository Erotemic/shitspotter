
def find_shit_coco_fpath(on_error='raise'):
    """
    Return the location of the shitspotter kwcoco file if it exists and is in a
    "standard" location.

    This assumes

    Ignore:
        ln -s /data/store/data/shit-pics/ $HOME/data/dvc-repos/shitspotter_dvc
        ln -s /data/data/shit-pics/ $HOME/data/dvc-repos/shitspotter_dvc
    """
    import ubelt as ub
    import os
    _default = ub.expandpath('$HOME/data/dvc-repos/shitspotter_dvc')
    dvc_dpath = os.environ.get('SHITSPOTTER_DVC_DPATH', _default)
    dvc_dpath = ub.Path(dvc_dpath)
    coco_fpath = dvc_dpath / 'data.kwcoco.json'

    if not coco_fpath.exists():
        if on_error == 'raise':
            raise Exception
        else:
            return None
    return coco_fpath
