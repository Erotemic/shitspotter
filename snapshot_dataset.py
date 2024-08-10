"""
Creates data suitable for uploading to data.kitware.com

https://data.kitware.com/?#user/598a19658d777f7d33e9c18b/folder/65d6c52fb40ab0fa6c57909b
"""

import zipfile
import ubelt as ub


def main():
    import shitspotter
    dataset_dpath = shitspotter.util.find_shit_coco_fpath().parent
    snapshot_dpath = dataset_dpath.parent / f'shitspotter-snapshot-{ub.timestamp()}'

    snapshot_dpath.ensuredir()

    coco_paths = list(dataset_dpath.glob('*.kwcoco.zip')) + list(dataset_dpath.glob('*.kwcoco.json'))
    asset_dpaths = list((dataset_dpath / 'assets').glob('*'))
    analysis_dpath = (dataset_dpath / 'analysis')

    tozip_dpaths = asset_dpaths + [analysis_dpath]

    import kwutil
    pman = kwutil.ProgressManager()

    with pman:
        for src_dpath in pman.ProgIter(tozip_dpaths, desc='zipping assets'):
            rel_dpath = src_dpath.relative_to(dataset_dpath)
            dst_fpath = ub.Path(snapshot_dpath / rel_dpath + '.zip')

            parent_dpath = src_dpath.parent
            to_write = []
            for root, ds, fs in src_dpath.walk():
                rel_root = root.relative_to(parent_dpath)
                for f in fs:
                    fpath = root / f
                    rel_fpath = rel_root / f
                    write_args = (fpath, str(rel_fpath))
                    to_write.append(write_args)

            dst_fpath.parent.ensuredir()
            zfile = zipfile.ZipFile(dst_fpath, compression=zipfile.ZIP_DEFLATED, mode='w')
            with zfile:
                for write_args in pman.ProgIter(to_write, desc=f'zipping: {src_dpath}'):
                    # print(f'write_args={write_args}')
                    zfile.write(*write_args)

        splits_zip = snapshot_dpath / 'kwcoco_splits.zip'
        zfile = zipfile.ZipFile(splits_zip, compression=zipfile.ZIP_DEFLATED, mode='w')
        with zfile:
            for coco_fpath in coco_paths:
                zfile.write(coco_fpath, arcname=coco_fpath.name)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/snapshot_dataset.py
    """
    main()
