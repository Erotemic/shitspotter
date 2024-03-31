import zipfile
import shitspotter
import ubelt as ub

dataset_dpath = shitspotter.util.find_shit_coco_fpath().parent
snapshot_dpath = dataset_dpath.parent / 'shitspotter-snapshot'

snapshot_dpath.ensuredir()

coco_paths = list(dataset_dpath.glob('*.kwcoco.zip')) + list(dataset_dpath.glob('*.kwcoco.json'))
asset_dpaths = list((dataset_dpath / 'assets').glob('*'))
analysis_dpath = (dataset_dpath / 'analysis')


tozip_dpaths = asset_dpaths + [analysis_dpath]

for src_dpath in tozip_dpaths:
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
        for write_args in to_write:
            print(f'write_args={write_args}')
            zfile.write(*write_args)


splits_zip = snapshot_dpath / 'kwcoco_splits.zip'
zfile = zipfile.ZipFile(splits_zip, compression=zipfile.ZIP_DEFLATED, mode='w')
with zfile:
    for coco_fpath in coco_paths:
        zfile.write(coco_fpath, arcname=coco_fpath.name)
