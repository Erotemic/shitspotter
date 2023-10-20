import fsspec

root_cid = "bafybeigovcysmghsyab6ia3raycsebbc32kea2k4qoxcsujmp52hzpsghy"

fs_cls = fsspec.get_filesystem_class('ipfs')
fs = fs_cls(asynchrounous=False)

items = fs.ls(root_cid)

assets_cid = None
for item in items:
    if item['name'].endswith('/assets'):
        assets_cid = item['CID']

asset_subitems = fs.ls(assets_cid)

asset_subitems = sorted(asset_subitems, key=lambda x: x['name'])

local_relpath = 'shitspotter_dvc/assets/'

lines = []

for item in asset_subitems:
    cid = item['CID']
    relpath = local_relpath + item['name'].split('/')[-1]
    lines.append(f'{cid} - {relpath}')

lines.append('')
lines.append(f'{assets_cid} - shitspotter_dvc/assets')
lines.append(f'{root_cid} - shitspotter_dvc')

import ubelt as ub
text = ub.indent('\n'.join(lines))
print(text)
