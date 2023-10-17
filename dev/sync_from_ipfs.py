"""
Requirements:
    pip install ipfsspec
"""
import fsspec
import os
import shitspotter
import ipfsspec
# import logging
# import sys
import ubelt as ub

avail = sorted(fsspec.available_protocols())
print('avail = {}'.format(ub.urepr(avail, nl=1)))

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# fs = fsspec.get_filesystem_class('ipfs')
fs = ipfsspec.core.IPFSFileSystem(timeout=100)

cid = "bafybeief7tmoarwmd26b2petx7crtvdnz6ucccek5wpwxwdvfydanfukna"
res = fs._gw_apipost("ls", arg=cid)
links = res["Objects"][0]["Links"]

results = fs.ls(cid)

coco_fpath = shitspotter.util.find_shit_coco_fpath()

dpath = coco_fpath.parent

to_ensure = []
to_copy = []

for ipfs_root, dnames, fnames in ub.ProgIter(fs.walk(cid), desc='walking'):
    rel_root = os.path.relpath(ipfs_root, cid)
    local_root = dpath / rel_root
    if not local_root.exists():
        to_ensure.append(local_root)

    for fname in fnames:
        ipfs_fpath = os.path.join(ipfs_root, fname)
        local_fpath = local_root / fname
        if not local_fpath.exists():
            to_copy.append((local_fpath, ipfs_fpath))


for dpath in to_ensure:
    dpath.ensuredir()


for dst, src in ub.ProgIter(to_copy, desc='copy files'):
    fs.get(src, str(dst))
