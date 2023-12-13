"""
Requirements:
    pip install ipfsspec

    # Requires: https://github.com/fsspec/ipfsspec/pull/27/files
"""


def sync_ipfs_pull():
    import os
    import shitspotter
    # import ipfsspec
    # import logging
    # import sys
    import ubelt as ub
    # avail = sorted(fsspec.available_protocols())
    # print('avail = {}'.format(ub.urepr(avail, nl=1)))
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    root_cid = "bafybeie275n5f4f64vodekmodnktbnigsvbxktffvy2xxkcfsqxlie4hrm"

    import fsspec
    fs_cls = fsspec.get_filesystem_class('ipfs')
    fs = fs_cls(asynchronous=False)
    results = fs.ls(root_cid)
    print('results = {}'.format(ub.urepr(results, nl=1)))

    coco_fpath = shitspotter.util.find_shit_coco_fpath()

    dpath = coco_fpath.parent

    to_ensure = []
    to_copy = []
    for ipfs_root, dnames, fnames in ub.ProgIter(fs.walk(root_cid), desc='walking'):
        ...
        rel_root = os.path.relpath(ipfs_root, root_cid)
        local_root = dpath / rel_root
        if not local_root.exists():
            to_ensure.append(local_root)

        for fname in fnames:
            ipfs_fpath = os.path.join(ipfs_root, fname)
            local_fpath = local_root / fname
            if not local_fpath.exists():
                to_copy.append({
                    'src': ipfs_fpath,
                    'dst': local_fpath,
                    'overwrite': False,
                })
            else:
                # if 'kwcoco' not in str(local_fpath) and '_cache' not in str(local_fpath):
                ipfs_stat = fs.stat(ipfs_fpath)
                local_stat = local_fpath.stat()
                # TODO: need better mechanism to determine if files are the same
                probably_same = (ipfs_stat['size'] == local_stat.st_size)
                if not probably_same:
                    to_copy.append({
                        'src': ipfs_fpath,
                        'dst': local_fpath,
                        'overwrite': True,
                    })

    # Execute Copy

    for dpath in to_ensure:
        dpath.ensuredir()

    for task in ub.ProgIter(to_copy, desc='copy files'):
        if task['overwrite']:
            src = task['src']
            dst = task['dst']
            fs.get(src, str(dst))
        else:
            src = task['src']
            dst = task['dst']
            bak = dst.augment(suffix='.old')
            dst.move(bak)
            fs.get(src, str(dst))
            bak.delete()


"""
Or for ooo: to just get the data

rsync -avprPR toothbrush:data/dvc-repos/./shitspotter_dvc $HOME/data/dvc-repos
rsync -avprPR toothbrush:data/dvc-repos/./shitspotter_expt_dvc $HOME/data/dvc-repos
"""


# async def test_http_async():
#     import fsspec
#     url1 = 'https://raw.githubusercontent.com/fsspec/filesystem_spec/master/pyproject.toml'
#     url2 = 'https://raw.githubusercontent.com/fsspec/filesystem_spec/master/README.md'
#     url3 = 'https://raw.githubusercontent.com/fsspec/filesystem_spec/master/LICENSE'

#     fs = fsspec.filesystem("http", asynchronous=True)
#     session = await fs.set_session()  # creates client
#     out = await fs._cat([url1, url2, url3])  # fetches data concurrently
#     out = await fs._cat([url1, url2, url3])  # fetches data concurrently
#     print(f'out={out}')
#     await session.close()  # explicit destructor


# async def test_ipfs_async():
#     import fsspec
#     fs_cls = fsspec.get_filesystem_class('ipfs')
#     fs = fs_cls(asynchronous=True)
#     session = await fs.set_session()  # creates client
#     result = await fs._ls("bafybeief7tmoarwmd26b2petx7crtvdnz6ucccek5wpwxwdvfydanfukna")
#     print(result)
#     await session.close()  # explicit destructor


if __name__ == '__main__':
    import asyncio
    asyncio.run(sync_ipfs_pull())
