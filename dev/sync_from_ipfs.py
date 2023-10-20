"""
Requirements:
    pip install ipfsspec
"""


def test_sync():
    import os
    import shitspotter
    # import ipfsspec
    # import logging
    # import sys
    import ubelt as ub
    # avail = sorted(fsspec.available_protocols())
    # print('avail = {}'.format(ub.urepr(avail, nl=1)))
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    import fsspec
    fs_cls = fsspec.get_filesystem_class('ipfs')
    fs = fs_cls(asynchronous=False)
    results = fs.ls("bafybeief7tmoarwmd26b2petx7crtvdnz6ucccek5wpwxwdvfydanfukna")
    print('results = {}'.format(ub.urepr(results, nl=1)))

    cid = "bafybeief7tmoarwmd26b2petx7crtvdnz6ucccek5wpwxwdvfydanfukna"
    # res = fs._gw_apipost("ls", arg=cid)
    # links = res["Objects"][0]["Links"]

    results = fs.ls(cid)
    print('results = {}'.format(ub.urepr(results, nl=1)))

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


"""
rsync -avprPR toothbrush:data/dvc-repos/./shitspotter_dvc $HOME/data/dvc-repos
"""


async def test_http_async():
    import fsspec
    url1 = 'https://raw.githubusercontent.com/fsspec/filesystem_spec/master/pyproject.toml'
    url2 = 'https://raw.githubusercontent.com/fsspec/filesystem_spec/master/README.md'
    url3 = 'https://raw.githubusercontent.com/fsspec/filesystem_spec/master/LICENSE'

    fs = fsspec.filesystem("http", asynchronous=True)
    session = await fs.set_session()  # creates client
    out = await fs._cat([url1, url2, url3])  # fetches data concurrently
    out = await fs._cat([url1, url2, url3])  # fetches data concurrently
    print(f'out={out}')
    await session.close()  # explicit destructor


async def test_ipfs_async():
    import fsspec
    fs_cls = fsspec.get_filesystem_class('ipfs')
    fs = fs_cls(asynchronous=True)
    session = await fs.set_session()  # creates client
    result = await fs._ls("bafybeief7tmoarwmd26b2petx7crtvdnz6ucccek5wpwxwdvfydanfukna")
    print(result)
    await session.close()  # explicit destructor


if __name__ == '__main__':
    import asyncio
    asyncio.run(test_ipfs_async())
