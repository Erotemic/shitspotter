"""
Requirements:
    pip install ipfsspec
"""

import fsspec

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


import ipfsspec
fs = fsspec.get_filesystem_class('ipfs')

fs = ipfsspec.core.IPFSFileSystem(timeout=100)

cid = "bafybeief7tmoarwmd26b2petx7crtvdnz6ucccek5wpwxwdvfydanfukna"
res = fs._gw_apipost("ls", arg=cid)
links = res["Objects"][0]["Links"]

fs.ls(cid)

# fs = fsspec.open(, "r")
