TODO: IPFS Setup Document
-------------------------

See dev/install_ipfs.sh



..

    https://chatgpt.com/share/61a1f243-1b48-4301-b91f-3825b19cf9e3

Instructions To Download Via IPFS
---------------------------------


.. code:: bash
    #export IPFS_PATH=/flash/ipfs
    ROOT_CID=bafybeidle54us5cdwpzzis4h52wjmtsk643gprx7nvvtd6g26mxq76kfjm
    ipfs ls $ROOT_CID
    DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
    mkdir -p $DVC_DATA_DPATH
    ipfs get -o "$DVC_DATA_DPATH" "$ROOT_CID"
