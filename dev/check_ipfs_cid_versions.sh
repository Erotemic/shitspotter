#!/bin/bash



cd /home/joncrall/data/dvc-repos/shitspotter_dvc/assets/poop-2021-04-19
FPATH=PXL_20210411_150641385.jpg


echo "foobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobarfoobar" > foobar
FPATH=foobar

dd if=/dev/zero of=lots_of_zeros count=1 bs=257k
FPATH=lots_of_zeros

sha256sum "$FPATH"
CID_V0_DEFAULT=$(ipfs add -qn --cid-version 0 "$FPATH")
CID_V1_DEFAULT=$(ipfs add -qn --cid-version 1 "$FPATH")
CID_V0_RLT=$(ipfs add -qn --cid-version 0 --raw-leaves=true "$FPATH")
CID_V0_RLF=$(ipfs add -qn --cid-version 0 --raw-leaves=false "$FPATH")
CID_V1_RLT=$(ipfs add -qn --cid-version 1 --raw-leaves=true "$FPATH")

CID_V1_RLF=$(ipfs add -qn --cid-version 1 --raw-leaves=false "$FPATH")
CID_V1_FROM_V0_RLT=$(ipfs cid base32 "$CID_V0_RLT")
CID_V1_FROM_V0_RLF=$(ipfs cid base32 "$CID_V0_RLF")


echo "--raw-leaves=true results"
echo "---"
echo "CID_V0_RLT         = $CID_V0_RLT"
echo "---"
echo "CID_V1_DEFAULT     = $CID_V1_DEFAULT"
echo "CID_V1_RLT         = $CID_V1_RLT"
echo "CID_V1_FROM_V0_RLT = $CID_V1_FROM_V0_RLT"
echo "---"
echo ""
echo ""
echo "--raw-leaves=false results"
echo "---"
echo "CID_V0_DEFAULT     = $CID_V0_DEFAULT"
echo "CID_V0_RLF         = $CID_V0_RLF"
echo "---"
echo "CID_V1_RLF         = $CID_V1_RLF"
echo "CID_V1_FROM_V0_RLF = $CID_V1_FROM_V0_RLF"
