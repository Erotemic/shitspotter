#!/usr/bin/bash
__doc__="
Manage data checked into IPFS
"

# To update the data

python -m shitspotter.ipfs add /home/joncrall/code/shitspotter/papers/wacv_2026/figures --name "shitspotter-figures-wacv_2026"
python -m shitspotter.ipfs add /home/joncrall/code/shitspotter/papers/wacv_2026/plots --name "shitspotter-plots-wacv_2026"

cat /home/joncrall/code/shitspotter/papers/wacv_2026/shitspotter-figures-wacv_2026
cd /home/joncrall/code/shitspotter/papers/wacv_2026/
git add figures-submitted-2024-09-09.ipfs


python -m shitspotter.ipfs pull /home/joncrall/code/shitspotter/papers/wacv_2026/figures-submitted-2024-09-09.ipfs
python -m shitspotter.ipfs pull /home/joncrall/code/shitspotter/papers/wacv_2026/figures.ipfs

# Get the pin commands to ensure we host this
python -m shitspotter.ipfs pin add -n -- /home/joncrall/code/shitspotter/papers/wacv_2026/figures.ipfs
python -m shitspotter.ipfs pin add -n -- /home/joncrall/code/shitspotter/papers/wacv_2026/figures-submitted-2024-09-09.ipfs


ipfs pin add --name shitspotter-figures --progress --recursive bafybeib5xppxduoudgamopadsbcregoxg2siyktbltotn522etadigopwe
ipfs pin add --name shitspotter-figures-submitted-2024-09-09 --progress --recursive bafybeib5xppxduoudgamopadsbcregoxg2siyktbltotn522etadigopwe
ipfs pin add --name shitspotter-figures --progress --recursive bafybeibcho3molnx2fgbueeptwzk3kbvpcswkmbscwlkexpuwtaqkqo2ii


python -m shitspotter.ipfs add /data/joncrall/dvc-repos/shitspotter_dvc/models --name "shitspotter-models-2024-04-20"
