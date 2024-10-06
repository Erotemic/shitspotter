#!/usr/bin/bash
__doc__="
Manage data checked into IPFS
"

# To update the data

python -m shitspotter.ipfs add /home/joncrall/code/shitspotter/papers/application-2024/figures-submitted-2024-09-09
python -m shitspotter.ipfs add /home/joncrall/code/shitspotter/papers/application-2024/figures

cat /home/joncrall/code/shitspotter/papers/application-2024/figures-submitted-2024-09-09.ipfs
cd /home/joncrall/code/shitspotter/papers/application-2024/
git add figures-submitted-2024-09-09.ipfs


python -m shitspotter.ipfs pull /home/joncrall/code/shitspotter/papers/application-2024/figures-submitted-2024-09-09.ipfs
python -m shitspotter.ipfs pull /home/joncrall/code/shitspotter/papers/application-2024/figures.ipfs
