#!/bin/bash
__doc__='
The official grounding dino codebase does not seem to have fine-tuning code
available.

Official Repo:
https://github.com/IDEA-Research/GroundingDINO

But there is a third party implementation that seems to:
https://github.com/longzw1997/Open-GroundingDino#

SeeAlso:

    ~/code/Open-GroundingDino/README.md
    ~/code/Open-GroundingDino/train_dist.sh

'

cd "$HOME"/code
git clone git@github.com:Erotemic/Open-GroundingDino.git

cd "$HOME"/code/Open-GroundingDino

# Remove conflict requirements
cp requirements.txt tmp_requirements.txt
sed -i 's|opencv-python|#opencv-python|g' tmp_requirements.txt
sed -i 's|supervision|#supervision|g' tmp_requirements.txt
#sed -i 's|yapf|#yapf|g' tmp_requirements.txt
uv pip install supervision==0.6.0 --no-deps
uv pip install -r tmp_requirements.txt


# OMG, c module hell
sudo apt install nvidia-cuda-toolkit

try_tpl(){
    mkdir tpl
    cd tpl
    git submodule add git@github.com:Erotemic/GroundingDINO.git
    cd GroundingDINO
    git remote add IDEA-Research https://github.com/IDEA-Research/GroundingDINO.git
    git fetch
    git reset --hard IDEA-Research/main
    # Modified 2 places in models/GroundingDINO/ops/src/cuda/ms_deform_attn_cuda.cu
    python setup.py build_ext --inplace -v
}

# Compile C extensions

cd models/GroundingDINO/ops
# Modified 2 places in ~/code/Open-GroundingDino/models/GroundingDINO/ops/src/cuda/ms_deform_attn_cuda.cu
# to go from type() -> scalar_type() to make this work on 3.13 with torch 2.7.1+cu126, nvcc cuda_12.0.r12.0/compiler.32267302_0
python setup.py build_ext --inplace -v
# IDK why, this can't find required libs. Force it to.
TORCH_LIB_DPATH=$(dirname $(find $(python -c "import torch; print(torch.__path__[0])") -name "libc10.so"))
export LD_LIBRARY_PATH=$TORCH_LIB_DPATH:$LD_LIBRARY_PATH
ldd MultiScaleDeformableAttention*
# Test we can import
python -c "import MultiScaleDeformableAttention"
cd ../../..
cp models/GroundingDINO/ops/MultiScaleDeformableAttention.*.so .
python -c "import MultiScaleDeformableAttention"


# Hack together datasets

#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs5747_1e73d54f.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.kwcoco.zip

test -e "$TRAIN_FPATH" || echo "CANNOT TRAIN_FPATH"
test -e "$VALI_FPATH" || echo "CANNOT VALI_FPATH"

# Hack around issues with kwcoco reroot
kwcoco reroot --absolute=True --src "$TRAIN_FPATH" --dst "$TRAIN_FPATH.absolute.tmp"
kwcoco modify_categories --src "$TRAIN_FPATH.absolute.tmp" --dst train_imgs5747_1e73d54f.tmp.json --start_id=0
kwcoco tables -g 3 train_imgs5747_1e73d54f.tmp.json
kwcoco tables -c 10 train_imgs5747_1e73d54f.tmp.json

kwcoco reroot --absolute=True --src "$VALI_FPATH" --dst "$VALI_FPATH.absolute.tmp"
kwcoco modify_categories --src "$VALI_FPATH.absolute.tmp" --dst vali_imgs691_99b22ad0.tmp.json --start_id=0
kwcoco conform --legacy=True --src "vali_imgs691_99b22ad0.tmp.json" --inplace

kwcoco tables -g 3 vali_imgs691_99b22ad0.tmp.json
kwcoco tables -c 10 vali_imgs691_99b22ad0.tmp.json


# We need to convert COCO to odvg
# NOTE: We may need to adjust hard-coded label mappings
python tools/coco2odvg.py --input "train_imgs5747_1e73d54f.tmp.json" --output train_imgs5747_1e73d54f.odvg.jsonl --idmap=False


# We are required to modify a dataset config file
# config/datasets_mixed_odvg.json
# Or maybe we can just write our own here:
# Oh how annoying and truly unnecessary this is
# If only there was a way to infer this from a single source of truth dataset file!
#
# Apparently it is essential to start from zero, but our dataset does that anyway.
python -c "if 1:
    import kwcoco
    import ubelt as ub
    import json
    dset = kwcoco.CocoDataset('train_imgs5747_1e73d54f.tmp.json')
    lblmap = {cat['id']: cat['name'] for cat in dset.cats.values()}
    fpath = ub.Path('shitspotter_label_map.json')
    fpath.write_text(json.dumps(lblmap))
"
cat shitspotter_label_map.json

# Write out a custom dataset file to declare our splits
echo '
{
  "train": [
    {
      "root": "path/coco_2017/train2017/",
      "anno": "/home/joncrall/code/Open-GroundingDino/train_imgs5747_1e73d54f.odvg.jsonl",
      "label_map": "/home/joncrall/code/Open-GroundingDino/shitspotter_label_map.json",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "/home/joncrall/data/dvc-repos/shitspotter_dvc",
      "anno": "/home/joncrall/code/Open-GroundingDino/vali_imgs691_99b22ad0.tmp.json",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
' > config/shitspotter_datasets.json
# Test for json errors
cat config/shitspotter_datasets.json | jq



cp config/cfg_odvg.py config/shitspotter_cfg_odvg.py
sed -i 's|use_coco_eval = True|use_coco_eval = False|g' config/shitspotter_cfg_odvg.py
# add the label list to our config
echo "" >> config/shitspotter_cfg_odvg.py
echo "label_list = ['poop', 'unknown']" >> config/shitspotter_cfg_odvg.py


# Download relevant pretrained weights
if [ ! -f groundingdino_swint_ogc.pth ]; then
    curl -L https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -o groundingdino_swint_ogc.pth
fi
python -c "if 1:
    from transformers import AutoModel, AutoTokenizer
    AutoModel.from_pretrained('bert-base-uncased')
    AutoTokenizer.from_pretrained('bert-base-uncased')
"


# Test the ODVGDataset
python -c "if 1:
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/Open-GroundingDino'))
    from datasets.odvg import ODVGDataset
    import kwutil
    kw = kwutil.Json.coerce('config/shitspotter_datasets.json')['train'][0]
    self = ODVGDataset(
        root=kw['root'],
        anno=kw['anno'],
        label_map_anno=kw['label_map']
    )
    self.label_index
    print(len(self))
    image, target = self[1]
    print(target)
"

DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=ShitSpotter
EXPERIMENT_NAME=grounding-dino-tune-v001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

export GPU_NUM=1
export CFG=config/shitspotter_cfg_odvg.py
export DATASETS=config/shitspotter_datasets.json
export OUTPUT_DIR=$DEFAULT_ROOT_DIR
export PRETRAIN_MODEL_PATH=groundingdino_swint_ogc.pth
#export TEXT_ENCODER_TYPE=/home/joncrall/code/Open-GroundingDino/weights/bert-base-uncased
export TEXT_ENCODER_TYPE=bert-base-uncased

bash train_dist.sh  ${GPU_NUM} ${CFG} ${DATASETS} "${OUTPUT_DIR}"

python main.py \
        --output_dir "${OUTPUT_DIR}" \
        -c "${CFG}" \
        --datasets "${DATASETS}"  \
        --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
        --options text_encoder_type="$TEXT_ENCODER_TYPE"

# Started training at 2025-07-26T18:03:08
