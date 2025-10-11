cd /home/joncrall/code/shitspotter/tpl

git submodule add https://github.com/Megvii-BaseDetection/YOLOX.git

cd /home/joncrall/code/shitspotter/tpl/YOLOX

we pyenv3.11.9
uv pip install onnx-simplifier
uv pip install pycocotools
uv pip install loguru tqdm thop ninja tabulate tensorboard torchvision torch numpy
python setup.py build_ext --inplace
pip install --no-deps -e . -v


#uv pip install -r requirements.txt
pip install -v -e .          # develop install
# COCO API (for COCO-format datasets & eval)
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
ORIG_TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs8763_e58dbbb2.kwcoco.zip
ORIG_VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs1258_63f16fdc.kwcoco.zip
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs8763_e58dbbb2-poop-only.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs1258_63f16fdc-poop-only.json

echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"

kwcoco modify_categories --keep poop --src "$ORIG_TRAIN_FPATH" --dst "$TRAIN_FPATH"
kwcoco modify_categories --keep poop --src "$ORIG_VALI_FPATH" --dst "$VALI_FPATH"
kwcoco conform --legacy=True --src "$TRAIN_FPATH" --inplace
kwcoco conform --legacy=True --src "$VALI_FPATH" --inplace
kwcoco reroot --absolute=True --src "$TRAIN_FPATH" --inplace
kwcoco reroot --absolute=True --src "$VALI_FPATH" --inplace

cd /home/joncrall/code/shitspotter/tpl/YOLOX
mkdir -p exps/custom
echo "
# exps/custom/yolox_s_custom.py
from yolox.exp import Exp as Exp_Base
import os

class Exp(Exp_Base):
    def __init__(self):
        super().__init__()

        import kwcoco
        train_fpath = kwcoco.CocoDataset('/home/joncrall/data/dvc-repos/shitspotter_dvc/train_imgs8763_e58dbbb2-poop-only.json')
        vali_fpath = kwcoco.CocoDataset('/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_63f16fdc-poop-only.json')

        # --- model scale (keep as yolox-s) ---
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

        # --- dataset + classes ---
        #self.num_classes =  <YOUR_NUM_CLASSES>   # e.g., 3
        #self.data_dir = '/ABS/PATH/TO/DATASET_ROOT'  # folder that contains your images/ and jsons

        # If your json is literally named 'vali.json', put that here. Otherwise use 'val.json'.
        # The way they use data dir is broken. Try and work around it
        #self.data_dir = train_fpath.bundle_dpath

        self.train_ann = train_fpath.fpath
        self.val_ann   = vali_fpath.fpath

        self.train_dir = train_fpath.bundle_dpath
        self.val_dir   = vali_fpath.bundle_dpath

        # --- training knobs (tweak as needed) ---
        #self.input_size = (640, 640)
        #self.test_size  = (640, 640)
        #self.max_epoch = 100
        #self.eval_interval = 5
        #self.enable_mixup = True

        self.num_classes = 1
        self.max_epoch = 300
        self.data_num_workers = 8
        self.eval_interval = 1

if 0:
    self = Exp()

" > exps/custom/yolox_s_custom.py


we pyenv3.11.9
python tools/train.py -f exps/custom/yolox_s_custom.py -d 1 -b 16 --fp16 -o

