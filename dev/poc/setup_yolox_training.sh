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
        #self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.exp_name = 'shitspotter-custom'

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
        self.input_size = (640, 640)
        self.test_size  = (640, 640)
        #self.max_epoch = 100
        #self.eval_interval = 5
        #self.enable_mixup = True

        self.num_classes = 1
        self.max_epoch = 300
        self.data_num_workers = 8
        self.eval_interval = 1

        self.enable_mixup = False    # big speedup, minor accuracy hit on small datasets
        self.mosaic_prob = 0.0       # or 0.0 for max speed
        self.mixup_prob  = 0.0
        self.random_size = (18, 22)  # narrower multi-scale range (around 608–736 px)
        self.eval_interval = 10      # evaluate less often

if 0:
    self = Exp()
    datset= self.get_dataset()

" > exps/custom/yolox_s_custom.py



curl -L https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -O

we pyenv3.11.9
python tools/train.py -f exps/custom/yolox_s_custom.py -d 2 -b 16 --fp16 -o


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
        self.exp_name = 'shitspotter-custom-v2'

        # --- dataset + classes ---
        # If your json is literally named 'vali.json', put that here. Otherwise use 'val.json'.
        # The way they use data dir is broken. Try and work around it
        #self.data_dir = train_fpath.bundle_dpath

        self.train_ann = train_fpath.fpath
        self.val_ann   = vali_fpath.fpath

        self.train_dir = train_fpath.bundle_dpath
        self.val_dir   = vali_fpath.bundle_dpath

        # --- training knobs (tweak as needed) ---
        self.input_size = (640, 640)
        self.test_size  = (640, 640)

        self.num_classes = 1
        self.max_epoch = 300
        self.data_num_workers = 12

        self.enable_mixup = True    # big speedup, minor accuracy hit on small datasets
        self.mosaic_prob = 0.5       # or 0.0 for max speed
        self.mixup_prob  = 0.5
        #self.random_size = (18, 22)  # narrower multi-scale range (around 608–736 px)
        self.eval_interval = 5      # evaluate less often

if 0:
    self = Exp()
    datset= self.get_dataset()

" > exps/custom/yolox_s_custom_v2.py
python tools/train.py -f exps/custom/yolox_s_custom_v2.py -d 2 -b 16 --fp16 -o -c yolox_s.pth


#### EXPORT TO ONNX
we pyenv3.11.9
pip install onnxruntime

we pyenv3.11.9
cd /home/joncrall/code/shitspotter/tpl/YOLOX
ls YOLOX_outputs/shitspotter-custom-v2/last_epoch_ckpt.pth
python tools/export_onnx.py \
    --output-name shitspotter-custom-v2.onnx \
    --exp_file exps/custom/yolox_s_custom_v2.py \
    --ckpt YOLOX_outputs/shitspotter-custom-v2/last_epoch_ckpt.pth


# --- Train V3

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
        self.exp_name = 'shitspotter-custom-v3'

        # --- dataset + classes ---
        # If your json is literally named 'vali.json', put that here. Otherwise use 'val.json'.
        # The way they use data dir is broken. Try and work around it
        #self.data_dir = train_fpath.bundle_dpath

        self.train_ann = train_fpath.fpath
        self.val_ann   = vali_fpath.fpath

        self.train_dir = train_fpath.bundle_dpath
        self.val_dir   = vali_fpath.bundle_dpath

        # --- training knobs (tweak as needed) ---
        self.input_size = (640, 640)
        self.test_size  = (640, 640)

        self.num_classes = 1
        self.max_epoch = 300
        self.data_num_workers = 8

        self.enable_mixup = False    # big speedup, minor accuracy hit on small datasets
        self.mosaic_prob = 0.0       # or 0.0 for max speed
        self.mixup_prob  = 0.0
        #self.random_size = (18, 22)  # narrower multi-scale range (around 608–736 px)
        self.eval_interval = 5      # evaluate less often

if 0:
    self = Exp()
    datset= self.get_dataset()

" > exps/custom/yolox_s_custom_v3.py
we pyenv3.11.9
python tools/train.py -f exps/custom/yolox_s_custom_v3.py -d 2 -b 16 --fp16 -o -c yolox_s.pth


# --- Train V4

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
        self.exp_name = 'shitspotter-custom-v3'

        # --- dataset + classes ---
        # If your json is literally named 'vali.json', put that here. Otherwise use 'val.json'.
        # The way they use data dir is broken. Try and work around it
        #self.data_dir = train_fpath.bundle_dpath

        self.train_ann = train_fpath.fpath
        self.val_ann   = vali_fpath.fpath

        self.train_dir = train_fpath.bundle_dpath
        self.val_dir   = vali_fpath.bundle_dpath

        # --- training knobs (tweak as needed) ---
        self.input_size = (640, 640)
        self.test_size  = (640, 640)

        self.num_classes = 1
        self.max_epoch = 300
        self.data_num_workers = 8

        self.enable_mixup = True    # big speedup, minor accuracy hit on small datasets
        self.mosaic_prob = 0.5       # or 0.0 for max speed
        self.mixup_prob  = 0.5
        #self.random_size = (18, 22)  # narrower multi-scale range (around 608–736 px)
        self.eval_interval = 5      # evaluate less often

if 0:
    self = Exp()
    datset= self.get_dataset()

" > exps/custom/yolox_s_custom_v4.py
we pyenv3.11.9
python tools/train.py -f exps/custom/yolox_s_custom_v4.py -d 2 -b 16 --fp16 -o -c yolox_s.pth


#### -----------

DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
ORIG_TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs9270_f2b4b17d.kwcoco.zip
ORIG_VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs1258_fe7f7dfe.kwcoco.zip
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs9270_f2b4b17d-simple-poop-only.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs1258_fe7f7dfe-simple-poop-only.json

echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"

kwcoco modify_categories --keep poop --src "$ORIG_TRAIN_FPATH" --dst "$TRAIN_FPATH"
kwcoco modify_categories --keep poop --src "$ORIG_VALI_FPATH" --dst "$VALI_FPATH"
kwcoco conform --legacy=True --src "$TRAIN_FPATH" --inplace
kwcoco conform --legacy=True --src "$VALI_FPATH" --inplace
kwcoco reroot --absolute=True --src "$TRAIN_FPATH" --inplace
kwcoco reroot --absolute=True --src "$VALI_FPATH" --inplace

python -m shitspotter.cli.simplify_kwcoco --src "$TRAIN_FPATH" --inplace
python -m shitspotter.cli.simplify_kwcoco --src "$VALI_FPATH" --inplace

kwcoco stats "$ORIG_TRAIN_FPATH" "$TRAIN_FPATH"
kwcoco stats "$ORIG_VALI_FPATH" "$VALI_FPATH"


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
        train_fpath = kwcoco.CocoDataset('/home/joncrall/data/dvc-repos/shitspotter_dvc/train_imgs9270_f2b4b17d-simple-poop-only.json')
        vali_fpath = kwcoco.CocoDataset('/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe-simple-poop-only.json')

        # --- model scale (keep as yolox-s) ---
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = 'shitspotter-custom-v5'

        # --- dataset + classes ---
        # If your json is literally named 'vali.json', put that here. Otherwise use 'val.json'.
        # The way they use data dir is broken. Try and work around it
        #self.data_dir = train_fpath.bundle_dpath

        self.train_ann = train_fpath.fpath
        self.val_ann   = vali_fpath.fpath

        self.train_dir = train_fpath.bundle_dpath
        self.val_dir   = vali_fpath.bundle_dpath

        # --- training knobs (tweak as needed) ---
        self.input_size = (640, 640)
        self.test_size  = (640, 640)

        self.num_classes = 1
        self.max_epoch = 300
        self.data_num_workers = 8

        self.enable_mixup = True    # big speedup, minor accuracy hit on small datasets
        self.mosaic_prob = 0.5       # or 0.0 for max speed
        self.mixup_prob  = 0.5
        #self.random_size = (18, 22)  # narrower multi-scale range (around 608–736 px)
        self.eval_interval = 5      # evaluate less often

if 0:
    self = Exp()
    datset= self.get_dataset()

" > exps/custom/yolox_s_custom_v5.py
we pyenv3.11.9
python tools/train.py -f exps/custom/yolox_s_custom_v5.py -d 2 -b 16 --fp16 -o -c yolox_s.pth

## This trained the best YOLOx model so far:
# /home/joncrall/code/shitspotter/tpl/YOLOX/YOLOX_outputs/shitspotter-custom-v5/epoch_115_ckpt.pth
we pyenv3.11.9
cd /home/joncrall/code/shitspotter/tpl/YOLOX
ls YOLOX_outputs/shitspotter-custom-v5/epoch_115_ckpt.pth
python tools/export_onnx.py \
    --output-name "$HOME"/code/shitspotter/tpl/poop_models/shitspotter-custom-v5-epoch_115.onnx \
    --exp_file exps/custom/yolox_s_custom_v2.py \
    --ckpt YOLOX_outputs/shitspotter-custom-v5/epoch_115_ckpt.pth


# TODO: look at the tensorflow lite convertor
#~/code/shitspotter/dev/poc/onnx_to_tflite.py

# This requires older versions of onnx and tensorflow
we pyenv3.11.9
uv pip install "onnx==1.15.0" "onnx-tf==1.10.0" "tensorflow==2.15.*" "tensorflow_probability==0.23.0"

python ~/code/shitspotter/dev/poc/onnx_to_tflite.py \
  --input "$HOME"/code/shitspotter/tpl/poop_models/shitspotter-custom-v5-epoch_115.onnx \
  --output "$HOME"/code/shitspotter/tpl/poop_models/shitspotter-custom-v5-epoch_115-float16.tf \
  --normalize False \
  --dtype=float16

KWCOCO_BUNDLE_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs9270_f2b4b17d-simple-poop-only.json

python ~/code/shitspotter/dev/poc/onnx_to_tflite.py \
  --input "$HOME"/code/shitspotter/tpl/poop_models/shitspotter-custom-v5-epoch_115.onnx \
  --output "$HOME"/code/shitspotter/tpl/poop_models/shitspotter-custom-v5-epoch_115-full-int8.tf \
  --normalize False \
  --calibration_data "$TRAIN_FPATH" \
  --dtype=full-int8 \
  --input_shape="1,640,640,3"
