# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
__doc__='

https://colab.research.google.com/drive/1DIk7bDpdZDkTTZyJbPADZklcbZKr1xkn#scrollTo=DvVulbjZcTdp

'

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

cd "$HOME"/code
git clone https://github.com/facebookresearch/detectron2.git
cd "$HOME"/code/detectron2

python -m pip install -e .

ls "$HOME"/data/dvc-repos/shitspotter_dvc
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
TRAIN_FPATH=$DVC_DATA_DPATH/train_imgs5747_1e73d54f.kwcoco.zip
VALI_FPATH=$DVC_DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip
TRAIN_FPATH2=$DVC_DATA_DPATH/train_imgs5747_1e73d54f.mscoco.json
VALI_FPATH2=$DVC_DATA_DPATH/vali_imgs691_99b22ad0.mscoco.json

kwcoco conform --legacy=True --src "$TRAIN_FPATH" --dst "$TRAIN_FPATH2"
kwcoco conform --legacy=True --src "$VALI_FPATH" --dst "$VALI_FPATH2"

echo $TRAIN_FPATH


python -c "if 1:

from detectron2.data.datasets import register_coco_instances
register_coco_instances('vali_imgs691_99b22ad0', {}, '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json', '/home/joncrall/data/dvc-repos/shitspotter_dvc')
register_coco_instances('train_imgs5747_1e73d54f', {}, '/home/joncrall/data/dvc-repos/shitspotter_dvc/train_imgs5747_1e73d54f.mscoco.json', '/home/joncrall/data/dvc-repos/shitspotter_dvc')


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

import detectron2
modpath = ub.Path(detectron2.__file__)
repo_path = modpath.parent.parent

cfg = get_cfg()
cfg.merge_from_file(repo_path / 'configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.DATASETS.TRAIN = ('train_imgs5747_1e73d54f',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

"

