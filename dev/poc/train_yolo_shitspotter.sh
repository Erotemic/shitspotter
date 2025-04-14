#!/bin/bash
BUNDLE_DPATH=/data/joncrall/dvc-repos/shitspotter_dvc
TRAIN_FPATH=$BUNDLE_DPATH/train_imgs5747_1e73d54f.mscoco.json
VALI_FPATH=$BUNDLE_DPATH/vali_imgs691_99b22ad0.mscoco.json
echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"

#kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"

REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
MODULE_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent)")
CONFIG_DPATH=$(python -c "import yolo.config, pathlib; print(pathlib.Path(yolo.config.__file__).parent / 'dataset')")
echo "REPO_DPATH = $REPO_DPATH"
echo "MODULE_DPATH = $MODULE_DPATH"
echo "CONFIG_DPATH = $CONFIG_DPATH"

DATASET_CONFIG_FPATH=$CONFIG_DPATH/shitspotter-v1.yaml

# Hack to construct the class part of the YAML
CLASS_YAML=$(python -c "if 1:
    import kwcoco
    train_fpath = kwcoco.CocoDataset('$TRAIN_FPATH')
    categories = train_fpath.categories().objs
    # It would be nice to have better class introspection, but in the meantime
    # do the same sorting as yolo.tools.data_conversion.discretize_categories
    categories = sorted(categories, key=lambda cat: cat['id'])
    class_num = len(categories)
    class_list = [c['name'] for c in categories]
    print(f'class_num: {class_num}')
    print(f'class_list: {class_list}')
")


CONFIG_YAML="
path: $BUNDLE_DPATH
train: $TRAIN_FPATH
validation: $VALI_FPATH

$CLASS_YAML
"
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01"
echo "$CONFIG_YAML" > "$DATASET_CONFIG_FPATH"

export CUDA_VISIBLE_DEVICES="1,"
cd "$REPO_DPATH"
export DISABLE_RICH_HANDLER=
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-v1 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=mit-yolo-v01 \
    cpu_num=8 \
    device=0 \
    trainer.accelerator=auto \
    task.data.batch_size=8 \
    "image_size=[640,640]" \
    task.optimizer.args.lr=0.0003 \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.001 \
    task.data.data_augment.Mosaic=0.5 \
    task.loss.objective.BCELoss=0.0000001 \
    weight="/home/joncrall/code/YOLO-v9/weights/v9-c.pt"



export CUDA_VISIBLE_DEVICES="1,"
cd "$REPO_DPATH"
export DISABLE_RICH_HANDLER=
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-v1 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=mit-yolo-v02 \
    cpu_num=8 \
    device=0 \
    trainer.accelerator=auto \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.optimizer.args.lr=0.00003 \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.0001 \
    task.data.data_augment.Mosaic=0.5 \
    task.loss.objective.BCELoss=0.001 \
    'weight="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_20/checkpoints/epoch=0257-step=065274-trainlosstrain_loss=0.000.ckpt.ckpt"'
    #weight="\"$CKPT_FPATH\"" \


export CUDA_VISIBLE_DEVICES="0,"
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03"
REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
cd "$REPO_DPATH"
export DISABLE_RICH_HANDLER=
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-v1 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=mit-yolo-v02 \
    cpu_num=8 \
    device=0 \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches=8 \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.optimizer.args.lr=0.0003 \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.0001 \
    task.data.data_augment.Mosaic=0.00001 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.01 \
    'weight="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/epoch=0126-step=016129-trainlosstrain_loss=0.028.ckpt.ckpt"'


export CUDA_VISIBLE_DEVICES="1,"
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04"
REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
cd "$REPO_DPATH"
export DISABLE_RICH_HANDLER=
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-v1 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=mit-yolo-v02 \
    cpu_num=8 \
    device=0 \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches=32 \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.optimizer.args.lr=0.001 \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.0001 \
    task.data.data_augment.Mosaic=0.00001 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.1 \
    'weight="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/epoch=0126-step=016129-trainlosstrain_loss=0.028.ckpt.ckpt"'



##### TRAIN WITH SIMPLIFIED DATA (bigger boxes when there is overlap, only 1 class, no empty images)
BUNDLE_DPATH=/data/joncrall/dvc-repos/shitspotter_dvc
TRAIN_FPATH=$BUNDLE_DPATH/simplified_train_imgs6917_05c90c75.kwcoco.zip
VALI_FPATH=$BUNDLE_DPATH/simplified_vali_imgs1258_4fb668db.kwcoco.zip
echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"

#kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"

REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
MODULE_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent)")
CONFIG_DPATH=$(python -c "import yolo.config, pathlib; print(pathlib.Path(yolo.config.__file__).parent / 'dataset')")
echo "REPO_DPATH = $REPO_DPATH"
echo "MODULE_DPATH = $MODULE_DPATH"
echo "CONFIG_DPATH = $CONFIG_DPATH"

DATASET_CONFIG_FPATH=$CONFIG_DPATH/shitspotter-simple-v2.yaml

# Hack to construct the class part of the YAML
CLASS_YAML=$(python -c "if 1:
    import kwcoco
    train_fpath = kwcoco.CocoDataset('$TRAIN_FPATH')
    categories = train_fpath.categories().objs
    # It would be nice to have better class introspection, but in the meantime
    # do the same sorting as yolo.tools.data_conversion.discretize_categories
    categories = sorted(categories, key=lambda cat: cat['id'])
    class_num = len(categories)
    class_list = [c['name'] for c in categories]
    print(f'class_num: {class_num}')
    print(f'class_list: {class_list}')
")

CONFIG_YAML="
path: $BUNDLE_DPATH
train: $TRAIN_FPATH
validation: $VALI_FPATH

$CLASS_YAML
"
echo "$CONFIG_YAML" > "$DATASET_CONFIG_FPATH"
export CUDA_VISIBLE_DEVICES="1,"
REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v2-run-v01"
export DISABLE_RICH_HANDLER=
export CUDA_VISIBLE_DEVICES=0
cd "$REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-simple-v2 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=shitspotter-simple-v2-run-v01 \
    cpu_num=8 \
    device=0 \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches=50 \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.optimizer.args.lr=0.001 \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.5 \
    task.data.data_augment.Mosaic=0.0 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.1 \
    weight="/home/joncrall/code/YOLO-v9/weights/v9-c.pt"


##
REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v2-run-v02"
export DISABLE_RICH_HANDLER=
export CUDA_VISIBLE_DEVICES=1
cd "$REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-simple-v2 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=shitspotter-simple-v2-run-v02 \
    cpu_num=8 \
    device=0 \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches=50 \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.optimizer.args.lr=0.003 \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.5 \
    task.data.data_augment.Mosaic=0.0 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.1 \
    weight="/home/joncrall/code/YOLO-v9/weights/v9-c.pt"


## TRY WITH ADAMW AND ONECYCLE SCHEDULER
REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v2-run-v03"
export DISABLE_RICH_HANDLER=
export CUDA_VISIBLE_DEVICES=0
cd "$REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-simple-v2 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=shitspotter-simple-v2-run-v03 \
    cpu_num=8 \
    device=0 \
    task.optimizer.type=AdamW \
    +task.optimizer.args='{lr: 0.001, weight_decay: 0.01, betas: [0.9, 0.99]}' \
    ~task.optimizer.args.nesterov \
    ~task.optimizer.args.momentum \
    task.scheduler.type=OneCycleLR \
    +task.scheduler.args='{max_lr: 0.001, total_steps: 10000, anneal_strategy: cos, pct_start: 0.3}' \
    ~task.scheduler.args.total_iters \
    ~task.scheduler.args.start_factor \
    ~task.scheduler.args.end_factor \
    ~task.scheduler.warmup \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches=50 \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.5 \
    task.data.data_augment.Mosaic=0.0 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.1 \
    weight="/home/joncrall/code/YOLO-v9/weights/v9-c.pt"


## TRY WITH ADAMW AND ONECYCLE SCHEDULER
REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v2-run-v04"
export DISABLE_RICH_HANDLER=
export CUDA_VISIBLE_DEVICES=1
cd "$REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-simple-v2 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=shitspotter-simple-v2-run-v04 \
    cpu_num=8 \
    device=0 \
    task.optimizer.type=AdamW \
    +task.optimizer.args='{lr: 0.001, weight_decay: 0.01, betas: [0.9, 0.99]}' \
    ~task.optimizer.args.nesterov \
    ~task.optimizer.args.momentum \
    task.scheduler.type=OneCycleLR \
    +task.scheduler.args='{max_lr: 0.001, total_steps: 10000, anneal_strategy: cos, pct_start: 0.3}' \
    ~task.scheduler.args.total_iters \
    ~task.scheduler.args.start_factor \
    ~task.scheduler.args.end_factor \
    ~task.scheduler.warmup \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches=50 \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.5 \
    task.data.data_augment.Mosaic=0.0 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.1 \
    weight="/home/joncrall/code/YOLO-v9/weights/v9-c.pt"


REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v2-run-v05"
export DISABLE_RICH_HANDLER=
export CUDA_VISIBLE_DEVICES=0
cd "$REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-simple-v2 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=shitspotter-simple-v2-run-v05 \
    cpu_num=8 \
    device=0 \
    task.optimizer.type=AdamW \
    +task.optimizer.args='{lr: 0.001, weight_decay: 0.0001, betas: [0.9, 0.99]}' \
    ~task.optimizer.args.nesterov \
    ~task.optimizer.args.momentum \
    task.scheduler.type=OneCycleLR \
    +task.scheduler.args='{max_lr: 0.001, total_steps: 2500, anneal_strategy: cos, pct_start: 0.1}' \
    ~task.scheduler.args.total_iters \
    ~task.scheduler.args.start_factor \
    ~task.scheduler.args.end_factor \
    ~task.scheduler.warmup \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches=50 \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.5 \
    task.data.data_augment.Mosaic=0.0 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.1 \
    weight="/home/joncrall/code/YOLO-v9/weights/v9-c.pt"


## TRY WITH ADAMW AND ONECYCLE SCHEDULER

##### TRAIN WITH bigger SIMPLIFIED DATA (bigger boxes when there is overlap, only 1 class, no empty images)
BUNDLE_DPATH=/data/joncrall/dvc-repos/shitspotter_dvc
TRAIN_FPATH=$BUNDLE_DPATH/simplified_train_imgs7350_4f0174d0.kwcoco.zip
VALI_FPATH=$BUNDLE_DPATH/simplified_vali_imgs1258_07ec447d.kwcoco.zip
echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"

#kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"

REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
MODULE_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent)")
CONFIG_DPATH=$(python -c "import yolo.config, pathlib; print(pathlib.Path(yolo.config.__file__).parent / 'dataset')")
echo "REPO_DPATH = $REPO_DPATH"
echo "MODULE_DPATH = $MODULE_DPATH"
echo "CONFIG_DPATH = $CONFIG_DPATH"

DATASET_CONFIG_FPATH=$CONFIG_DPATH/shitspotter-simple-v3.yaml

# Hack to construct the class part of the YAML
CLASS_YAML=$(python -c "if 1:
    import kwcoco
    train_fpath = kwcoco.CocoDataset('$TRAIN_FPATH')
    categories = train_fpath.categories().objs
    # It would be nice to have better class introspection, but in the meantime
    # do the same sorting as yolo.tools.data_conversion.discretize_categories
    categories = sorted(categories, key=lambda cat: cat['id'])
    class_num = len(categories)
    class_list = [c['name'] for c in categories]
    print(f'class_num: {class_num}')
    print(f'class_list: {class_list}')
")

CONFIG_YAML="
path: $BUNDLE_DPATH
train: $TRAIN_FPATH
validation: $VALI_FPATH

$CLASS_YAML
"
echo "$CONFIG_YAML" > "$DATASET_CONFIG_FPATH"
REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v06"
export DISABLE_RICH_HANDLER=
export CUDA_VISIBLE_DEVICES=1
cd "$REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-simple-v3 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=shitspotter-simple-v3-run-v06 \
    cpu_num=8 \
    device=0 \
    task.optimizer.type=AdamW \
    +task.optimizer.args='{lr: 0.001, weight_decay: 0.01, betas: [0.9, 0.99]}' \
    ~task.optimizer.args.nesterov \
    ~task.optimizer.args.momentum \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches=50 \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.5 \
    task.data.data_augment.Mosaic=0.0 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.5 \
    weight="/home/joncrall/code/YOLO-v9/weights/v9-c.pt"


REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v07"
export DISABLE_RICH_HANDLER=
export CUDA_VISIBLE_DEVICES=0
cd "$REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-simple-v3 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=shitspotter-simple-v3-run-v07 \
    cpu_num=8 \
    device=0 \
    task.optimizer.type=AdamW \
    +task.optimizer.args='{lr: 0.0003, weight_decay: 0.01, betas: [0.9, 0.99]}' \
    ~task.optimizer.args.nesterov \
    ~task.optimizer.args.momentum \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches=50 \
    task.data.batch_size=16 \
    "image_size=[640,640]" \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.5 \
    task.data.data_augment.Mosaic=0.0 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.5 \
    weight="/home/joncrall/code/YOLO-v9/weights/v9-c.pt"




cd "$HOME"/code/YOLO-v9
python yolo/lazy.py task=train dataset=shitspotter-v1 use_wandb=False task.data.batch_size=2 task.optimizer.args.lr=0.003 out_path=runs-v2 accelerator=auto


cd "$HOME"/code/YOLO-v9
python yolo/lazy.py task=inference dataset=kwcoco use_wandb=False

cd "$HOME"/code/YOLO-v9
BUNDLE_DPATH=/data/joncrall/dvc-repos/shitspotter_dvc
VALI_FPATH=$BUNDLE_DPATH/vali_imgs691_99b22ad0.mscoco.json
# Grab a checkpoint
CKPT_FPATH=$(python -c "if 1:
    import pathlib
    ckpt_dpath = pathlib.Path('runs/train/v9-dev/checkpoints')
    checkpoints = sorted(ckpt_dpath.glob('*'))
    print(checkpoints[-1].resolve())
")
echo "CKPT_FPATH = $CKPT_FPATH"

export DISABLE_RICH_HANDLER=1
export CUDA_VISIBLE_DEVICES="1,"
python yolo/lazy.py \
    task.data.source="$VALI_FPATH" \
    task=inference \
    dataset=kwcoco \
    use_wandb=False \
    out_path=shitspotter-inference \
    name=shitspotter-infernce-vali \
    cpu_num=8 \
    weight="\"$CKPT_FPATH\"" \
    accelerator=auto \
    task.nms.min_confidence=0.01 \
    task.nms.min_iou=0.5 \
    task.nms.max_bbox=10
    #save_predict=False \
