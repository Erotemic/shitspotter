#!/bin/bash

# Use splits from paper
BUNDLE_DPATH=/data/joncrall/dvc-repos/shitspotter_dvc
TRAIN_FPATH=$BUNDLE_DPATH/train_imgs5747_1e73d54f.mscoco.json
VALI_FPATH=$BUNDLE_DPATH/vali_imgs691_99b22ad0.mscoco.json

# Note:
# could use python -m shitspotter.cli.simplify_kwcoco
# but we are not to keep things consistent across experiments.

REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
MODULE_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent)")
CONFIG_DPATH=$(python -c "import yolo.config, pathlib; print(pathlib.Path(yolo.config.__file__).parent / 'dataset')")
echo "REPO_DPATH = $REPO_DPATH"
echo "MODULE_DPATH = $MODULE_DPATH"
echo "CONFIG_DPATH = $CONFIG_DPATH"

DATASET_CONFIG_FPATH=$CONFIG_DPATH/shitspotter-yolo-train_imgs5747_1e73d54f-v5.yaml

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
TRAIN_DPATH="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5"
export DISABLE_RICH_HANDLER=
export CUDA_VISIBLE_DEVICES=1
cd "$REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=shitspotter-yolo-train_imgs5747_1e73d54f-v5 \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$TRAIN_DPATH" \
    name=shitspotter-yolo-train_imgs5747_1e73d54f-v5 \
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
    weight="null"
    #/home/joncrall/code/YOLO-v9/weights/v9-c.pt"
