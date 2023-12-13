#!/bin/bash
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

#sdvc request "$WATCH_DVC_EXPT_DPATH/models/sam/sam_vit_h_4b8939.pth"

ln -fs "$HOME/data/dvc-repos/shitspotter_dvc" "$HOME/code/shitspotter/shitspotter_dvc"

WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.tasks.sam.predict \
    --input_kwcoco "$KWCOCO_BUNDLE_DPATH"/vali.kwcoco.zip \
    --output_kwcoco "$KWCOCO_BUNDLE_DPATH"/vali-sam.kwcoco.zip \
    --fixed_resolution=None \
    --channels="red|green|blue" \
    --weights_fpath "$WATCH_DVC_EXPT_DPATH/models/sam/sam_vit_h_4b8939.pth" \
    --window_overlap=0.33333 \
    --data_workers="2" \
    --io_workers 0 \
    --assets_dname="sam_feats"


DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.tasks.sam.predict \
    --input_kwcoco "$KWCOCO_BUNDLE_DPATH"/train.kwcoco.zip \
    --output_kwcoco "$KWCOCO_BUNDLE_DPATH"/train-sam.kwcoco.zip \
    --fixed_resolution=None \
    --channels="red|green|blue" \
    --weights_fpath "$WATCH_DVC_EXPT_DPATH/models/sam/sam_vit_h_4b8939.pth" \
    --window_overlap=0.33333 \
    --data_workers="2" \
    --io_workers 0 \
    --assets_dname="sam_feats"


geowatch visualize \
    /data/joncrall/dvc-repos/shitspotter_expt_dvc/shitspotter_dvc/vali-sam.kwcoco.zip \
    --stack=only \
    --channels="red|green|blue,sam.0:3,sam.3:6"

geowatch visualize \
    /data/joncrall/dvc-repos/shitspotter_expt_dvc/shitspotter_dvc/train-sam.kwcoco.zip \
    --stack=only \
    --channels="red|green|blue,sam.0:3,sam.3:6"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
mkdir -p "$DVC_EXPT_DPATH"
#WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


python -m watch.cli.coco_combine_features \
    --src \
        $KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip \
        $KWCOCO_BUNDLE_DPATH/vali-sam.kwcoco.zip \
    --dst \
        $KWCOCO_BUNDLE_DPATH/vali-sam2.kwcoco.zip

python -m watch.cli.coco_combine_features \
    --src \
        $KWCOCO_BUNDLE_DPATH/train.kwcoco.zip \
        $KWCOCO_BUNDLE_DPATH/train-sam.kwcoco.zip \
    --dst \
        $KWCOCO_BUNDLE_DPATH/train-sam2.kwcoco.zip


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WORKDIR="$DVC_EXPT_DPATH/training/$HOSTNAME/$USER"
DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train-sam2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali-sam2.kwcoco.zip
EXPERIMENT_NAME="shitspotter_ooo_sam_v5"

CHANNELS="phone:(red|green|blue,sam.0:3)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=8000000

DDP_WORKAROUND=0 python -m watch.tasks.fusion fit --config "
data:
    sampler_backend        : 'cog'
    select_videos          : $SELECT_VIDEOS
    num_workers            : 8
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '512,512'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : null
    input_resolution      : null
    output_resolution     : null
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    #normalize_perframe     : false
    #normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.1
    modality_dropout       : 0.1
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.2
    quality_threshold      : 0.2
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 4096
    balance_areas          : False
    #sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 8
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 3e-7
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.95
trainer:
    accumulate_grad_batches: 128
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy            : ddp
    limit_val_batches    : 2056
    limit_train_batches  : 20048
    num_sanity_val_steps : 0
    max_epochs           : 3600
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

#batch_plotter:
#    max_items: 8
#    overlay_on_image: False

torch_globals:
    float32_matmul_precision: auto
"

