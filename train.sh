#!/bin/bash
#
#kwcoco stats /home/joncrall/data/dvc-repos/shitspotter_dvc/data.kwcoco.json


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}

inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v1"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=8000000

DDP_WORKAROUND=0 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
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
    batch_size             : 4
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
        arch_name              : smt_it_stm_p16
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
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy            : ddp_find_unused_parameters_true
    limit_val_batches    : 2056
    limit_train_batches  : 20048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

batch_plotter:
    max_items: 8
    overlay_on_image: False

torch_globals:
    float32_matmul_precision: auto

initializer:
    #init: $WATCH_DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch343_step11008.pt
    #init: /data/joncrall/dvc-repos/shitspotter_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_8/checkpoints/epoch=122-step=369-val_loss=13.104.ckpt.ckpt
    #init: /data/joncrall/dvc-repos/shitspotter_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_11/checkpoints/epoch=121-step=366-val_loss=13.427.ckpt.ckpt
    #init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_13/checkpoints/last.ckpt
    #init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_14/checkpoints/last.ckpt
    #init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_15/checkpoints/last.ckpt
    init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_16/checkpoints/epoch=351-step=4224-val_loss=2.518.ckpt.ckpt
"


geowatch repackage "$HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_16/checkpoints/last.ckpt"
PACKAGE_FPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_16/checkpoints/last.pt


DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

#geowatch repackage /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/Ooo/joncrall/ShitSpotter/runs/shitspotter_ooo_scratch_v1/lightning_logs/version_3/checkpoints/last.ckpt
PRED_FPATH=$DVC_EXPT_DPATH/shitspotter-test-v2/pred.kwcoco.zip


python -m watch.tasks.fusion.predict \
    --package_fpath="$PACKAGE_FPATH" \
    --test_dataset="$VALI_FPATH"  \
    --pred_dataset="$PRED_FPATH" \
    --select_images=".id < 10" \
    --draw_batches=1 \
    --device="0,"


geowatch visualize /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/shitspotter-test-v2/pred.kwcoco.zip --smart



export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v5"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=8000000

DDP_WORKAROUND=1 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '256,256'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : null
    input_resolution      : null
    output_resolution     : null
    neg_to_pos_ratio       : 1.0
    batch_size             : 12
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
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : False
    #sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        perterb_scale          : 1e-8
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
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    #devices              : 0,
    devices              : 0,1
    strategy            : ddp
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

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $WATCH_DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch343_step11008.pt
    #init: /data/joncrall/dvc-repos/shitspotter_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_8/checkpoints/epoch=122-step=369-val_loss=13.104.ckpt.ckpt
    #init: /data/joncrall/dvc-repos/shitspotter_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_11/checkpoints/epoch=121-step=366-val_loss=13.427.ckpt.ckpt
    #init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_13/checkpoints/last.ckpt
    #init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_14/checkpoints/last.ckpt
    #init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_15/checkpoints/last.ckpt
    #init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_16/checkpoints/epoch=351-step=4224-val_loss=2.518.ckpt.ckpt
    #init: $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_16/checkpoints/last.pt
    #init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v3/lightning_logs/version_6/checkpoints/epoch=419-step=1145-val_loss=1.904.ckpt.ckpt
    #init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_16/checkpoints/epoch=351-step=4224-val_loss=2.518.ckpt.ckpt
" --ckpt_path=/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v5/lightning_logs/version_15/checkpoints/last.ckpt

#\
#--ckpt_path=/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v3/lightning_logs/version_7/checkpoints/last.ckpt
#\ --ckpt_path=/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v3/lightning_logs/version_5/checkpoints/epoch=282-step=734-val_loss=1.850.ckpt.ckpt
    #--ckpt_path=/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v3/lightning_logs/version_4/checkpoints/epoch=265-step=683-val_loss=1.832.ckpt.ckpt



export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v6"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=8000000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
#STRATEGY=ddp

DDP_WORKAROUND=$DDP_WORKAROUND python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
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
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : False
    #sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
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
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
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

torch_globals:
    float32_matmul_precision: auto

initializer:
    #init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/lightning_logs/version_17/checkpoints/epoch=283-step=3408-val_loss=2.590.ckpt.ckpt
    #init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v6/lightning_logs/version_2/checkpoints/last.ckpt
    #init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v6/lightning_logs/version_5/checkpoints/last.ckpt
    #init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v6/lightning_logs/version_14/checkpoints/last.ckpt
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v6/lightning_logs/version_15/checkpoints/last.ckpt
" --ckpt_path="$DEFAULT_ROOT_DIR"/lightning_logs/version_16/checkpoints/last.ckpt

#\
#--ckpt_path=/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v3/lightning_logs/version_7/checkpoints/last.ckpt
#\ --ckpt_path=/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v3/lightning_logs/version_5/checkpoints/epoch=282-step=734-val_loss=1.850.ckpt.ckpt
    #--ckpt_path=/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v3/lightning_logs/version_4/checkpoints/epoch=265-step=683-val_loss=1.832.ckpt.ckpt



export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v7"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=8000000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
#STRATEGY=ddp

DDP_WORKAROUND=$DDP_WORKAROUND python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
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
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : False
    #sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
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
    accumulate_grad_batches: 256
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
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

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v6/lightning_logs/version_17/checkpoints/last.ckpt
" --ckpt_path=/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v7/lightning_logs/version_0/checkpoints/last.ckpt



export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v7"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=8000000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
#STRATEGY=ddp

DDP_WORKAROUND=$DDP_WORKAROUND python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
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
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : False
    #sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
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
    accumulate_grad_batches: 256
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
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

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v6/lightning_logs/version_17/checkpoints/last.ckpt
" --ckpt_path=/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v7/lightning_logs/version_0/checkpoints/last.ckpt


export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v8"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-2
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
MAX_STEPS=8000000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
    max_version = max(version_to_checkpoints)
    candidates = version_to_checkpoints[max_version]
    checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
    chosen = checkpoints[-1]
    print(chosen)
")

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
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
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : True
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : False
    #sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
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
    accumulate_grad_batches: 512
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
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

torch_globals:
    float32_matmul_precision: auto

initializer:
    #init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v7/lightning_logs/version_0/checkpoints/last.ckpt
    init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v8/lightning_logs/version_2/checkpoints/epoch=1-step=14-val_loss=0.846.ckpt.pt
" --ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v9"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
MAX_STEPS=8000000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
    max_version = max(version_to_checkpoints)
    candidates = version_to_checkpoints[max_version]
    checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
    chosen = checkpoints[-1]
    print(chosen)
")

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
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
    normalize_perframe     : false
    normalize_peritem      : false
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
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : True
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : False
    #sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.ExponentialLR
  init_args:
    gamma: 0.96
trainer:
    accumulate_grad_batches: 32
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 20048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v8/lightning_logs/version_5/checkpoints/epoch=545-step=4368-val_loss=0.398.ckpt.ckpt
"
#\
#--ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v010"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
MAX_STEPS=8000000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
    max_version = max(version_to_checkpoints)
    candidates = version_to_checkpoints[max_version]
    checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
    chosen = checkpoints[-1]
    print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
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
    normalize_perframe     : false
    normalize_peritem      : false
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
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : True
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : False
    #sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.ExponentialLR
  init_args:
    gamma: 0.96
trainer:
    accumulate_grad_batches: 1024
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 20048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v8/lightning_logs/version_5/checkpoints/epoch=545-step=4368-val_loss=0.398.ckpt.ckpt
" --ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v011"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '384,384'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.5
    input_resolution      : 0.5
    output_resolution     : 0.5
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : false
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
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : True
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max        : $MAX_STEPS
    eta_min      : $ETA_MIN
    verbose      : 1
trainer:
    accumulate_grad_batches: 8
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 20048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v010/lightning_logs/version_2/checkpoints/epoch=147-step=592-val_loss=0.355.ckpt.ckpt
" --ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v012"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.25
    input_resolution      : 0.25
    output_resolution     : 0.25
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max        : $MAX_STEPS
    eta_min      : $ETA_MIN
trainer:
    accumulate_grad_batches: 128
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v011/lightning_logs/version_9/checkpoints/epoch=347-step=145812-val_loss=0.491.ckpt.ckpt
" # --ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v013"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.5
    input_resolution      : 0.5
    output_resolution     : 0.5
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max        : $MAX_STEPS
    eta_min      : $ETA_MIN
trainer:
    accumulate_grad_batches: 512
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v012/lightning_logs/version_0/checkpoints/last.ckpt
" # --ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v014"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.5
    input_resolution      : 0.5
    output_resolution     : 0.5
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max        : $MAX_STEPS
    eta_min      : $ETA_MIN
trainer:
    accumulate_grad_batches: 1024
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v013/lightning_logs/version_0/checkpoints/epoch=294-step=2360-val_loss=0.587.ckpt.ckpt
" # --ckpt_path="$PREV_CHECKPOINT"



# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v014"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 1.0
    input_resolution      : 1.0
    output_resolution     : 1.0
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max        : $MAX_STEPS
    eta_min      : $ETA_MIN
trainer:
    accumulate_grad_batches: 1024
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v014/lightning_logs/version_0/checkpoints/epoch=250-step=1004-val_loss=0.550.ckpt.ckpt
" --ckpt_path="$PREV_CHECKPOINT"




# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v015"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.001)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 1.0
    input_resolution      : 1.0
    output_resolution     : 1.0
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max        : $MAX_STEPS
    eta_min      : $ETA_MIN
trainer:
    accumulate_grad_batches: 1024
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v014/lightning_logs/version_3/checkpoints/epoch=309-step=1550-val_loss=0.352.ckpt.ckpt
"
#--ckpt_path="$PREV_CHECKPOINT"
#

# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v016"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.001)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 1.0
    input_resolution      : 1.0
    output_resolution     : 1.0
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 12
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max        : $MAX_STEPS
    eta_min      : $ETA_MIN
trainer:
    accumulate_grad_batches: 250
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v015/lightning_logs/version_1/checkpoints/epoch=30-step=155-val_loss=0.417.ckpt.ckpt
"
#--ckpt_path="$PREV_CHECKPOINT"
#
#sm_it_sm_s12
#
#
# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v016"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.001)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 1.0
    input_resolution      : 1.0
    output_resolution     : 1.0
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 12
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max        : $MAX_STEPS
    eta_min      : $ETA_MIN
trainer:
    accumulate_grad_batches: 250
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v016/lightning_logs/version_6/checkpoints/epoch=3-step=20-val_loss=1.938.ckpt.ckpt
"
#--ckpt_path="$PREV_CHECKPOINT"
#
#sm_it_sm_s12


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v017"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.001)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 1.0
    input_resolution      : 1.0
    output_resolution     : 1.0
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-7
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
    accumulate_grad_batches: 32
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v016/lightning_logs/version_8/checkpoints/epoch=78-step=395-val_loss=0.862.ckpt.pt
"
#--ckpt_path="$PREV_CHECKPOINT"
#
#sm_it_sm_s12

# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v018"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.001)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.5
    input_resolution      : 0.5
    output_resolution     : 0.5
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-7
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
    accumulate_grad_batches: 32
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v017/lightning_logs/version_1/checkpoints/epoch=82-step=3320-val_loss=0.968.ckpt.ckpt
"
#--ckpt_path="$PREV_CHECKPOINT"
#
#sm_it_sm_s12


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v019"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.0001)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
# TODO: find a good way to set this number I think it matters wrt to the
# OneCycle scheduler.
MAX_STEPS=16000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.25
    input_resolution      : 0.25
    output_resolution     : 0.25
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : 1e-7
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
    accumulate_grad_batches: 32
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v018/lightning_logs/version_1/checkpoints/epoch=147-step=7844-val_loss=0.353.ckpt.ckpt
"
#--ckpt_path="$PREV_CHECKPOINT"
#
#sm_it_sm_s12

# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v020"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
# TODO: find a good way to set this number I think it matters wrt to the
# OneCycle scheduler.
MAX_STEPS=16000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.25
    input_resolution      : 0.25
    output_resolution     : 0.25
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : $PERTERB_SCALE
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
    verbose : true
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v019/lightning_logs/version_0/checkpoints/epoch=42-step=2279-val_loss=0.167.ckpt.ckpt
" --ckpt_path="$PREV_CHECKPOINT"
#
#sm_it_sm_s12


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v021"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
# TODO: find a good way to set this number I think it matters wrt to the
# OneCycle scheduler.
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
#STRATEGY=ddp

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.25
    input_resolution      : 0.25
    output_resolution     : 0.25
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : $PERTERB_SCALE
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
    verbose : true
trainer:
    accumulate_grad_batches: 128
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 720
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v020/lightning_logs/version_2/checkpoints/epoch=0358-step=014360-val_loss=0.749.ckpt.pt
"
#--ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_v022"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
# TODO: find a good way to set this number I think it matters wrt to the
# OneCycle scheduler.
MAX_STEPS=8000

DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"


# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.25
    input_resolution      : 0.25
    output_resolution     : 0.25
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : $PERTERB_SCALE
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
    verbose : true
trainer:
    accumulate_grad_batches: 128
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : 5012
    num_sanity_val_steps : 0
    max_epochs           : 720
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_v020/lightning_logs/version_2/checkpoints/epoch=0358-step=014360-val_loss=0.749.ckpt.pt
"
#--ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_scratch_v023"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"


# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"


MAX_STEPS=10000
MAX_EPOCHS=1280
ITEMS_PER_EPOCH=2000
ACCUMULATE_GRAD_BATCHES=128
BATCH_SIZE=2

python -c "if 1:
    import sympy
    import ubelt as ub
    limit_train_batches, batch_size, accumulate_grad_batches, max_epochs, MAX_STEPS = sympy.symbols(
        'limit_train_batches, batch_size, accumulate_grad_batches, max_epochs, MAX_STEPS')

    subs = {
        limit_train_batches: $ITEMS_PER_EPOCH,
        batch_size: $BATCH_SIZE,
        accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES,
        max_epochs: $MAX_EPOCHS,
        MAX_STEPS: $MAX_STEPS,
    }

    effective_batch_size = accumulate_grad_batches * batch_size
    #steps_per_epoch = sympy.floor(limit_train_batches / effective_batch_size)
    steps_per_epoch = limit_train_batches / effective_batch_size
    total_steps = max_epochs * steps_per_epoch
    total_steps.subs(subs)

    effective_batch_size_ = effective_batch_size.subs(subs).evalf()

    print(f'{effective_batch_size_=}')

    # The training progress iterator should show this number as the total number
    import math
    train_epoch_prog_iters = math.ceil((limit_train_batches / batch_size).subs(subs).evalf())

    diff = MAX_STEPS - total_steps
    curr_diff = diff.subs(subs)
    print(f'curr_diff={curr_diff.evalf()}')

    if curr_diff > 0:
        print('Not enough total steps to fill MAX_STEPS')
    else:
        print('MAX STEPS will stop training short')

    for k, v in subs.items():
        print('--- Possible Adjustment For ---')
        print(k)
        tmp_subs = (ub.udict(subs) - {k})
        solutions = sympy.solve(diff.subs(tmp_subs), k)
        solutions = [s.evalf() for s in solutions]
        print(solutions)
"


DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.25
    input_resolution      : 0.25
    output_resolution     : 0.25
    neg_to_pos_ratio       : 1.0
    batch_size             : $BATCH_SIZE
    normalize_perframe     : false
    normalize_peritem      : false
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : $PERTERB_SCALE
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
    verbose : true
trainer:
    accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_val_batches    : 2056
    limit_train_batches  : $ITEMS_PER_EPOCH
    num_sanity_val_steps : 0
    max_epochs: $MAX_EPOCHS
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: noop
"
#--ckpt_path="$PREV_CHECKPOINT"


# ----
# LR was too high or batch size too small, loss went pretty low spiked and
# never recovered.

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_scratch_v024"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"


MAX_STEPS=10000
MAX_EPOCHS=1280
TRAIN_BATCHES_PER_EPOCH=2048
ACCUMULATE_GRAD_BATCHES=128
BATCH_SIZE=2
TRAIN_ITEMS_PER_EPOCH=$(python -c "print($TRAIN_BATCHES_PER_EPOCH * $BATCH_SIZE)")
echo "TRAIN_ITEMS_PER_EPOCH = $TRAIN_ITEMS_PER_EPOCH"

python -c "if 1:
    import sympy
    import ubelt as ub
    train_items_per_epoch, train_batches_per_epoch, batch_size, accumulate_grad_batches, max_epochs, MAX_STEPS = sympy.symbols(
        'train_items_per_epoch, train_batches_per_epoch, batch_size, accumulate_grad_batches, max_epochs, MAX_STEPS')

    subs = {
        train_batches_per_epoch: $TRAIN_BATCHES_PER_EPOCH,
        train_items_per_epoch: $TRAIN_ITEMS_PER_EPOCH,
        batch_size: $BATCH_SIZE,
        accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES,
        max_epochs: $MAX_EPOCHS,
        MAX_STEPS: $MAX_STEPS,
    }

    effective_batch_size = accumulate_grad_batches * batch_size
    #steps_per_epoch = sympy.floor(train_batches_per_epoch / effective_batch_size)
    steps_per_epoch = train_batches_per_epoch / effective_batch_size
    total_steps = max_epochs * steps_per_epoch
    total_steps.subs(subs)

    effective_batch_size_ = effective_batch_size.subs(subs).evalf()

    print(f'{effective_batch_size_=}')

    # The training progress iterator should show this number as the total number
    import math
    train_epoch_prog_iters = math.ceil((train_batches_per_epoch / batch_size).subs(subs).evalf())

    diff = MAX_STEPS - total_steps
    curr_diff = diff.subs(subs)
    print(f'curr_diff={curr_diff.evalf()}')

    if curr_diff > 0:
        print('Not enough total steps to fill MAX_STEPS')
    else:
        print('MAX STEPS will stop training short')

    for k, v in subs.items():
        print('--- Possible Adjustment For ---')
        print(k)
        tmp_subs = (ub.udict(subs) - {k})
        solutions = sympy.solve(diff.subs(tmp_subs), k)
        solutions = [s.evalf() for s in solutions]
        print(solutions)
"


# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -c "if 1:
    import ubelt as ub
    root_dir = ub.Path('$DEFAULT_ROOT_DIR')
    checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
    if len(checkpoints) == 0:
        print('None')
    else:
        version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
        max_version = max(version_to_checkpoints)
        candidates = version_to_checkpoints[max_version]
        checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
        chosen = checkpoints[-1]
        print(chosen)
")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"


DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.25
    input_resolution      : 0.25
    output_resolution     : 0.25
    neg_to_pos_ratio       : 1.0
    batch_size             : $BATCH_SIZE
    normalize_perframe     : false
    normalize_peritem      : false
    max_items_per_epoch    : $TRAIN_ITEMS_PER_EPOCH
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : $PERTERB_SCALE
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
    pct_start: 0.3
trainer:
    accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_train_batches  : $TRAIN_BATCHES_PER_EPOCH
    limit_val_batches    : 2056
    log_every_n_steps    : 1
    check_val_every_n_epoch: 1
    enable_checkpointing: true
    enable_model_summary: true
    num_sanity_val_steps : 0
    max_epochs: $MAX_EPOCHS
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: noop
" --ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_scratch_v025"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"


MAX_STEPS=10000
MAX_EPOCHS=1280
TRAIN_BATCHES_PER_EPOCH=1024
ACCUMULATE_GRAD_BATCHES=128
BATCH_SIZE=2
TRAIN_ITEMS_PER_EPOCH=$(python -c "print($TRAIN_BATCHES_PER_EPOCH * $BATCH_SIZE)")
echo "TRAIN_ITEMS_PER_EPOCH = $TRAIN_ITEMS_PER_EPOCH"

python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"


# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -m geowatch.cli.experimental.find_recent_checkpoint --default_root_dir="$DEFAULT_ROOT_DIR")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"


DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.25
    input_resolution      : 0.25
    output_resolution     : 0.25
    neg_to_pos_ratio       : 1.0
    batch_size             : $BATCH_SIZE
    normalize_perframe     : false
    normalize_peritem      : false
    max_items_per_epoch    : $TRAIN_ITEMS_PER_EPOCH
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : $PERTERB_SCALE
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
    pct_start: 0.3
trainer:
    accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_train_batches  : $TRAIN_BATCHES_PER_EPOCH
    limit_val_batches    : 2056
    log_every_n_steps    : 1
    check_val_every_n_epoch: 1
    enable_checkpointing: true
    enable_model_summary: true
    num_sanity_val_steps : 0
    max_epochs: $MAX_EPOCHS
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: noop
" \
#--ckpt_path="$PREV_CHECKPOINT"


# ----

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WATCH_DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH


TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="shitspotter_from_v025_upscale_v026"

CHANNELS="phone:(red|green|blue)"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
ETA_MIN=$(python -c "print($TARGET_LR * 0.0001)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"


MAX_STEPS=10000
MAX_EPOCHS=1280
TRAIN_BATCHES_PER_EPOCH=1024
ACCUMULATE_GRAD_BATCHES=128
BATCH_SIZE=2
TRAIN_ITEMS_PER_EPOCH=$(python -c "print($TRAIN_BATCHES_PER_EPOCH * $BATCH_SIZE)")
echo "TRAIN_ITEMS_PER_EPOCH = $TRAIN_ITEMS_PER_EPOCH"

python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"


# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT=$(python -m geowatch.cli.experimental.find_recent_checkpoint --default_root_dir="$DEFAULT_ROOT_DIR")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"


DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '416,416'
    time_steps             : 1
    time_sampling          : uniform
    #time_kernel            : '[0.0s,]'
    window_resolution     : 0.5
    input_resolution      : 0.5
    output_resolution     : 0.5
    neg_to_pos_ratio       : 1.0
    batch_size             : $BATCH_SIZE
    normalize_perframe     : false
    normalize_peritem      : false
    max_items_per_epoch    : $TRAIN_ITEMS_PER_EPOCH
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.0
    channel_dropout        : 0.05
    modality_dropout       : 0.05
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 8096
    balance_areas          : false
model:
    class_path: MultimodalTransformer
    init_args:
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_s12
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 4
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        multimodal_reduce      : max
        continual_learning     : true
        perterb_scale          : $PERTERB_SCALE
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
    pct_start: 0.3
trainer:
    accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_train_batches  : $TRAIN_BATCHES_PER_EPOCH
    limit_val_batches    : 2056
    log_every_n_steps    : 1
    check_val_every_n_epoch: 1
    enable_checkpointing: true
    enable_model_summary: true
    num_sanity_val_steps : 0
    max_epochs: $MAX_EPOCHS
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_v025/lightning_logs/version_2/checkpoints/epoch=1264-step=005060-val_loss=0.545.ckpt.ckpt
"
#--ckpt_path="$PREV_CHECKPOINT"
