DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali.kwcoco.zip

#sdvc request "$WATCH_DVC_EXPT_DPATH/models/sam/sam_vit_h_4b8939.pth"

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
