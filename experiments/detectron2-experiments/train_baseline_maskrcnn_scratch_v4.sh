#!/bin/bash
# References:
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
# https://github.com/facebookresearch/detectron2/issues/2442
export CUDA_VISIBLE_DEVICES="1,"
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
EXPERIMENT_NAME="train_baseline_maskrcnn_scratch_v4"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs5747_1e73d54f.mscoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.mscoco.json
echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"
echo "DEFAULT_ROOT_DIR = $DEFAULT_ROOT_DIR"

echo "
default_root_dir: $DEFAULT_ROOT_DIR
expt_name: $EXPERIMENT_NAME
train_fpath: $TRAIN_FPATH
vali_fpath: $VALI_FPATH
init: noop
cfg:
    DATALOADER:
        NUM_WORKERS: 2
    SOLVER:
        IMS_PER_BATCH: 2   # This is the real 'batch size' commonly known to deep learning people
        BASE_LR: 0.00025   # pick a good LR
        MAX_ITER: 120_000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        STEPS: []          # do not decay learning rate
" > train_config_v4.yaml
cat train_config_v4.yaml
# ~/code/shitspotter/shitspotter/detectron2/fit.py
python -m shitspotter.detectron2.fit --config train_config_v4.yaml


################# TODO: rewrite the eval logic after training is done.


python -m shitspotter.detectron2.predict \
    --checkpoint_fpath /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v3/v_966e49df/model_0119999.pth \
    --src /home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json \
    --dst pred.kwcoco.json \
    --workers=4

kwcoco eval_detections \
    --true_dataset /home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json \
    --pred_dataset pred.kwcoco.json




export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")

test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"

ls "$DEFAULT_ROOT_DIR"/v_*/*.pth -a1
echo "
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0004999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0009999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0014999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0019999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0024999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0029999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0034999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0039999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0044999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0049999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0054999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0059999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0064999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0069999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0074999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0079999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0084999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0089999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0094999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0099999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0104999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0109999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0114999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0119999.pth
" > "$HOME"/code/shitspotter/experiments/detectron_models_v4.yaml

# specified models
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_detectron_evals_v4

kwcoco info "$VALI_FPATH" -g1

python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'
        matrix:
            detectron_pred.checkpoint_fpath:
                 - $HOME/code/shitspotter/experiments/detectron_models_v4.yaml
            detectron_pred.src_fpath:
                - $VALI_FPATH
            detectron_pred.workers: 4
            detectron_pred.write_heatmap: true
            detectron_pred.nms_thresh: 0.5
            detection_eval.__enabled__: 1
            heatmap_eval.__enabled__: 1
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1," --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_detectron_evals_v4
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.detectron_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=1 \
    --io_workers=0 \
    --eval_nodes="
        - detection_evaluation
        - heatmap_eval
    " \
    --stdout_report="
        top_k: 10
        per_group: null
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 0
        show_csv: 0
    "


# TEST dataset results
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
SRC_FPATH=$KWCOCO_BUNDLE_DPATH/test_imgs30_d8988f8c.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_detectron_evals_v4_test
kwcoco info "$SRC_FPATH" -g1
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'
        matrix:
            detectron_pred.checkpoint_fpath:
                 - $HOME/code/shitspotter/experiments/detectron_models_v4.yaml
            detectron_pred.src_fpath:
                - $SRC_FPATH
            detectron_pred.workers: 4
            detectron_pred.write_heatmap: true
            detectron_pred.nms_thresh: 0.5
            detection_eval.__enabled__: 1
            heatmap_eval.__enabled__: 1
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1" --tmux_workers=8 \
    --backend=tmux --skip_existing=1 \
    --run=0
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_detectron_evals_v4_test
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.detectron_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=1 \
    --io_workers=0 \
    --eval_nodes="
        - detection_evaluation
        - heatmap_eval
    " \
    --stdout_report="
        top_k: 10
        per_group: null
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 0
        show_csv: 0
    "


# Estimate training resources
# SeeAlso: ~/code/shitspotter/dev/poc/estimate_train_resources.py
python -c "if 1:
    import ubelt as ub
    import kwutil
    helper = ub.import_module_from_path(ub.Path('~/code/shitspotter/dev/poc/estimate_train_resources.py').expanduser())
    dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v3/v_966e49df')
    files = list(dpath.glob('*'))
    times = [p.stat().st_mtime for p in files]
    min_time = kwutil.datetime.coerce(min(times))
    max_time = kwutil.datetime.coerce(max(times))
    total_delta = max_time - min_time
    print(total_delta)
    row = helper.find_offset_cost(total_delta)
    print(f'row = {ub.urepr(row, nl=1, align=":", precision=2)}')

"
