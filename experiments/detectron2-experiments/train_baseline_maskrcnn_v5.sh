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
EXPERIMENT_NAME="train_baseline_maskrcnn_v5"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

we pyenv3.11.9

#kwcoco tables $KWCOCO_BUNDLE_DPATH/train_imgs5747_1e73d54f.mscoco.json -a 1
#kwcoco tables $KWCOCO_BUNDLE_DPATH/train_imgs9270_f2b4b17d-simple-poop-only.json -a 1
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.mscoco.json

#TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs9270_f2b4b17d-simple-poop-only.json
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs1258_fe7f7dfe-simple-poop-only.json

#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
#ORIG_TRAIN_FPATH=$DVC_DATA_DPATH/train_imgs9270_f2b4b17d-simple-poop-only.json
#ORIG_VALI_FPATH=$DVC_DATA_DPATH/vali_imgs1258_fe7f7dfe-simple-poop-only.json
TRAIN_FPATH=$DVC_DATA_DPATH/train_imgs9270_f2b4b17d-simple-poop-only.json
VALI_FPATH=$DVC_DATA_DPATH/vali_imgs1258_fe7f7dfe-simple-poop-only.json

#cd $DVC_DATA_DPATH
#python -c "if 1:
#    import kwcoco
#    dset = kwcoco.CocoDataset('train_imgs9270_f2b4b17d-simple-poop-only.json')
#    for obj in dset.annots().objs:
#        if 'segmenation' in obj:
#            print(1)
#        assert 'segmenation' in obj

#"
#kwcoco conform --legacy=True --src "$ORIG_TRAIN_FPATH" --dst "$TRAIN_FPATH"
#kwcoco conform --legacy=True --src "$ORIG_VALI_FPATH" --dst "$VALI_FPATH"


echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"
echo "DEFAULT_ROOT_DIR = $DEFAULT_ROOT_DIR"

echo "
default_root_dir: $DEFAULT_ROOT_DIR
expt_name: train_baseline_maskrcnn_v5
train_fpath: $TRAIN_FPATH
vali_fpath: $VALI_FPATH
" > train_config_v5.yaml
cat train_config_v5.yaml
python -m shitspotter.detectron2.fit --config train_config_v5.yaml


export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")

test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"

#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs1258_fe7f7dfe.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_evals


echo "
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0119999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0109999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0079999.pth
- /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0069999.pth
" > "$HOME"/code/shitspotter/experiments/detectron_models_v5.yaml

# specified models
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH

# Validate on the older set for comparability
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.mscoco.json
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs1258_fe7f7dfe.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_detectron_evals_v5

we pyenv3.11.9
kwcoco info "$VALI_FPATH" -g1

python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'
        matrix:
            detectron_pred.checkpoint_fpath:
                 - $HOME/code/shitspotter/experiments/detectron_models_v5.yaml
            detectron_pred.src_fpath:
                - $VALI_FPATH
            detectron_pred.workers: 4
            detectron_pred.write_heatmap: true
            detectron_pred.nms_thresh: 0.5
            detection_eval.__enabled__: 1
            heatmap_eval.__enabled__: 1
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_detectron_evals_v5
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

__results__="

Varied Basis: = {
    'params.detectron_pred.src_fpath': {
        '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip': 4,
        '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json': 4,
    },
    'params.detectron_pred.checkpoint_fpath': {
        '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0109999.pth': 2,
        '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0119999.pth': 2,
        '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0079999.pth': 2,
        '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0069999.pth': 2,
    },
}
Constant Params: {
    'params.detectron_pred.workers': 4,
    'params.detectron_pred.write_heatmap': True,
    'params.detectron_pred.nms_thresh': 0.5,
}
Varied Parameter LUT: {
    'rjsyedwtcfdg': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0109999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip',
    },
    'kmnyhfskuhku': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0119999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip',
    },
    'ofqpujttkxsd': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0119999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json',
    },
    'lbkkqvxnhslb': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0109999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json',
    },
    'eqfygvkwiamm': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0079999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip',
    },
    'owzhkyriynjp': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0069999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip',
    },
    'hmujigquqtiy': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0069999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json',
    },
    'ttqsawfahqtu': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0079999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json',
    },
}
---
Top 8 / 8 for detection_evaluation, unknown
                                                                                                                                                                 fpath                 node region_id param_hashid       ap      auc  max_f1_f1  max_f1_tpr  max_f1_ppv
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/detection_evaluation/detection_evaluation_id_bdf92cdd/detect_metrics.json detection_evaluation   unknown rjsyedwtcfdg 0.529497 0.616593   0.603077    0.567020    0.644031
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/detection_evaluation/detection_evaluation_id_4eba5e3b/detect_metrics.json detection_evaluation   unknown kmnyhfskuhku 0.544668 0.590759   0.613861    0.538091    0.714469
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/detection_evaluation/detection_evaluation_id_f76938ea/detect_metrics.json detection_evaluation   unknown ofqpujttkxsd 0.552388 0.584068   0.619308    0.542265    0.721868
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/detection_evaluation/detection_evaluation_id_da02bf11/detect_metrics.json detection_evaluation   unknown lbkkqvxnhslb 0.556413 0.620511   0.620939    0.548644    0.715177
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/detection_evaluation/detection_evaluation_id_3feffece/detect_metrics.json detection_evaluation   unknown eqfygvkwiamm 0.558251 0.634876   0.614386    0.547734    0.699507
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/detection_evaluation/detection_evaluation_id_8e7bc909/detect_metrics.json detection_evaluation   unknown owzhkyriynjp 0.558823 0.669334   0.607368    0.556413    0.668598
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/detection_evaluation/detection_evaluation_id_0563d067/detect_metrics.json detection_evaluation   unknown hmujigquqtiy 0.563506 0.656020   0.613475    0.551834    0.690619
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/detection_evaluation/detection_evaluation_id_5d1515b1/detect_metrics.json detection_evaluation   unknown ttqsawfahqtu 0.565456 0.624575   0.625115    0.543860    0.734914

---
Top 8 / 8 for detection_evaluation, macro_01_8d24ce = frozenset({'unknown'})
                node region_id param_hashid       ap      auc  max_f1_f1  max_f1_tpr  max_f1_ppv
detection_evaluation   unknown rjsyedwtcfdg 0.529497 0.616593   0.603077    0.567020    0.644031
detection_evaluation   unknown kmnyhfskuhku 0.544668 0.590759   0.613861    0.538091    0.714469
detection_evaluation   unknown ofqpujttkxsd 0.552388 0.584068   0.619308    0.542265    0.721868
detection_evaluation   unknown lbkkqvxnhslb 0.556413 0.620511   0.620939    0.548644    0.715177
detection_evaluation   unknown eqfygvkwiamm 0.558251 0.634876   0.614386    0.547734    0.699507
detection_evaluation   unknown owzhkyriynjp 0.558823 0.669334   0.607368    0.556413    0.668598
detection_evaluation   unknown hmujigquqtiy 0.563506 0.656020   0.613475    0.551834    0.690619
detection_evaluation   unknown ttqsawfahqtu 0.565456 0.624575   0.625115    0.543860    0.734914


Varied Basis: = {
    'params.detectron_pred.src_fpath': {
        '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip': 4,
        '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json': 4,
    },
    'params.detectron_pred.checkpoint_fpath': {
        '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0069999.pth': 2,
        '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0109999.pth': 2,
        '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0119999.pth': 2,
        '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0079999.pth': 2,
    },
}
Constant Params: {
    'params.detectron_pred.workers': 4,
    'params.detectron_pred.write_heatmap': True,
    'params.detectron_pred.nms_thresh': 0.5,
}
Varied Parameter LUT: {
    'owzhkyriynjp': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0069999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip',
    },
    'rjsyedwtcfdg': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0109999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip',
    },
    'kmnyhfskuhku': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0119999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip',
    },
    'eqfygvkwiamm': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0079999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs1258_fe7f7dfe.kwcoco.zip',
    },
    'hmujigquqtiy': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0069999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json',
    },
    'lbkkqvxnhslb': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0109999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json',
    },
    'ttqsawfahqtu': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0079999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json',
    },
    'ofqpujttkxsd': {
        'params.detectron_pred.checkpoint_fpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0119999.pth',
        'params.detectron_pred.src_fpath': '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json',
    },
}
---
Top 4 / 4 for heatmap_eval, vali_imgs1258_fe7f7dfe
                                                                                                                                           fpath         node              region_id param_hashid  salient_AP  salient_AUC  salient_maxF1_F1  salient_maxF1_tpr  salient_maxF1_thresh
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/heatmap_eval/heatmap_eval_id_f254d2cb/pxl_eval.json heatmap_eval vali_imgs1258_fe7f7dfe owzhkyriynjp    0.773120     0.896917          0.755596           0.743508              0.965787
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/heatmap_eval/heatmap_eval_id_ea466ea2/pxl_eval.json heatmap_eval vali_imgs1258_fe7f7dfe rjsyedwtcfdg    0.783116     0.877119          0.761454           0.739525              0.956989
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/heatmap_eval/heatmap_eval_id_b6759d55/pxl_eval.json heatmap_eval vali_imgs1258_fe7f7dfe kmnyhfskuhku    0.808933     0.881396          0.780034           0.759067              0.971652
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/heatmap_eval/heatmap_eval_id_0b5b53bd/pxl_eval.json heatmap_eval vali_imgs1258_fe7f7dfe eqfygvkwiamm    0.810217     0.880459          0.789397           0.758903              0.964809

---
Top 4 / 4 for heatmap_eval, vali_imgs691_99b22ad0
                                                                                                                                           fpath         node             region_id param_hashid  salient_AP  salient_AUC  salient_maxF1_F1  salient_maxF1_tpr  salient_maxF1_thresh
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/heatmap_eval/heatmap_eval_id_8badd950/pxl_eval.json heatmap_eval vali_imgs691_99b22ad0 hmujigquqtiy    0.804176     0.902113          0.778893           0.749568              0.959922
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/heatmap_eval/heatmap_eval_id_974bc38c/pxl_eval.json heatmap_eval vali_imgs691_99b22ad0 lbkkqvxnhslb    0.817926     0.884511          0.783847           0.743055              0.956989
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/heatmap_eval/heatmap_eval_id_07cbccb2/pxl_eval.json heatmap_eval vali_imgs691_99b22ad0 ttqsawfahqtu    0.827729     0.889015          0.810099           0.752716              0.973607
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v5/eval/flat/heatmap_eval/heatmap_eval_id_e45008d1/pxl_eval.json heatmap_eval vali_imgs691_99b22ad0 ofqpujttkxsd    0.850768     0.883893          0.809498           0.827684              0.885630

---
Top 4 / 4 for heatmap_eval, macro_01_d61c16 = frozenset({'vali_imgs1258_fe7f7dfe'})
        node              region_id param_hashid  salient_AP  salient_AUC  salient_maxF1_F1  salient_maxF1_tpr  salient_maxF1_thresh
heatmap_eval vali_imgs1258_fe7f7dfe owzhkyriynjp    0.773120     0.896917          0.755596           0.743508              0.965787
heatmap_eval vali_imgs1258_fe7f7dfe rjsyedwtcfdg    0.783116     0.877119          0.761454           0.739525              0.956989
heatmap_eval vali_imgs1258_fe7f7dfe kmnyhfskuhku    0.808933     0.881396          0.780034           0.759067              0.971652
heatmap_eval vali_imgs1258_fe7f7dfe eqfygvkwiamm    0.810217     0.880459          0.789397           0.758903              0.964809

---
Top 4 / 4 for heatmap_eval, macro_01_6e26b3 = frozenset({'vali_imgs691_99b22ad0'})
        node             region_id param_hashid  salient_AP  salient_AUC  salient_maxF1_F1  salient_maxF1_tpr  salient_maxF1_thresh
heatmap_eval vali_imgs691_99b22ad0 hmujigquqtiy    0.804176     0.902113          0.778893           0.749568              0.959922
heatmap_eval vali_imgs691_99b22ad0 lbkkqvxnhslb    0.817926     0.884511          0.783847           0.743055              0.956989
heatmap_eval vali_imgs691_99b22ad0 ttqsawfahqtu    0.827729     0.889015          0.810099           0.752716              0.973607
heatmap_eval vali_imgs691_99b22ad0 ofqpujttkxsd    0.850768     0.883893          0.809498           0.827684              0.885630

No model columns are availble
/home/joncrall/code/kwutil/kwutil/util_time.py:1072: UserWarning: warning: pytimeparse fallback
  warnings.warn('warning: pytimeparse fallback')
/home/joncrall/code/kwutil/kwutil/util_time.py:1072: UserWarning: warning: pytimeparse fallback
  warnings.warn('warning: pytimeparse fallback')
             node   resource  num       total        mean
0  detectron_pred       time    8   1.27 hour   0.16 hour
1  detectron_pred  emissions    8  0.08 CO2Kg  0.01 CO2Kg
2  detectron_pred     energy    8    0.37 kWh    0.05 kWh
,node,resource,num,total,mean
0,detectron_pred,time,8,1.27 hour,0.16 hour
1,detectron_pred,emissions,8,0.08 CO2Kg,0.01 CO2Kg
2,detectron_pred,energy,8,0.37 kWh,0.05 kWh

\begin{tabular}{llrll}
\toprule
node & resource & num & total & mean \\
\midrule
detectron_pred & time & 8 & 1.27 hour & 0.16 hour \\
detectron_pred & emissions & 8 & 0.08 CO2Kg & 0.01 CO2Kg \\
detectron_pred & energy & 8 & 0.37 kWh & 0.05 kWh \\
\bottomrule
\end{tabular}

             node  resource  num            total             mean
0  detectron_pred  duration    8  0 days 01:16:29  0 days 00:09:34
1  detectron_pred    co2_kg    8         0.077154         0.009644
2  detectron_pred       kwh    8         0.366567         0.045821
/home/joncrall/code/kwutil/kwutil/util_time.py:1072: UserWarning: warning: pytimeparse fallback
  warnings.warn('warning: pytimeparse fallback')
/home/joncrall/code/kwutil/kwutil/util_time.py:1072: UserWarning: warning: pytimeparse fallback
  warnings.warn('warning: pytimeparse fallback')
             node   resource  num       total        mean
0    heatmap_eval       time    8   1.64 hour   0.21 hour
1  detectron_pred       time    8   1.27 hour   0.16 hour
2  detectron_pred  emissions    8  0.08 CO2Kg  0.01 CO2Kg
3  detectron_pred     energy    8    0.37 kWh    0.05 kWh
,node,resource,num,total,mean
0,heatmap_eval,time,8,1.64 hour,0.21 hour
1,detectron_pred,time,8,1.27 hour,0.16 hour
2,detectron_pred,emissions,8,0.08 CO2Kg,0.01 CO2Kg
3,detectron_pred,energy,8,0.37 kWh,0.05 kWh

\begin{tabular}{llrll}
\toprule
node & resource & num & total & mean \\
\midrule
heatmap_eval & time & 8 & 1.64 hour & 0.21 hour \\
detectron_pred & time & 8 & 1.27 hour & 0.16 hour \\
detectron_pred & emissions & 8 & 0.08 CO2Kg & 0.01 CO2Kg \\
detectron_pred & energy & 8 & 0.37 kWh & 0.05 kWh \\
\bottomrule
\end{tabular}

             node  resource  num            total             mean
0    heatmap_eval  duration    8  0 days 01:38:29  0 days 00:12:19
1  detectron_pred  duration    8  0 days 01:16:29  0 days 00:09:34
2  detectron_pred    co2_kg    8         0.077154         0.009644
3  detectron_pred       kwh    8         0.366567         0.045821
"


# Estimate training resources
# SeeAlso: ~/code/shitspotter/dev/poc/estimate_train_resources.py
python -c "if 1:
    import ubelt as ub
    import kwutil
    helper = ub.import_module_from_path(ub.Path('~/code/shitspotter/dev/poc/estimate_train_resources.py').expanduser())
    dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525')
    files = list(dpath.glob('*'))
    times = [p.stat().st_mtime for p in files]
    min_time = kwutil.datetime.coerce(min(times))
    max_time = kwutil.datetime.coerce(max(times))
    total_delta = max_time - min_time
    print(total_delta)
    row = helper.find_offset_cost(total_delta)
    print(f'row = {ub.urepr(row, nl=1, align=chr(58), precision=2)}')

"

__result__="
9:21:32.578234
kWh:  3.2288720807583338 hour * kilowatt
0.7 CO2 kg
cost_to_offset = $ 0.01
row = {
    'total_electricity'   : 3.23 kilowatt hour,
    'total_co2_kg'        : 0.68,
    'total_cost_to_offset': 0.01,
}
"


# TEST dataset results
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
SRC_FPATH=$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_detectron_evals_test_imgs121_6cb3b6ff_v5
kwcoco info "$SRC_FPATH" -g1
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'
        matrix:
            detectron_pred.checkpoint_fpath:
                 - /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0119999.pth
                 - /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v5/v_9988a525/model_0079999.pth
            detectron_pred.src_fpath:
                - $SRC_FPATH
            detectron_pred.workers: 4
            detectron_pred.write_heatmap: true
            detectron_pred.nms_thresh: 0.5
            detection_eval.__enabled__: 1
            heatmap_eval.__enabled__: 1
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_detectron_evals_test_imgs121_6cb3b6ff_v5
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
