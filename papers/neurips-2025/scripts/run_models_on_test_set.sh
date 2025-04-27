#!/bin/bash
__doc__="
This is a script to run the models on the updated test set. This is an attempt
to reconstruct experiments into a single location to make them easier to
reproduce.

This is based on original test / validation runs in:

    ~/code/shitspotter/experiments/geowatch-experiments/run_pixel_eval_on_test_pipeline.sh
    ~/code/shitspotter/experiments/detectron2-experiments/train_baseline_maskrcnn_scratch_v4.sh
    ~/code/shitspotter/experiments/detectron2-experiments/train_baseline_maskrcnn_v3.sh


The best models on the validation set were:

    (VIT Scratch)
    /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/epoch=0089-step=122940-val_loss=0.019.ckpt.pt

    (MaskRCNN From Pretrained)
    /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v3/v_966e49df/model_0014999.pth

    (MaskRCNN From Scratch)
    /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0099999.pth

"

### VIT MODELS
export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_test_evals
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'
        matrix:
            heatmap_pred.package_fpath:
                - '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/epoch=0089-step=122940-val_loss=0.019.ckpt.pt'
            heatmap_pred.test_dataset:
                - $TEST_FPATH
            heatmap_eval.workers: 1
            heatmap_eval.draw_heatmaps: 0
            heatmap_eval.draw_curves: True
            heatmap_pred.__enabled__: 1
            heatmap_eval.__enabled__: 1
            extract_polygons.__enabled__: 1
            extract_polygons.workers:
                - 4
            extract_polygons.thresh:
                - 0.5
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1," --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --run=1

# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_test_evals
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.polygon_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=0 \
    --rois=None \
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
        concise: 1
        show_csv: 0
    "




# DETECTRON MODELS
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_detectron_test_evals
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'
        matrix:
            detectron_pred.checkpoint_fpath:
                - /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v3/v_966e49df/model_0014999.pth
                - /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_scratch_v4/v_280638bd/model_0099999.pth
            detectron_pred.src_fpath:
                - $TEST_FPATH
            detectron_pred.workers: 4
            detectron_pred.write_heatmap: true
            detectron_pred.nms_thresh: 0.5
            detection_eval.__enabled__: 1
            heatmap_eval.__enabled__: 1
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --run=0



DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_detectron_test_evals
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.detectron_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=0 \
    --rois=None \
    --io_workers=0 \
    --eval_nodes="
        - heatmap_eval
        - detection_evaluation
    " \
    --stdout_report="
        top_k: 10
        per_group: null
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 1
        show_csv: 0
    "


__results__="

VIT box

  region_id  param_hashid        ap       auc  max_f1_f1  max_f1_tpr  max_f1_ppv
0   unknown  pbuznriqmgbh  0.421657  0.425515   0.556886     0.41704    0.837838

VIT pixel

               region_id  param_hashid  salient_AP  salient_AUC
0  test_imgs121_6cb3b6ff  nrxvegahrndd    0.473143     0.902277




DETECTRON box
Top 2 / 2 for detection_evaluation, unknown
region_id param_hashid       ap      auc  max_f1_f1  max_f1_tpr  max_f1_ppv
  unknown hsvbfwcwqurw 0.252947 0.464412   0.346253    0.300448    0.408537
  unknown lhrhjkrbsrte 0.612810 0.697425   0.650367    0.596413    0.715054


DETECTRON pixel

Top 2 / 2 for heatmap_eval, test_imgs121_6cb3b6ff
            region_id param_hashid  salient_AP  salient_AUC
test_imgs121_6cb3b6ff hsvbfwcwqurw    0.383513     0.797828
test_imgs121_6cb3b6ff lhrhjkrbsrte    0.810477     0.849399




"
