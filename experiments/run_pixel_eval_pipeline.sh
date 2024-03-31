#!/bin/bash
__doc__="
Setup:

Ensure you have an sdvc registery for the test dataset and packaged your checkpoints.
"

setup(){
    sdvc registery add --path "$HOME"/data/dvc-repos/shitspotter_dvc --name shitspotter_data --tags shitspotter_data
    sdvc registery add --path "$HOME"/data/dvc-repos/shitspotter_expt_dvc --name shitspotter_expt --tags shitspotter_expt
    sdvc registery list

    python -m geowatch.mlops.manager "status" --expt_dvc_dpath "$DVC_EXPT_DPATH"
    python -m geowatch.mlops.manager "list" --expt_dvc_dpath "$DVC_EXPT_DPATH"
    python -m geowatch.mlops.manager "repackage checkpoints" --expt_dvc_dpath "$DVC_EXPT_DPATH"
}
export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")

test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"

#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs228_20928c8c.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_evals



# TODO: script to subselect models based on train-time validation metrics.


## Use geowatch mlops to define a grid of bash jobs to run evaluation over
## specified models
#python -m geowatch.mlops.schedule_evaluation \
#    --params="
#        pipeline: 'shitspotter.pipelines.heatmap_evaluation_pipeline()'
#        matrix:
#            heatmap_pred.package_fpath:
#                - $DVC_DATA_DPATH/models/shitspotter_scratch_v025-version_2-epoch=1277-step=005112-val_loss=0.600.ckpt.pt
#            heatmap_pred.test_dataset:
#                - $VALI_FPATH
#            heatmap_eval.workers: 4
#            heatmap_eval.draw_heatmaps: True
#            heatmap_eval.draw_curves: True
#    " \
#    --root_dpath="$EVAL_PATH" \
#    --devices="0,1" --tmux_workers=2 \
#    --backend=serial --skip_existing=0 \
#    --run=0



# specified models
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.heatmap_evaluation_pipeline()'
        matrix:
            heatmap_pred.package_fpath:
                #- $HOME/code/shitspotter/experiments/first_chosen_eval_batch1.yaml
                - $HOME/code/shitspotter/experiments/first_eval_batch.yaml
                #- $DVC_DATA_DPATH/models/shitspotter_scratch_v025-version_2-epoch=1277-step=005112-val_loss=0.600.ckpt.pt
            heatmap_pred.test_dataset:
                - $VALI_FPATH
            heatmap_eval.workers: 1
            heatmap_eval.draw_heatmaps: 0
            heatmap_eval.draw_curves: True
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="1," --tmux_workers=1 \
    --backend=tmux --skip_existing=1 \
    --run=1


#python -m geowatch.mlops.aggregate \
#    --pipeline='shitspotter.pipelines.heatmap_evaluation_pipeline()' \
#    --target "
#        - $EVAL_PATH
#    " \
#    --output_dpath="$EVAL_PATH/aggregate" \
#    --resource_report=0 \
#    --eval_nodes="
#        - heatmap_eval
#    " \
#    --stdout_report="
#        top_k: 10
#        per_group: 1
#        macro_analysis: 0
#        analyze: 0
#        print_models: True
#        reference_region: final
#        concise: 1
#        show_csv: 0
#    " \
#    --plot_params="
#        enabled: 0
#        stats_ranking: 0
#        min_variations: 1
#        params_of_interest:
#            - params.bas_poly.thresh
#            - resolved_params.bas_pxl.channels
#    "
#    #\
#    #--rois="KR_R002,CN_C000,KW_C001,CO_C001"
#    #--rois="KR_R002"
#    #--rois="KR_R002,CN_C000"
#    #--rois="CN_C000"
#
#

python -c "if 1:
    import ubelt as ub
    eval_fpaths = list(ub.Path('$EVAL_PATH/eval/flat/heatmap_eval').glob('*/pxl_eval.json'))
    import json
    datas = [json.loads(p.read_text()) | {'path': p} for p in ub.ProgIter(eval_fpaths, desc='load')]

    datas = sorted(datas, key=lambda r: r['nocls_measures']['ap'])

    for data in datas:
        root_dir = data['meta']['info'][-2]['properties']['extra']['fit_config']['trainer']['default_root_dir']
        print(data['nocls_measures']['ap'], root_dir, data['path'])
"
