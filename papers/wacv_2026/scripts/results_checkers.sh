# These are the aggregators for main result locations.
__doc__="
TODO: link to scripts that actually performed these analysis.


~/code/shitspotter/experiments/yolo-experiments/run_yolo_experiments_v1.sh

~/code/shitspotter/experiments/grounding-dino-experiments/run_grounding_dino_experiments_v1.sh

~/code/shitspotter/experiments/grounding-dino-experiments/tune_grounding_dino.sh

~/code/shitspotter/experiments/detectron2-experiments/train_baseline_maskrcnn_v3.sh

~/code/shitspotter/experiments/detectron2-experiments/train_baseline_maskrcnn_scratch_v4.sh

~/code/shitspotter/experiments/geowatch-experiments/run_pixel_eval_on_vali_pipeline.sh


This script is just a quick reference to get the mlops aggregators for main experiments.
The script that should aggregate everything for the paper is:

~/code/shitspotter/papers/wacv_2026/scripts/build_v2_result_table.py


"



# Result aggregation and reporting (subset of models)
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_evals_2024_v2
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.polygon_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=0 \
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
        print_models: 0
        reference_region: null
        concise: 0
        show_csv: 0
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 2
        max_variations: 40
        min_support: 1
        params_of_interest:
            - resolved_params.heatmap_pred_fit.model.init_args.arch_name
            - resolved_params.heatmap_pred_fit.model.init_args.perterb_scale
            - resolved_params.heatmap_pred_fit.optimizer.init_args.lr
            - resolved_params.heatmap_pred_fit.optimizer.init_args.weight_decay
            - resolved_params.heatmap_pred_fit.trainer.default_root_dir
            - params.heatmap_pred.package_fpath
    " \
    --cache_resolved_results=False \
    --custom_query="
    import kwutil
    top_models = kwutil.Yaml.coerce(ub.Path('~/code/shitspotter/experiments/top_models2.yaml').expand())
    #top_models = ['/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/epoch=0089-step=122940-val_loss=0.019.ckpt.pt']
    new_eval_type_to_aggregator = {}
    for key, agg in eval_type_to_aggregator.items():
        flags = agg.table['resolved_params.heatmap_pred.package_fpath'].apply(lambda x: x in top_models)
        new_eval_type_to_aggregator[key] = agg.compress(flags)
    "




DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_detectron_evals
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.detectron_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=1 \
    --io_workers=0 \
    --eval_nodes="
        #- detection_evaluation
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



# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_evals_open_grounding_dino/vali_imgs691_99b22ad0
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.other.open_grounding_dino_pipeline.open_grounding_dino_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=0 \
    --rois=None \
    --io_workers=0 \
    --eval_nodes="
        - detection_evaluation
    " \
    --stdout_report="
        top_k: 100
        per_group: null
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 0
        show_csv: 0
    "

# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=0 \
    --rois=None \
    --io_workers=0 \
    --eval_nodes="
        - detection_evaluation
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


# Prompt Variation results

DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_evals/vali_imgs691_99b22ad0
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.other.grounding_dino_pipeline.grounding_dino_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=1 \
    --rois=None \
    --io_workers=0 \
    --eval_nodes="
        - detection_evaluation
    " \
    --cache_resolved_results=0 \
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

DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_evals/test_imgs121_6cb3b6ff
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.other.grounding_dino_pipeline.grounding_dino_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=0 \
    --rois=None \
    --io_workers=0 \
    --eval_nodes="
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
