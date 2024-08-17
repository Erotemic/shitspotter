#!/bin/bash
__doc__="
# Evaluation on the training dataset
Setup:

Ensure you have an sdvc registery for the test dataset and packaged your checkpoints.
"

setup(){
    sdvc registery add --path "$HOME"/data/dvc-repos/shitspotter_dvc --name shitspotter_data --tags shitspotter_data
    sdvc registery add --path "$HOME"/data/dvc-repos/shitspotter_expt_dvc --name shitspotter_expt --tags shitspotter_expt
    sdvc registery list
    DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
    DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
    python -m geowatch.mlops.manager "status" --expt_dvc_dpath "$DVC_EXPT_DPATH"
    python -m geowatch.mlops.manager "list" --expt_dvc_dpath "$DVC_EXPT_DPATH"
}
export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")

test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"

#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs5747_1e73d54f.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_train_evals


# specified models
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs5747_1e73d54f.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_train_evals
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.heatmap_evaluation_pipeline()'
        #pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'
        matrix:
            heatmap_pred.package_fpath:
                - /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/epoch=0089-step=122940-val_loss=0.019.ckpt.pt
            heatmap_pred.test_dataset:
                - $TRAIN_FPATH
            heatmap_eval.workers: 1
            heatmap_eval.draw_heatmaps: 1
            heatmap_eval.draw_legend: 0
            heatmap_eval.draw_curves: True
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1," --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --run=1


# Simple no-dependency result readout
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_train_evals

python -c "if 1:
    import ubelt as ub
    eval_fpaths = list(ub.Path('$EVAL_PATH/eval/flat/heatmap_eval').glob('*/pxl_eval.json'))
    import json
    datas = [json.loads(p.read_text()) | {'path': p} for p in ub.ProgIter(eval_fpaths, desc='load')]

    datas = sorted(datas, key=lambda r: r['nocls_measures']['ap'])

    for data in datas:
        root_dir = data['meta']['info'][-2]['properties']['extra']['fit_config']['trainer']['default_root_dir']

        package_fpath = ub.Path(data['meta']['info'][-2]['properties']['config']['package_fpath'])

        #if 'noboxes_v' in ub.Path(root_dir).name:
        print(data['nocls_measures']['ap'], root_dir, data['path'])
"


# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_train_evals

python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.heatmap_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=1 \
    --io_workers=0 \
    --eval_nodes="
        - heatmap_eval
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
    "


# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_train_evals

python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.heatmap_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/aggregate" \
    --resource_report=1 \
    --io_workers=0 \
    --eval_nodes="
        - heatmap_eval
    " \
    --stdout_report="
        top_k: 10
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 0
        show_csv: 0
    " \
    --plot_params="
        enabled: 1
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
    --query="
    df['resolved_params.heatmap_pred_fit.trainer.default_root_dir'].apply(lambda p: str(p).split('/')[-1]).str.contains('noboxes')
    " \
    --custom_query="
        new_eval_type_to_aggregator = {}
        for key, agg in eval_type_to_aggregator.items():
            chosen_idxs = []
            for group_id, group in agg.table.groupby('resolved_params.heatmap_pred_fit.trainer.default_root_dir'):
                group['metrics.heatmap_eval.salient_AP'].argsort()
                keep_idxs = group['metrics.heatmap_eval.salient_AP'].sort_values()[-5:].index
                chosen_idxs.extend(keep_idxs)
            new_agg = agg.filterto(index=chosen_idxs)
            rich.print(f'Special filter {key} filtered to {len(new_agg)}/{len(agg)} rows')
            new_eval_type_to_aggregator[key] = new_agg

        if 1:
            new_agg.table
            new_agg.table.search_columns('lr')
            new_agg.table.search_columns('resolved_params.heatmap_pred_fit')
            new_agg.table.search_columns('resolved_params.heatmap_pred_fit.lr_scheduler.init_args.max_lr')
            new_agg.table['resolved_params.heatmap_pred_fit.lr_scheduler.init_args.max_lr']

            subcols = [
                'resolved_params.heatmap_pred_fit.trainer.default_root_dir',
                'resolved_params.heatmap_pred_fit.optimizer.init_args.lr',
                'resolved_params.heatmap_pred_fit.optimizer.init_args.weight_decay',
                'resolved_params.heatmap_pred_fit.model.init_args.perterb_scale',
                'metrics.heatmap_eval.salient_AP',
                'metrics.heatmap_eval.salient_AUC',
            ]
            new_agg.table[subcols]

            from geowatch.utils.util_pandas import pandas_shorten_columns, pandas_condense_paths

            chosen_idxs = []
            for group_id, group in agg.table.groupby('resolved_params.heatmap_pred_fit.trainer.default_root_dir'):
                group['metrics.heatmap_eval.salient_AP'].argsort()
                keep_idxs = group['metrics.heatmap_eval.salient_AP'].sort_values()[-1:].index
                chosen_idxs.extend(keep_idxs)

            table = new_agg.table.safe_drop(['resolved_params.heatmap_pred_fit.trainer.callbacks'], axis=1)
            varied = table.varied_value_counts(min_variations=2)
            print(list(varied.keys()))

            subtable = new_agg.table.loc[chosen_idxs, subcols]
            subtable = pandas_shorten_columns(subtable)
            subtable['default_root_dir'] = pandas_condense_paths(subtable['default_root_dir'])[0]
            subtable = subtable.sort_values(['salient_AP'], ascending=False)
            print(subtable.to_latex(index=False))
    "

#
