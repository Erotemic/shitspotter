#!/bin/bash
# Localize figures

FIGURE_DPATH=$HOME/code/shitspotter/papers/wacv_2026/figures
cp $HOME/data/dvc-repos/shitspotter_dvc/analysis/viz_three_images.jpg "$FIGURE_DPATH"
cp $HOME/code/shitspotter/coco_annot_stats/annot_stat_plots/all_polygons.png "$FIGURE_DPATH"
cp $HOME/code/shitspotter/coco_annot_stats/annot_stat_plots/images_timeofday_distribution.png "$FIGURE_DPATH"
cp $HOME/code/shitspotter/coco_annot_stats/annot_stat_plots/anns_per_image_histogram_splity.png "$FIGURE_DPATH"
cp $HOME/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/vali_result_heatmaps.jpg "$FIGURE_DPATH"
cp $HOME/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/aggregate/plots/macro-plots-macro_01_c1edce/params/resolved_params.heatmap_pred_fit.trainer.default_root_dir/scatter_nolegend/macro_results_resolved_params.heatmap_pred_fit.trainer.default_root_dir_metrics.heatmap_eval.salient_AP_vs_metrics.heatmap_eval.salient_AUC_PLT02_scatter_nolegend.png "$FIGURE_DPATH"
cp $HOME/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/aggregate/plots/macro-plots-macro_01_c1edce/params/resolved_params.heatmap_pred_fit.trainer.default_root_dir/box/macro_results_resolved_params.heatmap_pred_fit.trainer.default_root_dir_metrics.heatmap_eval.salient_AP_PLT04_box.png "$FIGURE_DPATH"
cp $HOME/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/aggregate/plots/macro-plots-macro_01_c1edce/params/resolved_params.heatmap_pred_fit.trainer.default_root_dir/macro_results_resolved_params.heatmap_pred_fit.trainer.default_root_dir_PLT05_table.png "$FIGURE_DPATH"
cp $HOME/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/test_result_heatmaps.jpg "$FIGURE_DPATH"
cp $HOME/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/aggregate/plots/resources.png "$FIGURE_DPATH"


rsync -avprPR toothbrush:code/shitspotter/papers/wacv_2026/./figures "$HOME"/code/shitspotter/papers/wacv_2026/

# See new IPFS stuff in manage_data_resources
