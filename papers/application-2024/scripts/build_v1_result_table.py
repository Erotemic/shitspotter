#!/usr/bin/env python3
"""
This replicates the custom query to build the result table used in the
corrected version of the paper submitted to WACV. The original submission only
contained results for a subset of the validation set. This is fixed to agree
with the state of the paper as of 2024-10-06 on git hash ...
"""
from geowatch.mlops.aggregate import AggregateEvluationConfig
import pandas as pd
from geowatch.utils.util_pandas import pandas_shorten_columns, pandas_condense_paths
from geowatch.utils.util_pandas import DataFrame
import rich
import numpy as np
# from kwcoco.metrics.drawing import concice_si_display
import ubelt as ub

config = AggregateEvluationConfig(**{
    'target': ['/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals_2024_v2'],
    'pipeline': 'shitspotter.pipelines.heatmap_evaluation_pipeline()',
    'io_workers': 0,
    'eval_nodes': ['heatmap_eval'],
    'primary_metric_cols': 'auto',
    'display_metric_cols': 'auto',
    'cache_resolved_results': True,
    'output_dpath': '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals_2024_v2/aggregate',
    'export_tables': False,
    'plot_params': {'enabled': 0, 'stats_ranking': 0, 'min_variations': 2, 'max_variations': 40, 'min_support': 1, 'params_of_interest':
                    ['resolved_params.heatmap_pred_fit.model.init_args.arch_name', 'resolved_params.heatmap_pred_fit.model.init_args.perterb_scale',
                     'resolved_params.heatmap_pred_fit.optimizer.init_args.lr', 'resolved_params.heatmap_pred_fit.optimizer.init_args.weight_decay',
                     'resolved_params.heatmap_pred_fit.trainer.default_root_dir', 'params.heatmap_pred.package_fpath']},
    'stdout_report': '\n        top_k: 10\n        per_group: 1\n        macro_analysis: 0\n        analyze: 0\n        print_models: True\n        reference_region: final\nconcise: 0\n        show_csv: 0\n    ',
    'resource_report': 1,
    'symlink_results': False,
    'rois': 'auto',
    'inspect': None,
    'query': "df['resolved_params.heatmap_pred_fit.trainer.default_root_dir'].apply(lambda p: str(p).split('/')[-1]).str.contains('noboxes')",
    'custom_query': None,
    'embed': True,
    'snapshot': False,
})
eval_type_to_aggregator = config.coerce_aggregators()
orig_eval_type_to_aggregator = eval_type_to_aggregator  # NOQA


new_eval_type_to_aggregator = {}
for key, agg in eval_type_to_aggregator.items():
    chosen_idxs = []
    for group_id, group in agg.table.groupby('resolved_params.heatmap_pred_fit.trainer.default_root_dir'):
        group['metrics.heatmap_eval.salient_AP'].argsort()
        if 0:
            # Hack to remove baddies?
            flags = group['params.heatmap_pred.package_fpath'].apply(lambda x: 'last.' not in x.split('/')[-1])
            group = group[flags]
        keep_idxs = group['metrics.heatmap_eval.salient_AP'].sort_values()[-5:].index
        chosen_idxs.extend(keep_idxs)
    new_agg = agg.filterto(index=chosen_idxs)
    rich.print(f'Special filter {key} filtered to {len(new_agg)}/{len(agg)} rows')
    new_eval_type_to_aggregator[key] = new_agg

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
    'params.heatmap_pred.package_fpath',
]
new_agg.table[subcols]

chosen_idxs = []
for group_id, group in new_agg.table.groupby('resolved_params.heatmap_pred_fit.trainer.default_root_dir'):
    group['metrics.heatmap_eval.salient_AP'].argsort()
    keep_idxs = group['metrics.heatmap_eval.salient_AP'].sort_values()[-1:].index
    chosen_idxs.extend(keep_idxs)

table = new_agg.table.safe_drop(['resolved_params.heatmap_pred_fit.trainer.callbacks'], axis=1)
varied = table.varied_value_counts(min_variations=2)
print(list(varied.keys()))

if 0:
    # Hack to extract epoch numbers, doesnt work because of last.pt
    epoch_nums = []
    for path in table['params.heatmap_pred.package_fpath']:
        name = ub.Path(path).name
        if name == 'last.pt':
            #from geowatch.tasks.fusion.utils import load_model_header
            #header = load_model_header(path)
            epoch_nums.append(np.nan)
        else:
            n = name.split('-')[0].split('=')[1]
            epoch_nums.append(int(n))
    table['epoch_num'] = epoch_nums
    subtable = table.loc[chosen_idxs, subcols + ['epoch_num']]

subtable = table.loc[chosen_idxs, subcols]
subtable = DataFrame(subtable)
subtable = subtable.shorten_columns()
subtable = pandas_shorten_columns(subtable)
subtable['default_root_dir'] = pandas_condense_paths(subtable['default_root_dir'])[0]
subtable = subtable.sort_values(['salient_AP'], ascending=False)

# hack to get the right models for computing test numbers
target_order_for_test_set = subtable['package_fpath'].to_list()
print(ub.urepr(target_order_for_test_set))


def format_scientific_notation(val, precision=2):
    val_str = ('{:.' + str(precision) + 'e}').format(val)
    lhs, rhs = val_str.split('e')
    import re
    trailing_zeros = re.compile(r'\.0*$')
    rhs = rhs.replace('+', '')
    rhs = rhs.lstrip('0')
    rhs = rhs.replace('-0', '-')
    lhs = trailing_zeros.sub('', lhs)
    rhs = trailing_zeros.sub('', rhs)
    val_str = lhs + 'e' + rhs
    return val_str
subtable_display = subtable.copy()
si_params = ['lr', 'weight_decay', 'perterb_scale']
for p in si_params:
    subtable_display[p] = subtable[p].apply(format_scientific_notation)
metric_params = ['salient_AP', 'salient_AUC']
for p in metric_params:
    subtable_display[p] = subtable[p].apply(lambda x: '{:0.4f}'.format(x))

subtable_display = subtable_display.drop(['package_fpath'], axis=1)
print(subtable_display.to_latex(index=False))

# Add test results:
# TODO: we should be reading this from the test mlops directory
# and joining the information.
test_results_to_integrate = [
    {'default_root_dir': 'shitspotter_scratch_20240618_noboxes_v7',
     'salient_AP': 0.5051101001895235,
     'salient_AUC': 0.91250945170183},
    {'default_root_dir': 'shitspotter_scratch_20240618_noboxes_v6',
     'salient_AP': 0.4345697006282774,
     'salient_AUC': 0.8575508338320234},
    {'default_root_dir': 'shitspotter_scratch_20240618_noboxes_v5',
     'salient_AP': 0.4652248750059659,
     'salient_AUC': 0.7965005428232322},
    {'default_root_dir': 'shitspotter_scratch_20240618_noboxes_v4',
     'salient_AP': 0.5166517253996291,
     'salient_AUC': 0.9252187841782987},
    {'default_root_dir': 'shitspotter_scratch_20240618_noboxes_v2',
     'salient_AP': 0.42097989185404483,
     'salient_AUC': 0.7766404213212321},
    {'default_root_dir': 'shitspotter_scratch_20240618_noboxes_v3',
     'salient_AP': 0.4606774223010137,
     'salient_AUC': 0.9062428313078754},
    {'default_root_dir': 'shitspotter_scratch_20240618_noboxes_v8',
     'salient_AP': 0.41374633968103414,
     'salient_AUC': 0.8156542044578126}]
test_results = pd.DataFrame(test_results_to_integrate)
metric_params = ['salient_AP', 'salient_AUC']
for p in metric_params:
    test_results[p] = test_results[p].apply(lambda x: '{:0.4f}'.format(x))

columns = [('0', 'n'), ('0', 'p'), ('0', 'e'), ('1', 'n'), ('1', 'p'), ('1', 'e')]

tuples = [('', c) for c in subtable_display.columns]
mcols = pd.MultiIndex.from_tuples([
    ('', 'default_root_dir'),
    ('', 'lr'),
    ('', 'weight_decay'),
    ('', 'perterb_scale'),
    ('val', 'salient_AP'),
    ('val', 'salient_AUC'),
    ('test', 'salient_AP'),
    ('test', 'salient_AUC'),
])
subtable_display.columns = mcols[:-2]
print(subtable_display.to_latex(index=False))
toconcat = test_results[['salient_AP', 'salient_AUC']]
toconcat.columns = mcols[-2:]
new_table = pd.concat([subtable_display.reset_index(drop=1), toconcat.reset_index(drop=1)], axis=1)
# print(new_table.style.to_latex())
#
root_lut = {
    'shitspotter_scratch_20240618_noboxes_v7': 'D05',
    'shitspotter_scratch_20240618_noboxes_v6': 'D04',
    'shitspotter_scratch_20240618_noboxes_v5': 'D03',
    'shitspotter_scratch_20240618_noboxes_v4': 'D02',
    'shitspotter_scratch_20240618_noboxes_v2': 'D00',
    'shitspotter_scratch_20240618_noboxes_v3': 'D01',
    'shitspotter_scratch_20240618_noboxes_v8': 'D06',
}
new_table[('', 'default_root_dir')] = new_table[('', 'default_root_dir')].apply(root_lut.__getitem__)
new_table = new_table.rename({'default_root_dir': 'config name'}, axis=1)

idx_order = new_table.reset_index().set_index(('', 'config name'), drop=False).loc[list(root_lut.values())]['index'].tolist()
new_table = new_table.loc[idx_order]
rich.print(new_table)

new_table = new_table.sort_values(('val', 'salient_AP'), ascending=False)
print(new_table.style.format_index().hide().to_latex())
print(new_table.to_latex(index=False))


colormap = {
    'D05': '623682',
    'D04': '87b787',
    'D03': 'df8020',
    'D02': '207fdf',
    'D00': '20df20',
    'D01': 'df20df',
    'D06': 'b00403',
}


test = new_table.copy()
colnames = [
    ('val', 'salient_AP'),
    ('val', 'salient_AUC'),
    ('test', 'salient_AP'),
    ('test', 'salient_AUC'),
]
for colname in colnames:
    col = test[colname]
    max_idx = col.argsort().iloc[-1]
    val = test.iloc[max_idx][colname]
    test.iloc[max_idx][colname] = fr'\textbf{{{val}}}'

colname  = ('', 'config name')
col = test[colname]

for idx in range(len(col)):
    name = col.iloc[idx]
    hexval = colormap[name]
    newval = rf'\textcolor[HTML]{{{hexval}}}{{{name}}}'
    test.iloc[idx][colname] = newval


test.rename({
    'salient_AP': 'AP',
    'salient_AUC': 'AUC',
}, axis=1)
print(test.style.format_index().hide().to_latex())
print(test.to_latex(index=False, escape=False))
