import pandas as pd
import rich
import kwimage
import ubelt as ub
from geowatch.mlops.aggregate import AggregateEvluationConfig
import kwutil


def load_aggregators():
    """
    Load the aggregators for all experiments. We will use these to find
    comparable results, build a table, and draw example results.

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/shitspotter/papers/neurips-2025/scripts'))
        from build_v2_result_table import *  # NOQA
    """
    dvc_expt_dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/')

    inputs = kwutil.Yaml.coerce(
        '''

        - dname: _shitspotter_test_imgs121_6cb3b6ff_evals
          pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'

        - dname: _shitspotter_evals
          pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'

        - dname: _shitspotter_evals_2024_v2
          pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'

        - dname: _shitspotter_test_evals
          pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'

        - dname: _shitspotter_train_evals
          pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'
        - dname: _shitspotter_train_evals2
          pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'

        - dname: shitspotter-test-v2
          pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'

        - dname: _shitspotter_detectron_evals
          pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'

        - dname: _shitspotter_detectron_evals_old
          pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'
        - dname: _shitspotter_detectron_evals_test
          pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'

        - dname: _shitspotter_detectron_evals_v4
          pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'

        - dname: _shitspotter_detectron_evals_v4_test
          pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'

        - dname: _shitspotter_detectron_evals_test_imgs121_6cb3b6ff
          pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'

        - dname: _shitspotter_detectron_evals_v4_test_imgs121_6cb3b6ff
          pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'

        # NEW

        # Tuned Grounding Dino Validation691 Results
        - dname: _shitspotter_2025_rebutal_evals_open_grounding_dino/vali_imgs691_99b22ad0
          pipeline: 'shitspotter.other.open_grounding_dino_pipeline.open_grounding_dino_evaluation_pipeline()'
          eval_nodes: [detection_evaluation]

        # Tuned Grounding Dino Test121 Results
        - dname: _shitspotter_2025_rebutal_evals_open_grounding_dino/test_imgs121_6cb3b6ff
          pipeline: 'shitspotter.other.open_grounding_dino_pipeline.open_grounding_dino_evaluation_pipeline()'
          eval_nodes: [detection_evaluation]

        # ZeroShot Grounding Dino Validation691 Results
        - dname: _shitspotter_2025_rebutal_evals/vali_imgs691_99b22ad0
          pipeline: 'shitspotter.other.grounding_dino_pipeline.grounding_dino_evaluation_pipeline()'
          eval_nodes: [detection_evaluation]

        # ZeroShot Grounding Dino Test121 Results
        - dname: _shitspotter_2025_rebutal_evals/test_imgs121_6cb3b6ff
          pipeline: 'shitspotter.other.grounding_dino_pipeline.grounding_dino_evaluation_pipeline()'
          eval_nodes: [detection_evaluation]

        # From Pretrained YOLO-v9 Validation691 Results
        - dname: _shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0
          pipeline: 'shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()'
          eval_nodes: [detection_evaluation]

        - dname: _shitspotter_2025_rebutal_yolo_evals/test_imgs121_6cb3b6ff
          pipeline: 'shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()'
          eval_nodes: [detection_evaluation]

        # From Scratch YOLO-v9 Validation691 Results
        - dname: _shitspotter_2025_rebutal_yolo_scratch_evals/vali_imgs691_99b22ad0
          pipeline: 'shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()'
          eval_nodes: [detection_evaluation]

        # From Scratch YOLO-v9 Test121 Results
        - dname: _shitspotter_2025_rebutal_yolo_scratch_evals/test_imgs121_6cb3b6ff
          pipeline: 'shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()'
          eval_nodes: [detection_evaluation]
        ''')

    aggregators_rows = []
    for row in inputs:
        row['target'] = dvc_expt_dpath / row['dname']
        eval_nodes = row.get('eval_nodes', ['heatmap_eval', 'detection_evaluation'])
        config = AggregateEvluationConfig(**{
            'target': [row['target']],
            'pipeline': row['pipeline'],
            'eval_nodes': eval_nodes,
            'primary_metric_cols': 'auto',
            'display_metric_cols': 'auto',
            'io_workers': 16,
            # 'cache_resolved_results': False,
            'cache_resolved_results': True,
            'output_dpath': row['target'] / 'aggregate',
            # '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals_2024_v2/aggregate',
            # 'query': "df['resolved_params.heatmap_pred_fit.trainer.default_root_dir'].apply(lambda p: str(p).split('/')[-1]).str.contains('noboxes')",
        })
        print(f'Loading for: row={row}')
        eval_type_to_aggregator = config.coerce_aggregators()
        print(f'Loaded for: row={row}')
        for agg in eval_type_to_aggregator.values():
            agg_row = {
                'agg': agg,
                'node_type': agg.node_type,
                **row
            }
            agg_row = ub.udict(agg_row)
            if len(agg) > 0:
                aggregators_rows.append(agg_row)

        agg_row['model_type'] = get_heuristic_model_type(agg_row)
        REMOVE_NON_COMPARABLE_YOLO_MODEL = True
        if REMOVE_NON_COMPARABLE_YOLO_MODEL:
            if agg_row['model_type'] == 'yolo_v9':
                # Hack to remove the simplified model trained on the larger subset
                keep_flags = []
                for config_text in agg_row['agg'].table['resolved_params.yolo_pred.model_config']:
                    config_fpath = ub.Path(kwutil.Yaml.coerce(config_text)['config'])
                    train_config = kwutil.Yaml.coerce(config_fpath.read_text())
                    flag = 'train_imgs5747_1e73d54f' in train_config['dataset']['train']
                    keep_flags.append(flag)
                agg_row['agg'] = agg_row['agg'].compress(keep_flags)

    for agg_row in aggregators_rows:
        agg_row['model_type'] = get_heuristic_model_type(agg_row)
        # if 'detectron' in agg_row['pipeline'] else 'geowatch'

        agg = agg_row['agg']
        row_type = agg_row['model_type']
        if agg.node_type == 'detection_evaluation':
            if row_type == 'geowatch':
                try:
                    test_dsets = agg.table['resolved_params.heatmap_pred.test_dataset'].unique()
                except KeyError:
                    # not sure why resolved doesn't esxist
                    test_dsets = agg.table['params.heatmap_pred.test_dataset'].unique()
            elif row_type == 'grounding_dino-tuned':
                test_dsets = agg.table['resolved_params.open_grounding_dino_pred.src'].unique()
            elif row_type == 'grounding_dino-zero':
                test_dsets = agg.table['resolved_params.grounding_dino_pred.src'].unique()
            elif row_type == 'yolo_v9':
                test_dsets = agg.table['resolved_params.yolo_pred.src'].unique()
            elif row_type == 'detectron':
                test_dsets = agg.table['params.detectron_pred.src_fpath'].unique()
            else:
                raise Exception(f'no {row_type=}')

        elif agg.node_type == 'heatmap_eval':
            if row_type == 'geowatch':
                test_dsets = agg.table['params.heatmap_pred.test_dataset'].unique()
            elif row_type == 'detectron':
                test_dsets = agg.table['resolved_params.heatmap_eval.true_dataset'].unique()
                # agg.table['params.detectron_pred.src_fpath'].unique()
            else:
                raise Exception(f'no {row_type=}')
        else:
            raise NotImplementedError(agg.node_type)
        test_dsets = [p for p in test_dsets if not pd.isnull(p)]
        assert len(test_dsets) == 1
        agg_row['dataset_name'] = ub.Path(test_dsets[0]).stem

    dataset_name_blocklist = {
        'vali_imgs228_20928c8c.kwcoco',
        'test_imgs30_d8988f8c.kwcoco',
    }
    tmp = []
    for r in aggregators_rows:
        if r['dataset_name'] not in dataset_name_blocklist:
            tmp.append(r)
    aggregators_rows = tmp
    return aggregators_rows


def get_heuristic_model_type(agg_row):
    # Hacky method to get what type of model it is
    if 'detectron' in agg_row['pipeline']:
        return 'detectron'
    elif '.open_grounding_dino_pipeline.' in agg_row['pipeline']:
        return 'grounding_dino-tuned'
    elif '.grounding_dino_pipeline.' in agg_row['pipeline']:
        return 'grounding_dino-zero'
    elif 'yolo' in agg_row['pipeline']:
        return 'yolo_v9'
    else:
        return 'geowatch'


def process_aggregators(aggregators_rows):
    agg_summaries = []
    for agg_row in aggregators_rows:

        if 'detectron' in agg_row['pipeline']:
            # hack
            if '_v4' in str(agg_row['target']):
                model_type = 'detectron-scratch'
            else:
                model_type = 'detectron-pretrained'
        else:
            htype = get_heuristic_model_type(agg_row)
            if htype == 'geowatch':
                model_type = 'geowatch-scratch'
            else:
                if htype == 'yolo_v9':
                    if 'scratch' in str(agg_row['target']):
                        model_type = 'yolo_v9-scratch'
                    else:
                        model_type = 'yolo_v9-pretrained'
                else:
                    model_type = htype  # fixme
        agg_row['model_type'] = model_type
        agg = agg_row['agg']
        # row_type = agg_row['model_type']

        summary = agg_row & ['dataset_name', 'model_type', 'node_type', 'dname']
        summary['len'] = len(agg)
        metric = agg.primary_metric_cols[0]
        summary[metric] = agg.table[metric].max()
        agg_summaries.append(summary)
    summary_df = pd.DataFrame(agg_summaries)
    rich.print(summary_df.to_string())

    possible_model_columns = [
        'checkpoint_fpath',
        'package_fpath',
        'model_config',
        'checkpoint_path',
        'checkpoint',
        'model_id',
    ]

    # ENSURE APPLES-TO-APPLES COMPARISONS
    # Find the results run on the validation dataset, and find the parameters
    # that maximized results on that dataset. We will then use these to select
    # the test results to report, so we have consistency in our result
    # reporting.
    chosen_validation_dataset_code = 'vali_imgs691_99b22ad0.kwcoco'
    dataset_groups = ub.group_items(aggregators_rows, lambda agg: agg['dataset_name'])
    vali_aggs_rows = dataset_groups[chosen_validation_dataset_code]
    high_level_chosen_params = {}
    high_level_info = {}
    for agg_row in vali_aggs_rows:
        agg = agg_row['agg']

        # This is the high level experiment type which will correspond to a row
        # in our final report table.
        high_level_experiment_key = (agg_row['model_type'], agg_row['node_type'])

        cands = sum([agg.table.search_columns(c) for c in possible_model_columns], start=[])
        if not cands:
            DO_HACK = agg_row['dname'] == '_shitspotter_detectron_evals_v4'
            if DO_HACK:
                # HACK
                agg.output_dpath.parent.ls()
                (agg.output_dpath.parent / 'pred/flat/detectron_pred').ls()
                new_rows = []
                for fpath in agg.table['fpath']:
                    data = kwutil.Json.load(ub.Path(fpath).parent / 'pxl_eval.json')
                    # wow: havent had a real use-case for this in awhile.
                    pred_fpath = ub.argval('--pred_dataset', argv=data['meta']['info'][-1]['properties']['args'])
                    pred_fpath = ub.Path(pred_fpath)
                    from geowatch.mlops.smart_result_parser import parse_json_header
                    from geowatch.utils import util_dotdict
                    assert pred_fpath.exists()
                    pred_config = parse_json_header(pred_fpath)[-1]['properties']['config']
                    new_params = util_dotdict.DotDict(pred_config).add_prefix('params.detectron_pred')
                    new_rows.append(new_params)
                new_cols = pd.DataFrame(new_rows)
                agg.table = pd.concat([agg.table, new_cols], axis=1)
            else:
                raise AssertionError

        # Grabbing the best model + config item for each dataset
        cands = sum([agg.table.search_columns(c) for c in possible_model_columns], start=[])
        if cands:
            model_column = cands[0]
            agg_row['model_column'] = model_column
            primary_metric = agg.primary_metric_cols[0]
            models = agg.table[model_column]

            if 'params.yolo_pred.model_config' in cands:
                # Hack because the model checkpoint is special with yolo
                new_models = []
                for model in models:
                    if isinstance(model, dict):
                        fixed = model['checkpoint']
                    else:
                        fixed = model
                    new_models.append(fixed)
                models = new_models

            # models = agg.table[primary_metric]
            ranked = agg.table[primary_metric].sort_values(ascending=False)
            top_loc = ranked.index[0]
            best_row = agg.table.loc[top_loc]
            agg_row['model_column'] = model_column
            agg_row['best_model_fpath'] = best_row[model_column]
            agg_row['best_metrics'] = best_row[agg.primary_metric_cols].to_dict()
            # need to know about other params that could impact the result
            # besides the checkpoint.
            other_important_params = {
                'resolved_params.extract_polygons.thresh',
                'resolved_params.grounding_dino_pred.classes',
                'resolved_params.open_grounding_dino_pred.classes',
            }
            chosen_params = ub.udict(best_row.to_dict()) & {
                model_column, *other_important_params
            }
            assert high_level_experiment_key not in high_level_chosen_params
            high_level_chosen_params[high_level_experiment_key] = chosen_params
            high_level_info[high_level_experiment_key] = {
                'chosen_params': chosen_params,
                'available_datasets': [],
            }
        else:
            print(f'Miss: high_level_experiment_key={high_level_experiment_key}')

    print('Based on the validation set, we want to evaluate these parameters:')
    print(f'high_level_chosen_params = {ub.urepr(high_level_chosen_params, nl=2)}')

    PREFER_CHOICE_BY_DET = 1
    if PREFER_CHOICE_BY_DET:
        # Take the same model for heatmap/detection models
        model_to_typekeys = ub.group_items(high_level_chosen_params, key=lambda k: k[0])
        model_to_hackchoice = {}
        for model, typekeys in model_to_typekeys.items():
            if len(typekeys) > 1:
                hack_choicekey = (model, 'detection_evaluation')
                model_to_hackchoice[model] = hack_choicekey

    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=1, name='redo-preds')

    # Loop over all aggregators and find the results that correspond to the
    # chosen validation parameters.
    for agg_row in aggregators_rows:
        agg_row.pop('error', None)
        agg = agg_row['agg']
        cands = sum([agg.table.search_columns(c) for c in possible_model_columns], start=[])
        if not cands:
            print(f'row with {agg_row["model_type"]=} does have registered models')
        else:
            model_column = cands[0]
            #model_column = agg_row['model_column']
            # print(f'model_column={model_column}')
            primary_metric = agg.primary_metric_cols[0]
            models = agg.table[model_column]
            ranked = agg.table[primary_metric].sort_values(ascending=False)
            # model_ranks = len(agg.table) - agg.table[primary_metric].argsort()
            model_ranks = ub.dzip(agg.table[primary_metric].sort_values(ascending=False).index, range(len(agg.table)))
            # top_loc = ranked.index[0]
            # best_row = agg.table.loc[top_loc]
            high_level_experiment_key = (agg_row['model_type'], agg_row['node_type'])
            if high_level_experiment_key not in high_level_chosen_params:
                print('--')
                print('Unable to find model relevant to this node')
                print(f'high_level_experiment_key={high_level_experiment_key}')
                print(f'agg_row={agg_row}')
                print('--')
                agg_row['error'] = f'no model col {high_level_experiment_key}'
            else:
                model_type = agg_row['model_type']
                chosen_params = high_level_chosen_params[high_level_experiment_key]
                if PREFER_CHOICE_BY_DET and model_type in model_to_hackchoice:
                    hackchoice_key = model_to_hackchoice[model_type]
                    hack_chosen_params = high_level_chosen_params[hackchoice_key]
                    # Choose this model based on a different evaluation type
                    chosen_params = ub.udict(hack_chosen_params) & set(chosen_params)

                try:
                    col_flags = {
                        k: agg.table[k] == v for k, v in chosen_params.items()
                    }
                except KeyError:
                    print('Group didnt have all cols')
                    agg_row['error'] = 'missing cols'
                else:
                    import numpy as np
                    flags = np.logical_and.reduce(list(col_flags.values()))
                    nfound = flags.sum()
                    if nfound == 0:
                        print('found none')
                        agg_row['error'] = 'no matching params'
                    else:
                        if nfound > 1:
                            print(f'Warning: should not happen: {nfound=}, but might if something was rerun with a new param, this is ok, as long as all metrics are the same')

                        locs = models[flags].index
                        ranks = [model_ranks[loc] for loc in locs]
                        chosen_rows = agg.table.loc[locs]
                        if nfound > 1:
                            # should at least have the same metrics
                            assert chosen_rows[agg.primary_metric_cols].apply(ub.allsame).all()

                        # chosen_metric_cols = agg.primary_metric_cols
                        chosen_metric_cols = agg.primary_metric_cols + list(ub.oset([
                            # Hack because reviewers want F1
                            'metrics.detection_evaluation.max_f1_f1',
                            'metrics.detection_evaluation.max_f1_tpr',
                            'metrics.heatmap_eval.salient_maxF1_F1',
                            'metrics.heatmap_eval.salient_maxF1_tpr',
                        ]) & set(agg.table.columns))
                        # print(f'chosen_metric_cols={chosen_metric_cols}')

                        chosen_metric_cols = chosen_metric_cols

                        rank = ranks[0]
                        chosen_row = chosen_rows.iloc[0]
                        chosen_metrics = chosen_row[chosen_metric_cols].to_dict()

                        resource_cols = chosen_rows.search_columns('kwh') + chosen_rows.search_columns('co2_kg') + chosen_rows.search_columns('duration')
                        # Remove non important rows
                        resource_cols = [c for c in resource_cols if 'eval' not in c and 'extract_polygons' not in c]
                        resource_data = chosen_rows[resource_cols]
                        from geowatch.utils import util_pandas
                        resource_data = util_pandas.DataFrame(resource_data)

                        resource_data = resource_data.shorten_columns().iloc[0].to_dict()

                        # print(f'resource_data={resource_data}')
                        # print(f'fpath={fpath}')
                        fpath = ub.Path(chosen_row['fpath'])

                        invoke_fpaths = list(fpath.parent.ls('.pred/*/*/invoke.sh'))
                        target_time = kwutil.datetime.coerce('2025-07-30T12:48:55')
                        for invoke_fpath in invoke_fpaths:
                            pred_fpath = invoke_fpath.parent / 'pred.kwcoco.zip'
                            if pred_fpath.exists():
                                create_time = kwutil.datetime.coerce(pred_fpath.stat().st_mtime)
                                if create_time < target_time:
                                    print(f'Found older result: create_time={create_time}')
                                    # print(invoke_fpath)
                                    job = queue.submit(f'bash "{invoke_fpath}"')
                                    for cache_path in fpath.parent.glob('resolved_result_row*'):
                                        queue.submit(f'rm -f "{cache_path}"')
                            # else:
                            #     print(invoke_fpath)
                        # if len(invoke_fpaths) > 1:
                        #     raise Exception

                        needs_compute = pd.isna(resource_data.get('kwh', None))
                        if needs_compute:
                            print(agg_row['dataset_name'], model_type)
                            print(f'fpath={fpath}')
                            jobs = []
                            for invoke_fpath in fpath.parent.ls('.pred/*/*/invoke.sh'):
                                print(f'invoke_fpath={invoke_fpath}')
                                job = queue.submit(f'bash "{invoke_fpath}"')
                                jobs.append(job)

                            if jobs:
                                for cache_path in fpath.parent.glob('resolved_result_row*'):
                                    queue.submit(f'rm -f "{cache_path}"', depends=jobs)

                            # raise Exception
                            # if 0:
                            # TODO: get commands to rerun the prediction so we
                            # can ensure consistent energy usage reporting
                        # raise Exception

                        chosen_metrics.update(resource_data)

                        chosen_metrics['rank'] = rank
                        agg_row['chosen_row'] = chosen_row
                        agg_row['chosen_metrics'] = chosen_metrics

                        high_level_info[high_level_experiment_key]['available_datasets'].append(
                            agg_row['dataset_name']
                        )

            # agg_row['best_model_fpath'] = best_row[model_column]
            # agg_row['best_metrics'] = best_row[agg.primary_metric_cols].to_dict()

    if len(queue):
        queue.print_commands()
        raise Exception
        # queue.run()

    print(f'high_level_info = {ub.urepr(high_level_info, nl=3)}')

    for agg_row in aggregators_rows:
        if agg_row.get('error', '').startswith('no model col'):
            agg = agg_row['agg']
            raise Exception

    agg_summaries = []
    for agg_row in aggregators_rows:
        agg = agg_row['agg']
        # row_type = agg_row['model_type']
        chosen_metrics = agg_row.get('chosen_metrics', {}) or {}
        summary = agg_row & ['dataset_name', 'model_type', 'node_type', 'dname', 'error']
        summary['len'] = len(agg)
        summary |= chosen_metrics
        agg_summaries.append(summary)
    summary_df = pd.DataFrame(agg_summaries)
    rich.print(summary_df.to_string())

    from geowatch.utils.util_pandas import DataFrame  # NOQA
    summary_df = DataFrame(summary_df)
    summary_df = summary_df[summary_df['dataset_name'] != 'vali_imgs228_20928c8c.kwcoco']
    summary_df = summary_df[summary_df['dataset_name'] != 'train_imgs5747_1e73d54f.kwcoco']

    print('---\nsorted\n---')
    # summary_df = summary_df.sort_values(['node_type', 'dataset_name'])
    # summary_df = summary_df.sort_values(['dataset_name', 'node_type'])
    summary_df = summary_df.sort_values(['dataset_name', 'model_type', 'node_type'])
    rich.print(summary_df.shorten_columns().to_string())
    return summary_df


def parse_bash_invocation(bash_text, with_tokens=False):
    r"""
    Modified from ChatGPT response

    SeeAlso:
        bashlex - https://pypi.org/project/bashlex/

    Example:
        >>> from build_v2_result_table import *  # NOQA
        >>> bash_text = r'''
            # Leading comment
            python -m geowatch.tasks.fusion.evaluate \
                --pred_dataset=pred.kwcoco.zip \
                --true_dataset=/test_imgs30_d8988f8c.kwcoco.zip \
                --eval_dpath=/heatmap_eval/heatmap_eval_id_d0acafbb \
                --age=30 --verbose \
                -o output.txt positional1 -- all rest are positions \
                positional2 --even me --and=me
        >>> '''
        >>> components = parse_bash_invocation(bash_text)
        >>> import pandas as pd
        >>> pd.DataFrame(components)
        >>> components = parse_bash_invocation(bash_text, with_tokens=True)
        >>> pd.DataFrame(components)
    """
    import re
    # Split the bash_text into tokens based on spaces, keeping the structure intact
    import bashlex
    tokens = list(bashlex.split(bash_text.strip()))
    # import shlex
    # bash_text = bash_text.replace('\\\n', ' ')
    # tokens = shlex.split(bash_text)
    # tokens = bash_text.split()
    components = []
    i = 0

    found_start = False
    position_only_mode = False

    while i < len(tokens):
        token = tokens[i]

        if not found_start:
            if token.strip():
                found_start = True
            else:
                # Skip leading blank tokens
                i += 1
                continue

        if token == '--':
            position_only_mode = True
            i += 1
            continue

        handled = False
        if not position_only_mode:
            # Handle --key=value or -key=value
            if re.match(r'--?\w+=', token):
                key, value = token.split('=', 1)
                param_name = key.lstrip('-')
                # args_dict[param_name] = value
                item = {
                    'type': 'keyvalue',
                    'key': param_name,
                    'value': value,
                }
                if with_tokens:
                    item['token_index'] = i
                    item['tokens'] = tokens[i: i + 1]
                handled = True

            # Handle --key value or -key value
            elif token.startswith('--') or token.startswith('-'):
                key = token.lstrip('-')
                if i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                    value = tokens[i + 1]
                    item = {
                        'type': 'keyvalue',
                        'key': key,
                        'value': value,
                    }
                    if with_tokens:
                        item['token_index'] = i
                        item['tokens'] = tokens[i: i + 2]
                    i += 1  # Skip the next token since it's part of the key-value pair
                else:
                    item = {
                        'type': 'flag',
                        'key': key,
                        'value': True,
                    }
                    if with_tokens:
                        item['token_index'] = i
                        item['tokens'] = tokens[i: i + 1]
                handled = True

        # Handle positional arguments
        if not handled:
            item = {
                'type': 'positional',
                'value': token,
            }
            if with_tokens:
                item['token_index'] = i
                item['tokens'] = tokens[i: i + 1]
        components.append(item)
        i += 1
    return components


def parse_invoke_fpath(invoke_fpath):
    invoke_text = invoke_fpath.read_text()
    components = parse_bash_invocation(invoke_text)
    config = {}
    for item in components:
        if item['type'] == 'positional':
            continue
        key = item['key']
        if key == 'm':
            continue
        value = item['value']
        assert key not in config
        config[key] = value
    return config


def build_latex_result_table(summary_df):
    metric_cols = [
        'metrics.heatmap_eval.salient_AP',
        'metrics.heatmap_eval.salient_AUC',
        'metrics.detection_evaluation.ap',
        'metrics.detection_evaluation.auc',
        'metrics.detection_evaluation.max_f1_f1',
        'metrics.detection_evaluation.max_f1_tpr',
        'metrics.heatmap_eval.salient_maxF1_F1',
        'metrics.heatmap_eval.salient_maxF1_tpr',
        # 'kwh',
        # 'duration',
    ]
    summary_df['duration'] = summary_df['duration'].apply(kwutil.timedelta.coerce)

    ignore_dsets = [
        'test_imgs30_d8988f8c.kwcoco',
    ]
    import kwarray
    flags = kwarray.isect_flags(summary_df['dataset_name'], ignore_dsets)
    sub_summary_df = summary_df[~flags]

    piv = sub_summary_df.pivot_table(
        index=[
            'model_type',
            # 'node_type'
        ],
        columns=['dataset_name'],
        values=metric_cols
    )
    piv = piv.round(3)
    num_params_detectron = 43_918_038
    num_params_geowatch = 25_543_369
    num_params_grounding_dino = 172_249_090
    num_params_open_grounding_dino = 172_249_090
    num_params_yolov9 = 50_999_558

    piv['num_params'] = None
    piv.loc['detectron-pretrained', 'num_params'] = num_params_detectron
    piv.loc['detectron-scratch', 'num_params'] = num_params_detectron
    piv.loc['geowatch-scratch', 'num_params'] = num_params_geowatch
    piv.loc['grounding_dino-zero', 'num_params'] = num_params_grounding_dino
    piv.loc['grounding_dino-tuned', 'num_params'] = num_params_open_grounding_dino
    piv.loc['yolo_v9-pretrained', 'num_params'] = num_params_yolov9
    piv.loc['yolo_v9-scratch', 'num_params'] = num_params_yolov9

    colmapper = {
        'metrics.heatmap_eval.salient_AP': 'AP-pixel',
        'metrics.heatmap_eval.salient_AUC': 'AUC-pixel',
        'metrics.detection_evaluation.ap': 'AP-box',
        'metrics.detection_evaluation.auc': 'AUC-box',
        'test_imgs30_d8988f8c.kwcoco': 'test-small',
        'vali_imgs228_20928c8c.kwcoco': 'vali-small',
        'vali_imgs691_99b22ad0.kwcoco': 'vali',
        'test_imgs121_6cb3b6ff.kwcoco': 'test',
        'train_imgs5747_1e73d54f.kwcoco': 'train',

        'detectron-pretrained': 'MaskRCNN-p',
        'detectron-scratch': 'MaskRCNN-s',
        'geowatch-scratch': 'VIT-sseg-s',
        'grounding_dino-tuned': 'GroundingDino-t',
        'grounding_dino-zero': 'GroundingDino-z',
        'yolo_v9-pretrained': 'YOLO-v9-p',
        'yolo_v9-scratch': 'YOLO-v9-s',

        # 'detectron-pretrained': 'MaskRCNN-pretrained',
        # 'detectron-scratch': 'MaskRCNN-scratch',
        # 'geowatch-scratch': 'VIT-sseg-scratch',
        # 'grounding_dino-tuned': 'GroundingDino-tuned',
        # 'grounding_dino-zero': 'GroundingDino-zero',
        # 'yolo_v9-pretrained': 'YOLO-v9-pretrained',
        # 'yolo_v9-scratch': 'YOLO-v9-scratch',

        'metrics.heatmap_eval.salient_maxF1_F1': 'F1-pixel',
        'metrics.detection_evaluation.max_f1_f1': 'F1-box',
        'metrics.heatmap_eval.salient_maxF1_tpr': 'TPR-pixel',
        'metrics.detection_evaluation.max_f1_tpr': 'TPR-box',
        'kwh': 'kWH',
        'duration': 'Duration',
    }
    piv = piv.rename(colmapper, axis=1)
    piv = piv.rename(colmapper, axis=0)

    piv = piv.reorder([
        ('num_params', ''),
        ('AP-box', 'test'),
        ('AUC-box', 'test'),
        ('F1-box', 'test'),
        ('TPR-box', 'test'),
        ('AP-pixel', 'test'),
        ('AUC-pixel', 'test'),
        ('F1-pixel', 'test'),
        ('TPR-pixel', 'test'),
        ('AP-box', 'vali'),
        ('AUC-box', 'vali'),
        ('F1-box', 'vali'),
        ('TPR-box', 'vali'),
        ('AP-pixel', 'vali'),
        ('AUC-pixel', 'vali'),
        ('F1-pixel', 'vali'),
        ('TPR-pixel', 'vali'),
    ], axis=1)
    # rich.print(piv.to_string())

    piv = piv.rename_axis(columns={'dataset_name': 'split'})
    piv = piv.rename_axis(index={'model_type': 'model'})

    try:
        piv[piv.columns.get_level_values(1) == 'Duration'].apply(lambda x: kwutil.timedelta.coerce(x).format())
    except Exception:
        ...

    # import humanize
    # from kwplot.tables import humanize_dataframe
    # print(humanize_dataframe(piv))
    # humanize.intword(piv['num_params'].iloc[0])
    # humanize.intword(piv['num_params'].iloc[2])
    piv = piv.swaplevel(axis=1)

    piv_human = piv.rename({'num_params': '# params'}, axis=1)
    rich.print(piv_human)

    piv.loc[:, piv_human.columns.get_level_values(0) == 'vali']

    print('')
    test_table = piv_human.loc[:, piv_human.columns.get_level_values(0) == 'test']
    vali_table = piv_human.loc[:, piv_human.columns.get_level_values(0) == 'vali']

    vali_text = vali_table.to_latex(index=True, escape=True)
    test_text = test_table.to_latex(index=True, escape=True)

    print(ub.highlight_code(vali_text, 'latex'))
    print(ub.highlight_code(test_text, 'latex'))

    renamed = {
        'AP-box': 'AP\\\\Box',
        'AUC-box': 'AUC\\\\Box',
        'F1-box': 'F1\\\\Box',
        'TPR-box': 'TPR\\\\Box',
        'AP-pixel': 'AP\\\\Pixel',
        'AUC-pixel': 'AUC\\\\Pixel',
        'F1-pixel': 'F1\\\\Pixel',
        'TPR-pixel': 'TPR\\\\Pixel',
        'Duration': 'Duration',
    }
    # rename_map = {
    #     'AP-box': r'\makecell{AP\\Box}',
    #     'AUC-box': r'\makecell{AUC\\Box}',
    #     'F1-box': r'\makecell{F1\\Box}',
    #     'TPR-box': r'\makecell{TPR\\Box}',
    #     'AP-pixel': r'\makecell{AP\\Pixel}',
    #     'AUC-pixel': r'\makecell{AUC\\Pixel}',
    #     'F1-pixel': r'\makecell{F1\\Pixel}',
    #     'TPR-pixel': r'\makecell{TPR\\Pixel}',
    #     'Duration': r'Duration',  # leave as is
    # }

    make_latex_table(vali_table.loc[:, 'vali'])
    latex_friendly_df = vali_table.loc[:, 'vali'].rename(columns=renamed)
    vali_text = make_latex_table(latex_friendly_df)

    # # Example: use your original vali_table
    # styled_vali = bold_max_vals(vali_table.loc[:, 'vali'])

    # # Output to LaTeX using tabulate
    # from tabulate import tabulate
    # latex_str = tabulate(
    #     styled_vali,
    #     headers='keys',
    #     missingval="--"
    # )
    # print(latex_str)

    print('\n\n')
    new_text = piv_human.to_latex(index=True, escape=True)
    print(ub.highlight_code(new_text, 'latex'))

    test_text = make_latex_table(test_table.loc[:, 'test'])

    # vali_text = tabulate(
    #     # bold_max_vals(vali_table.loc[:, 'vali']),
    #     bold_max_vals(latex_friendly_df),
    #     headers='keys',
    #     # tablefmt='latex',
    #     tablefmt='latex_raw',
    #     floatfmt=".2f",
    #     missingval="--"
    # )
    # test_text = tabulate(
    #     bold_max_vals(test_table.loc[:, 'test']),
    #     headers='keys',
    #     # tablefmt='latex',
    #     tablefmt='latex_raw',
    #     floatfmt=".2f",
    #     missingval="--"
    # )
    # print('Validation Table:')
    print('')
    print('(a) Validation (n=691)')
    print('')
    print(vali_text)
    print('')
    print('(b) Test (n=121)')
    print('')
    # print('Test Table:')
    print(test_text)
    # print('')

    # print(piv_human.loc[:, piv_human.columns.get_level_values(0) == 'vali'].to_string().replace('NaN', '---'))

    print('\n\n')
    new_text = piv_human.to_latex(index=True, escape=True)
    print(ub.highlight_code(new_text, 'latex'))

    # text2 = format_results_for_latex2(piv)
    # from ubelt.util_repr import _align_lines
    # lines = piv_human.style.to_latex(
    #     # environment='table*'
    # ).split('\n')
    # new_lines = lines[:2] + _align_lines(lines[2:], '&', pos=None)
    # new_text = '\n'.join(new_lines).replace('#', r'\#')
    # print(new_text)
    return new_text


def bold_max_vals(df):
    """Return a copy of the DataFrame where the max in each column is bolded."""
    import numpy as np
    styled = df.copy().astype(str)

    for col in df.columns:
        # Only operate on numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            max_val = df[col].max()
            styled[col] = df[col].apply(
                lambda x: f"\\textbf{{{x:.2f}}}" if np.isclose(x, max_val) else f"{x:.2f}"
                if pd.notnull(x) else "--"
            )
        else:
            styled[col] = df[col].fillna("--")

    return styled


def make_latex_table(table):
    # Example: use your original vali_table
    styled_table = bold_max_vals(table)
    import tabulate as tabulate_mod
    from functools import partial
    from tabulate import TableFormat, _latex_line_begin_tabular, Line, _latex_row
    tabulate_mod._table_formats['latex_booktabs_raw'] = TableFormat(
        lineabove=partial(_latex_line_begin_tabular, booktabs=True),
        linebelowheader=Line("\\midrule", "", "", ""),
        linebetweenrows=None,
        linebelow=Line("\\bottomrule\n\\end{tabular}", "", "", ""),
        headerrow=partial(_latex_row, escrules={}),
        datarow=partial(_latex_row, escrules={}),
        padding=1,
        with_header_hide=None,
    )

    # Output to LaTeX using tabulate
    from tabulate import tabulate
    latex_str = tabulate(
        styled_table,
        # tablefmt='latex_raw',  # <-- prevents escaping
        tablefmt='latex_booktabs_raw',
        headers='keys',
        missingval="--",
        floatfmt=".3f",
        showindex=False,
    )
    return latex_str
    # print(latex_str)


def format_results_for_latex2(piv: pd.DataFrame):
    """
    Format a pivoted result DataFrame into a LaTeX tabular environment.

    Parameters:
        piv (pd.DataFrame): Multi-level column DataFrame with results.
        test_name (str): The name of the test column level.
        vali_name (str): The name of the validation column level.
        metrics (List[str]): The list of metrics to extract per split.
        bold_best (bool): Whether to bold the best result per metric/split.
        param_col (str): Name of the parameter count column.
        param_format (str): Format string for parameter column.
        float_fmt (str): Format string for floats.
    """
    # Hardcoded config
    test_name = 'test'
    vali_name = 'vali'
    param_col = ('', 'num_params')
    param_fmt = '{:.1e}'
    float_fmt = '{:.3f}'
    missing_str = '--'
    metrics = ['AP-box', 'AUC-box', 'AP-pixel', 'AUC-pixel', 'F1-box', 'F1-pixel']
    splits = [test_name, vali_name]

    piv = piv.copy()
    model_names = piv.index.tolist()

    # Format param column
    piv['FormattedParams'] = piv[param_col].apply(lambda x: param_fmt.format(x))

    # Determine bold entries per metric/split
    bold_flags = {}
    for split in splits:
        for metric in metrics:
            col = (split, metric)
            if col in piv.columns:
                vals = piv[col]
                best_val = vals.max()
                is_best = vals == best_val
                bold_flags[col] = is_best

    model_names = piv.index.tolist()
    rows = []

    # Build LaTeX rows
    latex_rows = []
    for model in model_names:
        model_short = model.replace('MaskRCNN-pretrained', 'MaskRCNN-p') \
                           .replace('MaskRCNN-scratch', 'MaskRCNN-s') \
                           .replace('VIT-sseg-scratch', 'VIT-s') \
                           .replace('GroundingDino-tuned', 'GD-t') \
                           .replace('GroundingDino-zero', 'GD-0') \
                           .replace('YOLO-v9-pretrained', 'YOLOv9-p')
        row = [model_short, piv.loc[model, 'FormattedParams']]
        for split in splits:
            for metric in metrics:
                col = (split, metric)
                if col in piv.columns:
                    val = piv.loc[model, col]
                    if pd.isna(val):
                        row.append(missing_str)
                    else:
                        val_str = float_fmt.format(val)
                        if bold_flags.get(col, pd.Series(False, index=piv.index)).get(model, False):
                            val_str = f'\\tb{{{val_str}}}'
                        row.append(val_str)
                else:
                    row.append(missing_str)
        latex_rows.append(" & ".join(row) + r" \\")

    header = ub.codeblock(
        r"""
        \begin{tabular}{ll rrrr rrrr}
        \toprule
        \multicolumn{2}{c}{Dataset split:} & \multicolumn{4}{c}{Test (n=121)} & \multicolumn{4}{c}{Validation (n=691)} \\
        \multicolumn{2}{c}{Evaluation type:} & AP & AUC & AP & AUC & AP & AUC & AP & AUC \\
        Model type & \# Params & Box & Box & Pixel & Pixel & Box & Box & Pixel & Pixel \\
        \midrule
        """)

    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    footer = r"\bottomrule\n\end{tabular}"
    text = header + body + footer
    return text


def find_corresponding_test121_imgs():
    import kwcoco
    dset1 = kwcoco.CocoDataset('/home/joncrall/code/shitspotter/shitspotter_dvc/test_imgs30_d8988f8c.kwcoco.zip')
    dset2 = kwcoco.CocoDataset('/home/joncrall/code/shitspotter/shitspotter_dvc/test_imgs121_6cb3b6ff.kwcoco.zip')
    gids = [15, 8, 10, 23, 26, 30, 24, 1, 6, 4]
    new_gids = []
    for gid in gids:
        img1 = dset1.coco_image(gid)
        new_id = dset2.index.name_to_img[img1.name]['id']
        new_gids.append(new_id)


def gather_heatmap_figures(aggregators_rows):
    # Now: get comparable figures
    heatmap_figure_rows = []
    detection_figure_rows = []
    for agg_row in aggregators_rows:
        try:
            if agg_row['node_type'] == 'heatmap_eval':
                heatmap_figure_rows.append(agg_row)
            if agg_row['node_type'] == 'detection_evaluation':
                detection_figure_rows.append(agg_row)
        except Exception as ex:
            print(f'ex = {ub.urepr(ex, nl=1)}')

    split_to_gids_of_interest = {
        'test_imgs30_d8988f8c.kwcoco': [15, 8, 10, 23, 26, 30, 24, 1, 6, 4],
        'test_imgs121_6cb3b6ff.kwcoco': [10, 6, 11, 23, 3, 110, 1, 115, 117, 119],
        'vali_imgs691_99b22ad0.kwcoco': [
            # 5513,
            12, 113, 156, 99, 90,
            # 5460,
            177, 18, 121, 74],
    }

    # dpaths = []
    # dataset_split = 'test_imgs30_d8988f8c.kwcoco'
    dataset_split = 'test_imgs121_6cb3b6ff.kwcoco'
    heatmap_split_to_rows = ub.group_items(heatmap_figure_rows, key=lambda x: x['dataset_name'])
    detection_split_to_rows = ub.group_items(detection_figure_rows, key=lambda x: x['dataset_name'])
    dataset_splits = [
        # 'test_imgs30_d8988f8c.kwcoco',
        'test_imgs121_6cb3b6ff.kwcoco',
        'vali_imgs691_99b22ad0.kwcoco',
    ]

    model_types = [
        'geowatch-scratch',
        'detectron-pretrained',
        'detectron-scratch',
        'yolo_v9-scratch',
        'yolo_v9-pretrained',
        'grounding_dino-zero',
        'grounding_dino-tuned',
    ]
    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=4, name='viz-compute')

    redraw_detection_confusion = True

    for dataset_split in dataset_splits:

        # Ensure detection confusion viz are computed for relevant images
        detection_rows = detection_split_to_rows[dataset_split]
        gids = split_to_gids_of_interest[dataset_split]
        for agg_row in detection_rows:
            agg = agg_row['agg']
            # if 'chosen_row' not in agg_row:
            #     continue
            fpath = ub.Path(agg_row['chosen_row']['fpath'])
            eval_dpath = fpath.parent
            viz_dpath = eval_dpath / '_viz_paper_confusion'
            # print(f'viz_dpath={viz_dpath}')

            cfsn_fpath = eval_dpath / 'confusion.kwcoco.zip'
            dep = None
            if not cfsn_fpath.exists():
                print(f'no confusion, need it in {eval_dpath}')
                eval_invoke = eval_dpath / 'invoke.sh'
                assert eval_invoke.exists()
                text = eval_invoke.read_text()
                if 'confusion_path' not in text:
                    text = text.rstrip() + f' --confusion_fpath={eval_dpath}/confusion.kwcoco.zip'
                # print(text)
                dep = queue.submit(text)

            if not viz_dpath.exists() or redraw_detection_confusion:
                select_img_query = kwutil.Json.dumps(gids)
                viz_cmd = ub.paragraph(
                    f'''
                    geowatch visualize
                        --src {cfsn_fpath}
                        --viz_dpath {viz_dpath}
                        --select_images '{select_img_query}'
                        --darken_images 0.7
                        --draw_imgs=False --draw_anns=True \
                        --max_dim=448 --draw_chancode=False --draw_header=False
                    ''')
                queue.submit(viz_cmd, depends=dep)
            agg_row['viz_dpath'] = viz_dpath

        # if len(queue):
        #     queue.run(block=True)
        # for dataset_split in dataset_splits:

        # Ensure visual heatmaps are computed for relevant images
        heatmap_rows = heatmap_split_to_rows[dataset_split]
        gids = split_to_gids_of_interest[dataset_split]
        for agg_row in heatmap_rows:
            agg = agg_row['agg']

            heatmap_node = agg.dag.nodes['heatmap_eval']
            fpath = ub.Path(agg_row['chosen_row']['fpath'])
            node_dpath = fpath.parent
            if 1:
                found = list(node_dpath.glob('.pred/*/*'))
                assert len(found) == 1
                dpath = found[0]
                fpath = dpath / 'invoke.sh'
                # print(fpath)
                fpath = node_dpath / 'invoke.sh'
                # print(fpath)

            # print(node_dpath)
            viz_dpath = node_dpath / '_viz'
            viz_fpath = viz_dpath / 'viz_eval.json'
            agg_row['viz_dpath'] = viz_dpath

            if not viz_fpath.exists():
                invoke_fpath = (node_dpath / 'invoke.sh')
                assert invoke_fpath.exists()
                config = parse_invoke_fpath(invoke_fpath)
                config['viz_thresh'] = 0.5
                config['eval_dpath'] = viz_dpath
                config['eval_fpath'] = viz_fpath
                config['draw_legend'] = False
                config['workers'] = 6
                config['draw_heatmaps'] = True
                config['draw_components'] = True
                config['draw_burnin'] = False
                config['select_images'] = f'[.id] | inside({gids})'

                heatmap_node.executable = 'python -m kwcoco.metrics.segmentation_metrics'
                heatmap_node.configure(config)

                viz_dpath.ensuredir()
                new_invoke_fpath = (viz_dpath / 'invoke.sh')
                new_invoke_fpath.write_text(heatmap_node.command)
                # print(heatmap_node.command)
                queue.submit(heatmap_node.command)

    if len(queue):
        queue.print_commands()
        if 1:
            queue.run(block=True)
        # raise NotImplementedError

    for dataset_split in dataset_splits:

        SUPER_HACK_FOR_ASSOCIATION = 1
        if SUPER_HACK_FOR_ASSOCIATION:
            import kwcoco
            dset = kwcoco.CocoDataset(ub.Path('~/code/shitspotter/shitspotter_dvc').expand() / (dataset_split + '.zip'))

        component_rows = []
        heatmap_rows = heatmap_split_to_rows[dataset_split]
        gids = split_to_gids_of_interest[dataset_split]
        # Read heatmap confusion images for requested image ids.
        for agg_row in heatmap_rows:
            viz_dpath = agg_row['viz_dpath']
            img_fpath = (viz_dpath / 'heatmaps/_components')
            gid_to_component_dpath = {}
            for dpath in img_fpath.ls():
                gid = int(dpath.name.split('_')[-1])
                gid_to_component_dpath[gid] = dpath
            for gid in gids:
                dpath = gid_to_component_dpath[gid]
                component_rows.append({
                    'model_type': agg_row['model_type'],
                    'viz_type': 'heatmap_confusion_components',
                    'path': dpath,
                    'gid': gid,
                })

        # Read detection confusion images for requested image ids.
        detection_rows = detection_split_to_rows[dataset_split]
        for agg_row in detection_rows:
            if 'viz_dpath' not in agg_row:
                continue
            viz_dpath = agg_row['viz_dpath']
            img_dpath = viz_dpath.ls('*/_anns/red_green_blue')[0]
            # img_fpath = (viz_dpath / 'heatmaps/_components')
            gid_to_component_fpath = {}

            for fpath in img_dpath.ls():
                # This is actually the frame index, which I
                # think will agree with the query order. Hopefully.
                # Nope, hack it
                if SUPER_HACK_FOR_ASSOCIATION:
                    images = dset.images()
                    found = None
                    for imgid, name in zip(images, images.lookup('name')):
                        if name in fpath.name:
                            assert found is None
                            found = imgid
                    gid = found
                else:
                    frame_index = int(fpath.name.split('_')[0])
                    gid = gids[frame_index]
                gid_to_component_fpath[gid] = fpath
            for gid in gids:
                fpath = gid_to_component_fpath[gid]
                component_rows.append({
                    'model_type': agg_row['model_type'],
                    'viz_type': 'detection_confusion',
                    'path': fpath,
                    'gid': gid,
                })
        # print(pd.DataFrame(component_rows).to_string())

        # Using these to access the input images
        hack_heat_subrows = None

        vert_parts = []
        figure_dpath = ub.Path('$HOME/code/shitspotter/papers/neurips-2025/figures').expand()
        split_dpath = (figure_dpath / 'agg_viz_results2/' / dataset_split).ensuredir()
        split_fpaths = {}
        model_type_to_rows = ub.group_items(component_rows, key=lambda x: x['model_type'])
        for model_type in ub.ProgIter(model_types):
            rows = model_type_to_rows[model_type]
            viz_type_to_subrows = ub.group_items(rows, key=lambda x: x['viz_type'])
            if len(viz_type_to_subrows) == 2:
                # hack: for items with both, only show heatmaps
                viz_type_to_subrows.pop('detection_confusion')
            for viz_type, subrows in viz_type_to_subrows.items():
                if viz_type == 'heatmap_confusion_components':
                    hack_heat_subrows = subrows
                gid_to_row = {r['gid']: r for r in subrows}
                hzparts = []
                for gid in gids:
                    row = gid_to_row[gid]
                    if row['viz_type'] == 'heatmap_confusion_components':
                        dpath = row['path']
                        confusion = kwimage.imread(dpath / 'saliency_confusion.jpg')
                        pred = kwimage.imread(dpath / 'salient_pred.jpg')
                        confusion = kwimage.imresize(confusion, dsize=(None, 1024))
                        pred = kwimage.imresize(pred, dsize=(None, 1024))
                        stack = kwimage.stack_images([confusion, pred], axis=0)
                    elif row['viz_type'] == 'detection_confusion':
                        fpath = row['path']
                        confusion = kwimage.imread(fpath)
                        stack = kwimage.imresize(confusion, dsize=(None, 1024))

                    hzparts.append(stack)
                model_canvas = kwimage.stack_images(hzparts, axis=1, pad=10, bg_value='white')
                fpath = split_dpath / f'results_{model_type}_{viz_type}.jpg'
                # print(f'fpath={fpath}')
                kwimage.imwrite(fpath, model_canvas)
                split_fpaths[model_type] = fpath
                vert_parts.append(model_canvas)

        if 1:
            # hack
            gid_to_row = {r['gid']: r for r in hack_heat_subrows}
            hzparts = []
            for gid in gids:
                row = gid_to_row[gid]
                dpath = row['path']
                input_img = kwimage.imread(dpath / 'input_image.jpg')
                input_img = kwimage.imresize(input_img, dsize=(None, 1024))
                hzparts.append(input_img)
            input_img_canvas = kwimage.stack_images(hzparts, axis=1, pad=10, bg_value='white')
            fpath = split_dpath / 'results_input_images.jpg'
            split_fpaths['__input__'] = fpath
            kwimage.imwrite(fpath, input_img_canvas)
            vert_parts.append(input_img_canvas)

        if 0:
            stack = kwimage.stack_images(vert_parts, axis=0, pad=100, bg_value='white')
            stack.shape
            import kwplot
            kwplot.autompl()
            kwplot.imshow(stack)

        subcaption_mapper = {
            'detectron-pretrained': 'MaskRCNN-pretrained',
            'detectron-scratch': 'MaskRCNN-scratch',
            'geowatch-scratch': 'VIT-sseg-scratch',
            'grounding_dino-tuned': 'GroundingDino-tuned',
            'grounding_dino-zero': 'GroundingDino-zero',
            'yolo_v9-pretrained': 'YOLO-v9-pretrained',
            'yolo_v9-scratch': 'YOLO-v9-scratch',
        }

        rel_dpath = ub.Path('$HOME/code/shitspotter/papers/neurips-2025').expand()
        parts = []
        header = ub.codeblock(
            r'''
            \begin{figure*}[ht]
            \centering
            ''')
        parts.append(header)
        index = 0

        split_name = 'validation' if 'vali' in dataset_split else 'Test'
        for key, fpath in split_fpaths.items():
            rel_path = fpath.relative_to(rel_dpath)
            line = r'\includegraphics[width=1.0\textwidth]{' + str(rel_path) + '}%'
            parts.append(line)
            parts.append(r'\hfill')
            ordinal = chr(index + 97)
            if key == '__input__':
                subcaption = f'Inputs from the {split_name} set'
            else:
                subcaption = subcaption_mapper[key] + f' ({split_name} set results)'
            parts.append('(' + ordinal + ') ' + subcaption)
            index += 1

        from kwutil import partial_format
        footer = partial_format.subtemplate(ub.codeblock(
            r'''
            \caption[]{
                Qualitative results using the top-performing models on the validation set.
            }
            \label{fig:${split_name}_results_all_models2}
            \end{figure*}
            '''), **locals())

        parts.append(footer)
        text = '\n'.join(parts)
        print(text)

    # for key, fpaths in (model_type_to_fpaths).items():
    #     images = [kwimage.imread(p) for p in fpaths]
    #     canvas = kwimage.stack_images(images, axis=1, resize='larger', pad=70, bg_value='white')
    #     canvas = kwimage.imresize(canvas, max_dim=4096)
    #     if key == 'detectron-scratch':
    #         kwplot.imshow(canvas[15:,...][:-340, ...])
    #         ...

    # for dpath in dpaths:
    #     import xdev
    #     xdev.startfile(dpath)

    # /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_test/eval/flat/heatmap_eval/heatmap_eval_id_29ec800f/pxl_eval.json
    figure_dpath = ub.Path('$HOME/code/shitspotter/papers/neurips-2025/figures').expand()


def report_resources(aggregators_rows):
    node_types = [
        'detectron_pred',
        'heatmap_pred',
        'heatmap_eval',
        'extract_polygons',
        'heatmap_pred_fit',
        'open_grounding_dino_pred',
        'grounding_dino_pred',
        'yolo_pred',
    ]
    dfs = []
    for agg_row in aggregators_rows:
        agg = agg_row['agg']

        for col in agg.table.search_columns('resources'):
            print(f'col={col}')
            assert col.split('.')[1] in node_types
        for col in agg.table.search_columns('context'):
            print(f'col={col}')
            assert col.split('.')[1] in node_types

        for node_type in node_types:
            cols = agg.table.search_columns(f'context.{node_type}.')
            cols += agg.table.search_columns(f'resources.{node_type}.')
            if cols:
                subtable = agg.table[cols]
                subtable = subtable.shorten_columns()
                subtable['dname'] = agg_row['dname']
                subtable['dataset_name'] = agg_row['dataset_name']
                subtable['node_type'] = node_type
                subtable = subtable.reorder(['node_type', 'dname', 'dataset_name'], axis=1)
                dfs.append(subtable)

        # df = agg.resource_summary_table()
        # dfs.append(df)

    combo = pd.concat(dfs)
    combo['node_type'].value_counts()
    combo = combo[combo['node_type'] != 'heatmap_pred_fit']
    if 0:
        for _, group in combo.groupby('uuid'):
            if len(group) > 1:
                print(group)
    combo = combo[~combo['duration'].isna()]
    deduped = combo.groupby('uuid').agg('first')
    deduped['duration'] = deduped['duration'].apply(kwutil.timedelta.coerce)

    grouped = deduped.groupby(['node_type', 'dname', 'dataset_name'])
    print('time', kwutil.timedelta.coerce(deduped['duration'].sum()).to('pint').to('days'))
    # deduped['duration'].sum()
    print('kwh', deduped['kwh'].sum())
    print('co2', deduped['co2_kg'].sum())

    grouped['duration'].sum()
    grouped['kwh'].sum()
    grouped['co2_kg'].sum()

    detectron = deduped[deduped['dname'].str.contains('detectron')]
    print('time', kwutil.timedelta.coerce(detectron['duration'].sum()).to('pint').to('days'))
    print('kwh', detectron['kwh'].sum())
    print('co2', detectron['co2_kg'].sum())

    for key, group in list(deduped.groupby(['node_type', 'dname', 'dataset_name'])):
        print(key)
        print('  time', kwutil.timedelta.coerce(group['duration'].sum()).to('pint').to('days'))
        print('  kwh', group['kwh'].sum())
        print('  co2', group['co2_kg'].sum())


def get_prompt_variation_results(aggregators_rows):
    captions = {
        'test_imgs121_6cb3b6ff.kwcoco': 'Test (n=121)',
        'vali_imgs691_99b22ad0.kwcoco': 'Validation (n=691)',
    }
    header = ub.codeblock(
        r'''
        \begin{table*}[t]
        \caption{Zero-shot detection results as a function of prompt. Prompt variation has a significant impact on scores, but overall zero-shot results are all low scoring. }
        \label{tab:prompt_variations}
        \centering
        ''')

    footer = ub.codeblock(
         r'''
         \end{table*}
         ''')

    parts = []

    for _agg_row in aggregators_rows[::-1]:
        _agg = _agg_row['agg']
        if '_shitspotter_2025_rebutal_evals' in _agg.output_dpath.parts:
            agg_row = _agg_row
            dataset_name = agg_row['dataset_name']
            agg = _agg
            agg.table['resolved_params.grounding_dino_pred.src']
            sub_columns = [
                'params.grounding_dino_pred.classes',
                'metrics.detection_evaluation.ap',
                'metrics.detection_evaluation.auc',
                'metrics.detection_evaluation.max_f1_f1',
                'metrics.detection_evaluation.max_f1_tpr',
            ]
            subtable = agg.table[sub_columns].shorten_columns()
            subtable = subtable.sort_values('ap')
            subtable['classes'] = subtable['classes'].apply(lambda x: x.replace('[', '').replace(']', ''))
            rich.print(f'Dataset Name: {dataset_name}')
            # rich.print(rich.markup.escape(subtable.to_string()))
            subtable_human = subtable.rename({
                'classes': 'Prompt',
                'ap': 'AP',
                'auc': 'AUC',
                'max_f1_tpr': 'TPR',
                'max_f1_f1': 'F1',
            }, axis=1)
            # bolded = bold_max_vals(subtable_human)
            # print(bolded.to_latex(index=False, escape=False))
            # print(bolded.style.to_latex())
            tabular_text = make_latex_table(subtable_human)

            caption = captions[dataset_name]
            subheader = ub.codeblock(
                r'''
                \begin{subtable}[b]{\textwidth} % Adjust width as needed
                  \caption{''' + caption + r'''}
                  \centering
                ''')

            subfooter = ub.codeblock(
                r'''
                \end{subtable}
                ''')

            subtable_parts = [subheader, ub.indent(tabular_text), subfooter]
            subtable_text = '\n'.join(subtable_parts)
            parts.append(subtable_text)
    text = header + '\n' + '\n\n\\hfill\n\n'.join(parts) + '\n' + footer
    print(text)


def main():
    aggregators_rows = load_aggregators()
    summary_df = process_aggregators(aggregators_rows)

    get_prompt_variation_results(aggregators_rows)

    new_text = build_latex_result_table(summary_df)
    print(new_text)
    report_resources(aggregators_rows)
    gather_heatmap_figures(aggregators_rows)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/papers/neurips-2025/scripts/build_v2_result_table.py
    """
    main()


"""
Fixup:
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/.pred/heatmap_pred/heatmap_pred_id_3dc3d393/invoke.sh
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/invoke.sh

/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals_2024_v2/eval/flat/heatmap_eval/heatmap_eval_id_39f7705b/.pred/heatmap_pred/heatmap_pred_id_11a71f31/invoke.sh
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals_2024_v2/eval/flat/heatmap_eval/heatmap_eval_id_39f7705b/invoke.sh

bash /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_test/eval/flat/heatmap_eval/heatmap_eval_id_29ec800f/.pred/detectron_pred/detectron_pred_id_49138b7b/invoke.sh
bash /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v4_test/eval/flat/heatmap_eval/heatmap_eval_id_d0acafbb/.pred/detectron_pred/detectron_pred_id_29e9c269/invoke.sh
bash /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals/eval/flat/heatmap_eval/heatmap_eval_id_b72a6bd3/.pred/detectron_pred/detectron_pred_id_797fc75a/invoke.sh
bash /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v4/eval/flat/heatmap_eval/heatmap_eval_id_91a45b0f/.pred/detectron_pred/detectron_pred_id_a53ddd08/invoke.sh

bash /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_test/eval/flat/heatmap_eval/heatmap_eval_id_29ec800f/invoke.sh
bash /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v4_test/eval/flat/heatmap_eval/heatmap_eval_id_d0acafbb/invoke.sh
bash /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals_v4/eval/flat/heatmap_eval/heatmap_eval_id_91a45b0f/invoke.sh
bash /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_detectron_evals/eval/flat/heatmap_eval/heatmap_eval_id_b72a6bd3/invoke.sh

"""
