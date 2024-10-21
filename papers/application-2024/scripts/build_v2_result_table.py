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
    """
    dvc_expt_dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/')

    inputs = kwutil.Yaml.coerce(
        '''
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
        ''')

    aggregators_rows = []
    for row in inputs:
        row['target'] = dvc_expt_dpath / row['dname']
        config = AggregateEvluationConfig(**{
            'target': [row['target']],
            'pipeline': row['pipeline'],
            'eval_nodes': ['heatmap_eval', 'detection_evaluation'],
            'primary_metric_cols': 'auto',
            'display_metric_cols': 'auto',
            'io_workers': 16,
            'cache_resolved_results': True,
            'output_dpath': row['target'] / 'aggregate',
            # '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals_2024_v2/aggregate',
            # 'query': "df['resolved_params.heatmap_pred_fit.trainer.default_root_dir'].apply(lambda p: str(p).split('/')[-1]).str.contains('noboxes')",
        })
        eval_type_to_aggregator = config.coerce_aggregators()
        for agg in eval_type_to_aggregator.values():
            agg_row = {
                'agg': agg,
                'node_type': agg.node_type,
                **row
            }
            agg_row = ub.udict(agg_row)
            if len(agg) > 0:
                aggregators_rows.append(agg_row)

    for agg_row in aggregators_rows:
        agg_row['model_type'] = 'detectron' if 'detectron' in agg_row['pipeline'] else 'geowatch'
        agg = agg_row['agg']
        row_type = agg_row['model_type']
        if agg.node_type == 'detection_evaluation':
            if row_type == 'geowatch':
                test_dsets = agg.table['resolved_params.heatmap_pred.test_dataset'].unique()
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
        # 'vali_imgs228_20928c8c.kwcoco',
    }
    tmp = []
    for r in aggregators_rows:
        if r['dataset_name'] not in dataset_name_blocklist:
            tmp.append(r)
    aggregators_rows = tmp
    return aggregators_rows


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
            model_type = 'geowatch-scratch'
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

    group_to_chosen_params = {}
    dataset_groups = ub.group_items(aggregators_rows, lambda agg: agg['dataset_name'])
    vali_aggs_rows = dataset_groups['vali_imgs691_99b22ad0.kwcoco']
    for agg_row in vali_aggs_rows:
        agg = agg_row['agg']
        key = (agg_row['model_type'], agg_row['node_type'])

        cands = agg.table.search_columns('checkpoint_fpath') + agg.table.search_columns('package_fpath') + agg.table.search_columns('checkpoint_path')
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

        cands = agg.table.search_columns('checkpoint_fpath') + agg.table.search_columns('package_fpath') + agg.table.search_columns('checkpoint_path')
        if cands:
            model_column = cands[0]
            agg_row['model_column'] = model_column
            primary_metric = agg.primary_metric_cols[0]
            models = agg.table[model_column]
            models = agg.table[primary_metric]
            ranked = agg.table[primary_metric].sort_values(ascending=False)
            top_loc = ranked.index[0]
            best_row = agg.table.loc[top_loc]
            agg_row['best_model_fpath'] = best_row[model_column]
            agg_row['best_metrics'] = best_row[agg.primary_metric_cols].to_dict()
            chosen_params = ub.udict(best_row.to_dict()) & {
                model_column, 'resolved_params.extract_polygons.thresh',
            }
            print(f'key={key}')
            assert key not in group_to_chosen_params
            group_to_chosen_params[key] = chosen_params
        else:
            print(f'Miss: key={key}')

    for agg_row in aggregators_rows:
        agg = agg_row['agg']
        cands = agg.table.search_columns('checkpoint_fpath') + agg.table.search_columns('package_fpath')
        if not cands:
            print('row does have registered models')
        else:
            model_column = cands[0]
            primary_metric = agg.primary_metric_cols[0]
            models = agg.table[model_column]
            ranked = agg.table[primary_metric].sort_values(ascending=False)
            # model_ranks = len(agg.table) - agg.table[primary_metric].argsort()
            model_ranks = ub.dzip(agg.table[primary_metric].sort_values(ascending=False).index, range(len(agg.table)))
            # top_loc = ranked.index[0]
            # best_row = agg.table.loc[top_loc]
            key = (agg_row['model_type'], agg_row['node_type'])
            if key not in group_to_chosen_params:
                print('Unable to find model relevant to this node')
                agg_row['error'] = f'no model col {key}'
            else:
                chosen_params = group_to_chosen_params[key]
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
                            print(f'Warning: should not happen: {nfound=}, but might if something was rerun with a new param')

                        locs = models[flags].index
                        ranks = [model_ranks[loc] for loc in locs]
                        chosen_rows = agg.table.loc[locs]
                        if nfound > 1:
                            # should at least have the same metrics
                            assert chosen_rows[agg.primary_metric_cols].apply(ub.allsame).all()

                        rank = ranks[0]
                        chosen_row = chosen_rows.iloc[0]
                        chosen_metrics = chosen_row[agg.primary_metric_cols].to_dict()
                        chosen_metrics['rank'] = rank
                        agg_row['chosen_row'] = chosen_row
                        agg_row['chosen_metrics'] = chosen_metrics

            # agg_row['best_model_fpath'] = best_row[model_column]
            # agg_row['best_metrics'] = best_row[agg.primary_metric_cols].to_dict()

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
    ]
    piv = summary_df.pivot_table(
        index=[
            'model_type',
            # 'node_type'
        ],
        columns=['dataset_name'],
        values=metric_cols
    )

    colmapper = {
        'metrics.heatmap_eval.salient_AP': 'AP-pixel',
        'metrics.heatmap_eval.salient_AUC': 'AUC-pixel',
        'metrics.detection_evaluation.ap': 'AP-box',
        'metrics.detection_evaluation.auc': 'AUC-box',
        'test_imgs30_d8988f8c.kwcoco': 'test',
        'vali_imgs228_20928c8c.kwcoco': 'vali-small',
        'vali_imgs691_99b22ad0.kwcoco': 'vali',
        'train_imgs5747_1e73d54f.kwcoco': 'train',
        'detectron-pretrained': 'MaskRCNN-pretrained',
        'detectron-scratch': 'MaskRCNN-scratch',
        'geowatch-scratch': 'VIT-sseg-scratch',
    }
    piv = piv.rename(colmapper, axis=1)
    piv = piv.rename(colmapper, axis=0)
    piv = piv.round(3)

    num_params_detectron = 43_918_038
    num_params_geowatch = 25_543_369
    piv['num_params'] = None
    piv = piv.reorder([
        ('num_params', ''),
        ('AP-box', 'test'),
        ('AUC-box', 'test'),
        ('AP-pixel', 'test'),
        ('AUC-pixel', 'test'),
        ('AP-box', 'vali'),
        ('AUC-box', 'vali'),
        ('AP-pixel', 'vali'),
        ('AUC-pixel', 'vali'),
    ], axis=1)
    rich.print(piv.to_string())
    piv.loc['MaskRCNN-pretrained', 'num_params'] = num_params_detectron
    piv.loc['MaskRCNN-scratch', 'num_params'] = num_params_detectron
    piv.loc['VIT-sseg-scratch', 'num_params'] = num_params_geowatch
    piv = piv.rename_axis(columns={'dataset_name': 'split'})
    piv = piv.rename_axis(index={'model_type': 'model'})

    # import humanize
    from kwplot.tables import humanize_dataframe
    print(humanize_dataframe(piv))
    # humanize.intword(piv['num_params'].iloc[0])
    # humanize.intword(piv['num_params'].iloc[2])
    piv = piv.swaplevel(axis=1)

    piv_human = piv.rename({'num_params': '# params'}, axis=1)
    rich.print(piv_human)
    print('\n\n')
    new_text = piv_human.to_latex(index=True, escape=True)
    print(ub.highlight_code(new_text, 'latex'))

    # from ubelt.util_repr import _align_lines
    # lines = piv_human.style.to_latex(
    #     # environment='table*'
    # ).split('\n')
    # new_lines = lines[:2] + _align_lines(lines[2:], '&', pos=None)
    # new_text = '\n'.join(new_lines).replace('#', r'\#')
    # print(new_text)
    return new_text


def gather_heatmap_figures(aggregators_rows):
    # Now: get comparable figures
    heatmap_figure_rows = []
    for agg_row in aggregators_rows:
        try:
            if agg_row['node_type'] == 'heatmap_eval':
                heatmap_figure_rows.append(agg_row)
        except Exception as ex:
            print(f'ex = {ub.urepr(ex, nl=1)}')

    split_to_gids_of_interest = {
        'test_imgs30_d8988f8c.kwcoco': [15, 8, 10, 23, 26, 30, 24, 1, 6, 4],
        'vali_imgs691_99b22ad0.kwcoco': [
            # 5513,
            12, 113, 156, 99, 90,
            # 5460,
            177, 18, 121, 74],
    }

    # dpaths = []
    dataset_split = 'test_imgs30_d8988f8c.kwcoco'
    split_to_rows = ub.group_items(heatmap_figure_rows, key=lambda x: x['dataset_name'])
    dataset_splits = [
        'test_imgs30_d8988f8c.kwcoco',
        'vali_imgs691_99b22ad0.kwcoco',
    ]

    model_types = [
        'detectron-pretrained',
        'detectron-scratch',
        'geowatch-scratch'
    ]

    for dataset_split in dataset_splits:

        # Ensure visual heatmaps are computed for relevant images
        import cmd_queue
        queue = cmd_queue.Queue.create(backend='tmux', size=4)
        rows = split_to_rows[dataset_split]
        gids = split_to_gids_of_interest[dataset_split]

        for agg_row in rows:
            agg = agg_row['agg']
            heatmap_node = agg.dag.nodes['heatmap_eval']
            fpath = ub.Path(agg_row['chosen_row']['fpath'])
            node_dpath = fpath.parent
            if 1:
                found = list(node_dpath.glob('.pred/*/*'))
                assert len(found) == 1
                dpath = found[0]
                fpath = dpath / 'invoke.sh'
                print(fpath)
                fpath = node_dpath / 'invoke.sh'
                print(fpath)

            print(node_dpath)
            viz_dpath = node_dpath / '_viz'
            viz_fpath = viz_dpath / 'viz_eval.json'
            agg_row['viz_dpath'] = viz_dpath

            if not viz_fpath.exists() or 0:
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
                print(heatmap_node.command)
                queue.submit(heatmap_node.command)

        if len(queue):
            queue.run(block=False)
            raise NotImplementedError

        component_rows = []
        for agg_row in rows:
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
                    'dpath': dpath,
                    'gid': gid,
                })

        vert_parts = []
        figure_dpath = ub.Path('$HOME/code/shitspotter/papers/application-2024/figures').expand()
        split_dpath = (figure_dpath / 'agg_viz_results/' / dataset_split).ensuredir()
        split_fpaths = {}
        model_type_to_rows = ub.group_items(component_rows, key=lambda x: x['model_type'])
        for model_type in ub.ProgIter(model_types):
            rows = model_type_to_rows[model_type]
            gid_to_row = {r['gid']: r for r in rows}
            hzparts = []
            for gid in gids:
                row = gid_to_row[gid]
                dpath = row['dpath']
                confusion = kwimage.imread(dpath / 'saliency_confusion.jpg')
                pred = kwimage.imread(dpath / 'salient_pred.jpg')
                confusion = kwimage.imresize(confusion, dsize=(None, 1024))
                pred = kwimage.imresize(pred, dsize=(None, 1024))
                stack = kwimage.stack_images([confusion, pred], axis=0)
                hzparts.append(stack)
            model_canvas = kwimage.stack_images(hzparts, axis=1, pad=10, bg_value='white')
            fpath = split_dpath / f'results_{model_type}.jpg'
            kwimage.imwrite(fpath, model_canvas)
            split_fpaths[model_type] = fpath
            vert_parts.append(model_canvas)

        if 1:
            # hack
            gid_to_row = {r['gid']: r for r in rows}
            hzparts = []
            for gid in gids:
                row = gid_to_row[gid]
                dpath = row['dpath']
                input_img = kwimage.imread(dpath / 'input_image.jpg')
                input_img = kwimage.imresize(input_img, dsize=(None, 1024))
                hzparts.append(input_img)
            input_img_canvas = kwimage.stack_images(hzparts, axis=1, pad=10, bg_value='white')
            fpath = split_dpath / 'results_input_images.jpg'
            split_fpaths['__input__'] = fpath
            kwimage.imwrite(fpath, input_img_canvas)
            vert_parts.append(input_img_canvas)

            # stack = kwimage.stack_images(vert_parts, axis=0, pad=100, bg_value='white')
            # stack.shape
            # import kwplot
            # kwplot.imshow(stack)

        rel_dpath = ub.Path('$HOME/code/shitspotter/papers/application-2024').expand()
        parts = []
        header = ub.codeblock(
            r'''
            \begin{figure*}[ht]
            \centering
            ''')
        parts.append(header)
        index = 0
        for key, fpath in split_fpaths.items():
            rel_path = fpath.relative_to(rel_dpath)
            line = r'\includegraphics[width=1.0\textwidth]{' + str(rel_path) + '}%'
            parts.append(line)
            parts.append(r'\hfill')
            ordinal = chr(index + 97)
            parts.append('(' + ordinal + ') ' + key)
            index += 1
        footer = ub.codeblock(
            r'''
            \caption[]{
                Qualitative results using the top-performing model on the validation set, applied to a selection of images
                  from the (a) test, (b) validation, and (c) training sets.
            }
            \label{fig:test_results_all_models}
            \end{figure*}
            ''')

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
    figure_dpath = ub.Path('$HOME/code/shitspotter/papers/application-2024/figures').expand()


def main():
    aggregators_rows = load_aggregators()
    summary_df = process_aggregators(aggregators_rows)
    new_text = build_latex_result_table(summary_df)
    print(new_text)
    gather_heatmap_figures(aggregators_rows)

    node_types = [
        'detectron_pred',
        'heatmap_pred',
        'heatmap_eval',
        'extract_polygons',
        'heatmap_pred_fit',
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

    for key, group in list(deduped.groupby(['node_type', 'dname', 'dataset_name'])):
        print(key)
        print('  time', kwutil.timedelta.coerce(group['duration'].sum()).to('pint').to('days'))
        print('  kwh', group['kwh'].sum())
        print('  co2', group['co2_kg'].sum())


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/papers/application-2024/scripts/build_v2_result_table.py
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
