"""
Define predict / evaluation pipelines for geowatch's MLops

To define a pipeline, the user specifies a bash executable, which parameters
correspond to inputs, outputs, algorithm settting, performance settings (perf
settings should not impact the output).

The main assumptions are:

    * All inputs and outputs are files on disk.

    * All parameters can be specified as key/value pairs (
      that includes boolean flags!)

    * File path names encode content. That is, if the content of a file
      changes, the pipeline will not detect that. We may loosen this assumption
      in the future.

The "in_paths" and "out_paths" are the most important entries to specify a
pipeline that can run. Everything else is for bookkeeping.

If the assumption of arguments as key/value pairs is broken, nodes can specify
a "command" method, where the user can define exactly what shell command to
run.
"""
# import shlex
# import json
from geowatch.mlops.pipeline_nodes import ProcessNode
from geowatch.mlops.pipeline_nodes import PipelineDAG
import ubelt as ub  # NOQA

PREDICT_NAME = 'pred'
EVALUATE_NAME = 'eval'


class HeatmapPrediction(ProcessNode):
    """
    CommandLine:
        xdoctest -m shitspotter.pipelines HeatmapPrediction

    Example:
        >>> from shitspotter.pipelines import *  # NOQA
        >>> node = HeatmapPrediction()
        >>> node.configure({
        >>>     'package_fpath': 'model.pt',
        >>>     'test_dataset': 'test.kwcoco.zip',
        >>> })
        >>> command = node.command
        >>> print(node.command)
    """
    name = 'heatmap_pred'
    group_dname = PREDICT_NAME

    executable = 'python -m geowatch.tasks.fusion.predict'

    in_paths = {
        'package_fpath',
        'test_dataset',
    }

    out_paths = {
        'pred_dataset' : 'pred.kwcoco.zip',
    }
    primary_out_key = 'pred_dataset'

    perf_params = {
        'num_workers': 2,
        'devices': '0,',
        'batch_size': 1,
    }

    algo_params = {
    }

    def load_result(self, node_dpath):
        from geowatch.mlops import smart_result_parser
        from geowatch.mlops.aggregate_loader import new_process_context_parser
        from geowatch.utils import util_dotdict
        from kwutil import util_pattern
        node_type = self.name
        pat = util_pattern.Pattern.coerce(node_dpath / 'pred.kwcoco.*')
        found = list(pat.paths())
        if len(found) == 0:
            raise FileNotFoundError(f'Unable to find expected kwcoco file in {node_type} node_dpath: {node_dpath}')
        fpath = found[0]
        bas_pxl_info = smart_result_parser.parse_json_header(fpath)
        proc_item = smart_result_parser.find_pred_pxl_item(bas_pxl_info)
        nest_resolved = new_process_context_parser(proc_item)
        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(node_type, index=1)

        # Record the train-time parameters
        fit_node_type = node_type + '_fit'
        extra = proc_item['properties']['extra']
        fit_config = extra['fit_config']
        if 'data' not in fit_config:
            raise Exception(ub.paragraph(
                f'''
                A kwcoco has an old fit-config that did not contain all
                train-time params. To fix this run for a single file run:
                ``python -m geowatch.cli.experimental.fixup_predict_kwcoco_metadata {fpath}``
                ''') +
                '\n\n' +
                ub.paragraph(
                    '''
                    For more details see:
                    ``python -m geowatch.cli.experimental.fixup_predict_kwcoco_metadata --help``
                    '''))

        fit_nested = {
            'context': {'task': 'geowatch.tasks.fusion.fit'},
            'resolved_params': fit_config,
            'resources': {},
            'machine': {},
        }
        flat_fit_resolved = util_dotdict.DotDict.from_nested(fit_nested)
        flat_fit_resolved = flat_fit_resolved.insert_prefix(fit_node_type, index=1)
        flat_resolved |= flat_fit_resolved
        return flat_resolved


class PolygonExtraction(ProcessNode):
    """
    CommandLine:
        xdoctest -m shitspotter.pipelines PolygonExtraction

    Example:
        >>> from shitspotter.pipelines import *  # NOQA
        >>> node = PolygonExtraction()
        >>> node.configure({
        >>>     'pred_dataset': 'pred.kwcoco.zip',
        >>> })
        >>> print(node.command)
    """
    name = 'polygon_extraction'
    group_dname = PREDICT_NAME

    executable = 'python -m geowatch.cli.run_tracker'

    in_paths = {
        'pred_dataset',
    }

    algo_params = {
        'resolution': None,
        'thresh': 0.5,
        'agg_fn': 'probs',
        'moving_window_size': 1,
    }

    out_paths = {
        'site_summaries_fpath': 'site_summaries_manifest.json',
        'site_summaries_dpath': 'site_summaries',
        'sites_fpath': 'sites_manifest.json',
        'sites_dpath': 'sites',
        'poly_kwcoco_fpath': 'poly.kwcoco.zip'
    }
    primary_out_key = 'poly_kwcoco_fpath'

    default_track_fn = 'saliency_heatmaps'

    @property
    def command(self):
        import shlex
        import json
        fmtkw = self.final_config.copy()
        fmtkw['default_track_fn'] = self.default_track_fn
        external_args = {
            'site_summary',
            'boundary_region',
            'site_score_thresh',
            'smoothing', 'append_mode',
            'time_pad_before',
            'time_pad_after',
        }
        track_kwargs = self.final_algo_config.copy() - external_args
        track_kwargs = track_kwargs - {'pred_dataset'}  # not sure why this is needed
        fmtkw['kwargs_str'] = shlex.quote(json.dumps(track_kwargs))
        fmtkw['external_argstr'] = self._make_argstr(self.final_config & external_args)
        command = ub.codeblock(
            r'''
            python -m geowatch.cli.run_tracker \
                --input_kwcoco "{pred_dataset}" \
                --default_track_fn {default_track_fn} \
                --track_kwargs {kwargs_str} \
                --clear_annots=True \
                --out_site_summaries_fpath "{site_summaries_fpath}" \
                --out_site_summaries_dir "{site_summaries_dpath}" \
                --out_sites_fpath "{sites_fpath}" \
                --out_sites_dir "{sites_dpath}" \
                --out_kwcoco "{poly_kwcoco_fpath}" \
                {external_argstr}
            ''').format(**fmtkw)
        return command

    @property
    def final_algo_config(self):
        return ub.udict({
            # 'boundaries_as': 'polys'
            "agg_fn": "probs",
        }) | super().final_algo_config


class DetectionEvaluation(ProcessNode):
    """
    CommandLine:
        xdoctest -m shitspotter.pipelines DetectionEvaluation

    Example:
        >>> from shitspotter.pipelines import *  # NOQA
        >>> node = DetectionEvaluation()
        >>> node.configure({
        >>>     'true_dataset': 'test.kwcoco.zip',
        >>>     'pred_dataset': 'poly.kwcoco.zip',
        >>> })
        >>> print(node.command)
    """
    name = 'detection_evaluation'
    executable = 'python -m kwcoco eval'
    group_dname = EVALUATE_NAME

    in_paths = {
        'true_dataset',
        'pred_dataset',
    }

    out_paths = {
        'out_dpath': '.',
        'out_fpath': 'detect_metrics.json',
    }
    primary_out_key = 'out_fpath'

    algo_params = {
        'iou_thresh': 0.5,
    }

    perf_params = {
        'draw': False,
    }


class HeatmapEvaluation(ProcessNode):
    """
    CommandLine:
        xdoctest -m shitspotter.pipelines HeatmapEvaluation

    Example:
        >>> from shitspotter.pipelines import *  # NOQA
        >>> node = HeatmapEvaluation()
        >>> node.configure({
        >>>     'true_dataset': 'test.kwcoco.zip',
        >>>     'pred_dataset': 'pred.kwcoco.zip',
        >>> })
        >>> print(node.command)
    """
    name = 'heatmap_eval'
    executable = 'python -m geowatch.tasks.fusion.evaluate'
    group_dname = EVALUATE_NAME

    in_paths = {
        'true_dataset',
        'pred_dataset',
    }

    out_paths = {
        'eval_dpath': '.',
        'eval_fpath': 'pxl_eval.json',
    }
    primary_out_key = 'eval_fpath'

    algo_params = {
        'score_space': 'image',
    }

    perf_params = {
        'workers': 2,
        # These arent quite perf params.
        # They control intermediate visualization, but they don't impact
        # effective outputs so we are putting them here.
        'draw_curves': True,
        'draw_heatmaps': False,
        'viz_thresh': 'auto',
    }

    def load_result(self, node_dpath):
        from geowatch.mlops import smart_result_parser
        from geowatch.mlops.aggregate_loader import new_process_context_parser
        from geowatch.utils import util_dotdict
        fpath = node_dpath / self.out_paths[self.primary_out_key]
        info = smart_result_parser.load_pxl_eval(fpath, with_param_types=False)
        metrics = info['metrics']

        proc_item = smart_result_parser.find_pxl_eval_item(
            info['json_info']['meta']['info'])

        nest_resolved = new_process_context_parser(proc_item)
        # Hack for region ids
        nest_resolved['context']['region_ids'] = ub.Path(nest_resolved['resolved_params']['true_dataset']).name.split('.')[0]
        nest_resolved['metrics'] = metrics

        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(self.name, 1)
        return flat_resolved

    def _default_metrics(self):
        _display_metrics_suffixes = []
        _primary_metrics_suffixes = [
            'salient_AP',
            # 'salient_APUC',
            'salient_AUC',
        ]
        return _primary_metrics_suffixes, _display_metrics_suffixes

    @property
    def default_vantage_points(self):
        vantage_points = [
            {
                'metric1': 'metrics.heatmap_eval.salient_AP',
                'metric2': 'metrics.heatmap_eval.salient_AUC',

                'scale1': 'linear',
                'scale2': 'linear',

                'objective1': 'maximize',
            },
        ]
        return vantage_points


def heatmap_evaluation_pipeline():
    """
    TODO:
        there is likely a nice and intuitive way for users to specify pipelines.

    CommandLine:
        xdoctest -m shitspotter.pipelines heatmap_evaluation_pipeline

    Example:
        >>> from shitspotter.pipelines import *  # noqa
        >>> dag = heatmap_evaluation_pipeline()
        >>> dag.print_graphs(shrink_labels=1, show_types=1)
        >>> queue = dag.make_queue()['queue']
        >>> queue.print_commands(with_locks=0)
    """
    nodes = {}
    heatmap_pred = nodes['heatmap_pred'] = HeatmapPrediction()
    heatmap_eval = nodes['heatmap_eval'] = HeatmapEvaluation()

    # Heatmap evaluation needs the test dataset given to heatmap_pred prediction
    heatmap_pred.inputs['test_dataset'].connect(heatmap_eval.inputs['true_dataset'])

    # The output of heatmap_pred prediction is given to heatmap_pred evaluation
    heatmap_pred.outputs['pred_dataset'].connect(heatmap_eval.inputs['pred_dataset'])

    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()

    return dag


def polygon_evaluation_pipeline():
    """
    TODO:
        there is likely a nice and intuitive way for users to specify pipelines.

    CommandLine:
        xdoctest -m shitspotter.pipelines polygon_evaluation_pipeline

    Example:
        >>> from shitspotter.pipelines import *  # noqa
        >>> dag = polygon_evaluation_pipeline()
        >>> dag.print_graphs(shrink_labels=1, show_types=1)
        >>> queue = dag.make_queue()['queue']
        >>> queue.print_commands(with_locks=0)
    """
    nodes = {}
    heatmap_pred = nodes['heatmap_pred'] = HeatmapPrediction()
    heatmap_eval = nodes['heatmap_eval'] = HeatmapEvaluation()
    poly_pred = nodes['polygon_pred'] = PolygonExtraction()

    # Heatmap evaluation needs the test dataset given to heatmap_pred prediction
    heatmap_pred.inputs['test_dataset'].connect(heatmap_eval.inputs['true_dataset'])

    # The output of heatmap_pred prediction is given to heatmap_pred evaluation
    heatmap_pred.outputs['pred_dataset'].connect(heatmap_eval.inputs['pred_dataset'])

    # Connect heatmaps to polygon extraction
    heatmap_pred.outputs['pred_dataset'].connect(poly_pred.inputs['pred_dataset'])

    # Connect polygon extraction to polygon evaluation
    poly_eval = nodes['polygon_eval'] = DetectionEvaluation()
    heatmap_pred.outputs['pred_dataset'].connect(poly_eval.inputs['pred_dataset'])
    heatmap_pred.inputs['test_dataset'].connect(poly_eval.inputs['true_dataset'])

    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()

    return dag
