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

    perf_params = {
        'num_workers': 2,
        'devices': '0,',
        #'accelerator': 'gpu',
        'batch_size': 1,
    }

    algo_params = {
        # 'drop_unused_frames': True,
        # 'with_saliency': True,
        # 'with_saliency': 'auto',
        # 'with_class': 'auto',
        # 'with_change': 'auto',
    }

    # def command(self):
    #     fmtkw = self.final_config.copy()
    #     perf_config = self.final_perf_config
    #     algo_config = self.final_algo_config - {
    #         'package_fpath', 'test_dataset', 'pred_dataset'}
    #     fmtkw['params_argstr'] = self._make_argstr(algo_config)
    #     fmtkw['perf_argstr'] = self._make_argstr(perf_config)
    #     command = ub.codeblock(
    #         r'''
    #         python -m geowatch.tasks.fusion.predict \
    #             --package_fpath={package_fpath} \
    #             --test_dataset={test_dataset} \
    #             --pred_dataset={pred_pxl_fpath} \
    #             {params_argstr} \
    #             {perf_argstr}
    #         ''').format(**fmtkw).rstrip().rstrip('\\').rstrip()
    #     return command


class HeatmapEvaluation(ProcessNode):
    name = 'heatmap_eval'
    group_dname = EVALUATE_NAME

    executable = 'python -m geowatch.tasks.fusion.evaluate'

    in_paths = {
        'true_dataset',
        'pred_dataset',
    }

    out_paths = {
        'eval_dpath': '.',
        'eval_fpath': 'pxl_eval.json',
    }

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

    # def command(self):
    #     # TODO: better score space
    #     fmtkw = self.final_config.copy()
    #     extra_opts = {
    #         'draw_curves': True,
    #         'draw_heatmaps': False,
    #         'viz_thresh': 'auto',
    #         'workers': 2,
    #         'score_space': 'video',
    #     }
    #     fmtkw['extra_argstr'] = self._make_argstr(extra_opts)  # NOQA
    #     command = ub.codeblock(
    #         r'''
    #         python -m geowatch.tasks.fusion.evaluate \
    #             --true_dataset={true_dataset} \
    #             --pred_dataset={pred_pxl_fpath} \
    #             --eval_dpath={eval_pxl_dpath} \
    #             --eval_fpath={eval_pxl_fpath} \
    #             {extra_argstr}
    #         ''').format(**fmtkw)
    #     # .format(**eval_act_pxl_kw).strip().rstrip('\\')
    #     return command


def heatmap_evaluation_pipeline():
    """
    TODO:
        there is likely a nice and intuitive way for users to specify pipelines.
    """
    nodes = {}
    heatmap_pred = nodes['heatmap_pred'] = HeatmapPrediction()
    heatmap_eval = nodes['heatmap_eval'] = HeatmapEvaluation()

    # Heatmap evaluation needs the test dataset given to heatmap_pred prediction
    heatmap_pred.inputs['test_dataset'].connect(heatmap_eval.inputs['true_dataset'])

    # The output of heatmap_pred prediction is given to heatmap_pred evaluation
    heatmap_pred.outputs['pred_dataset'].connect(heatmap_eval.inputs['pred_dataset'])
    # heatmap_pred.connect(heatmap_eval)
    # import xdev
    # xdev.embed()

    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()

    # dag.print_graphs(shrink_labels=1, show_types=1)
    return dag
