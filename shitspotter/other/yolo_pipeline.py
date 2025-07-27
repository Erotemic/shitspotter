"""
SeeAlso:
    ~/code/shitspotter/experiments/yolo-experiments/run_yolo_experiments_v1.sh
"""
from geowatch.mlops.pipeline_nodes import ProcessNode, PipelineDAG
import ubelt as ub  # NOQA
from shitspotter.pipelines import DetectionEvaluation

PREDICT_NAME = 'pred'
EVALUATE_NAME = 'eval'


class YoloDetectionPrediction(ProcessNode):
    """
    Runs GroundingDino prediction for a prompt.
    """
    name = 'yolo_pred'
    group_dname = PREDICT_NAME
    executable = 'python -m shitspotter.other.predict_yolo'

    in_paths = {
        'src',
    }

    out_paths = {
        'dst': 'pred.kwcoco.zip',
    }
    primary_out_key = 'dst'

    algo_params = {
        'model_config': None,
        'checkpoint': None,
    }

    def load_result(self, node_dpath):
        from geowatch.mlops import smart_result_parser
        from geowatch.mlops.aggregate_loader import new_process_context_parser
        from geowatch.utils import util_dotdict

        fpath = node_dpath / self.out_paths[self.primary_out_key]
        coco_pred_info = smart_result_parser.parse_json_header(fpath)
        assert len(coco_pred_info) == 1
        proc_item = coco_pred_info[0]
        nest_resolved = new_process_context_parser(proc_item)
        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(self.name, index=1)
        return flat_resolved


def yolo_evaluation_pipeline():
    """
    Example:
        >>> from shitspotter.other.grounding_dino_pipeline import *  # NOQA
        >>> dag = grounding_dino_evaluation_pipeline()
        >>> dag.configure(config={
        >>>     'yolo_pred.src': 'foo.kwcoco.zip',
        >>>     'yolo_pred.classes': '[feces]',
        >>> })
        >>> dag.print_graphs(shrink_labels=1, show_types=1)
        >>> queue = dag.make_queue()['queue']
        >>> queue.print_commands(with_locks=0)
    """
    nodes = {}

    gpred = nodes['yolo_pred'] = YoloDetectionPrediction()
    eval_ = nodes['detection_evaluation'] = DetectionEvaluation()

    gpred.outputs['dst'].connect(eval_.inputs['pred_dataset'])
    gpred.inputs['src'].connect(eval_.inputs['true_dataset'])

    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()
    return dag
