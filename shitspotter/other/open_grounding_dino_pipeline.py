"""
TODO: it would be nice if the open grounding dino pipeline worked in the same
way as the huggingface official grounding dino stuff, but alas it doesn't, and
this pipeline runs a the separate variant on a repo that is not setup properly
as a python package, so we have to do some hacks.
"""
from geowatch.mlops.pipeline_nodes import ProcessNode, PipelineDAG
import ubelt as ub  # NOQA
from shitspotter.pipelines import DetectionEvaluation

PREDICT_NAME = 'pred'
EVALUATE_NAME = 'eval'


class OpenGroundingDinoPrediction(ProcessNode):
    """
    Runs GroundingDino prediction for a prompt.
    """
    name = 'open_grounding_dino_pred'
    group_dname = PREDICT_NAME
    executable = 'python ~/code/Open-GroundingDino/coco_predict_open_grounding_dino.py'

    in_paths = {
        'src',
    }

    out_paths = {
        'dst': 'pred.kwcoco.zip',
    }
    primary_out_key = 'dst'

    algo_params = {
        'classes': '[dogpoop]',
        'force_classname': 'poop',
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


def open_grounding_dino_evaluation_pipeline():
    """
    Example:
        >>> from shitspotter.other.grounding_dino_pipeline import *  # NOQA
        >>> dag = open_grounding_dino_evaluation_pipeline()
        >>> dag.configure(config={
        >>>     'open_grounding_dino_pred.src': 'foo.kwcoco.zip',
        >>>     'open_grounding_dino_pred.classes': '[feces]',
        >>> })
        >>> dag.print_graphs(shrink_labels=1, show_types=1)
        >>> queue = dag.make_queue()['queue']
        >>> queue.print_commands(with_locks=0)
    """
    nodes = {}

    gpred = nodes['open_grounding_dino_pred'] = OpenGroundingDinoPrediction()
    eval_ = nodes['detection_evaluation'] = DetectionEvaluation()

    gpred.outputs['dst'].connect(eval_.inputs['pred_dataset'])
    gpred.inputs['src'].connect(eval_.inputs['true_dataset'])

    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()
    return dag
