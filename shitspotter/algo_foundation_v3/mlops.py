"""
Geowatch MLOps nodes for the foundation v3 pipeline.
"""

try:
    from geowatch.mlops.pipeline_nodes import PipelineDAG
    from geowatch.mlops.pipeline_nodes import ProcessNode
    _GEOWATCH_IMPORT_ERROR = None
except Exception as ex:  # pragma: no cover
    PipelineDAG = None
    ProcessNode = object
    _GEOWATCH_IMPORT_ERROR = ex


PREDICT_NAME = 'pred'


def _require_geowatch():
    if _GEOWATCH_IMPORT_ERROR is not None:
        raise ImportError('geowatch is required for mlops pipeline usage') from _GEOWATCH_IMPORT_ERROR


class FoundationPrediction(ProcessNode):
    name = 'foundation_v3_pred'
    group_dname = PREDICT_NAME
    executable = 'python -m shitspotter.algo_foundation_v3.cli_predict'

    in_paths = {
        'src',
        'package_fpath',
    }

    out_paths = {
        'dst': 'pred.kwcoco.zip',
    }
    primary_out_key = 'dst'

    algo_params = {
        'backend': None,
        'create_labelme': False,
        'score_thresh': None,
        'nms_thresh': None,
        'crop_padding': None,
        'polygon_simplify': None,
    }

    perf_params = {}

    def load_result(self, node_dpath):
        from geowatch.mlops import smart_result_parser
        from geowatch.mlops.aggregate_loader import new_process_context_parser
        from geowatch.utils import util_dotdict

        fpath = node_dpath / self.out_paths[self.primary_out_key]
        coco_pred_info = smart_result_parser.parse_json_header(fpath)
        proc_item = coco_pred_info[-1]
        nest_resolved = new_process_context_parser(proc_item)
        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(self.name, index=1)
        return flat_resolved


def build_foundation_prediction_dag(detection_evaluation_cls):
    _require_geowatch()
    nodes = {}
    pred = nodes['foundation_v3_pred'] = FoundationPrediction()
    eval_ = nodes['detection_evaluation'] = detection_evaluation_cls()
    pred.outputs['dst'].connect(eval_.inputs['pred_dataset'])
    pred.inputs['src'].connect(eval_.inputs['true_dataset'])
    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()
    return dag
