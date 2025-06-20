"""
Requirements:
    pip install onnx onnxruntime
    pip install onnx-simplifier

Notes:
    # Move the YOLO models to the model distribution directory

    /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v06/train/shitspotter-simple-v3-run-v06/lightning_logs/version_1/checkpoints/epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt

    mkdir /home/joncrall/code/shitspotter/shitspotter_dvc/models/yolo-v9

    cp /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v06/train/shitspotter-simple-v3-run-v06/lightning_logs/version_1/checkpoints/epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt \
        /home/joncrall/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt

    cp /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v06/train/shitspotter-simple-v3-run-v06/train_config.yaml \
        /home/joncrall/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-train_config.yaml

    cp /home/joncrall/code/YOLO-v9/yolov9-simplified.onnx \
        /home/joncrall/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.onnx

"""
import onnxruntime as ort
import numpy as np
import torch
import kwutil
import ubelt as ub
from yolo.utils.kwcoco_utils import tensor_to_kwimage
from yolo.utils.bounding_box_utils import create_converter
from yolo.utils.model_utils import PostProcess
from omegaconf.dictconfig import DictConfig
from yolo.tools.solver import InferenceModel

# See ~/code/shitspotter/dev/poc/train_yolo_shitspotter.sh
checkpoint_path = ub.Path('/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v06/train/shitspotter-simple-v3-run-v06/lightning_logs/version_1/checkpoints/epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt')
train_config = ub.Path('/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v06/train/shitspotter-simple-v3-run-v06/train_config.yaml')

config = kwutil.Yaml.coerce(train_config, backend='pyyaml')
cfg = DictConfig(config)
cfg.weight = checkpoint_path
model = InferenceModel(cfg)
model.eval()
model.post_process = PostProcess(model.vec2box, model.validation_cfg.nms)
vec2box = create_converter(
    model.cfg.model.name, model.model, model.cfg.model.anchor, model.cfg.image_size, model.device
)

input_tensor = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Test a regular forward pass.
model.cfg.task.nms = DictConfig(kwutil.Yaml.coerce(
    '''
    min_confidence: 0.01
    min_iou: 0.5
    max_bbox: 300
    ''', backend='pyyaml'))
post_process = PostProcess(vec2box, model.cfg.task.nms)
torch_outputs = model.forward(torch.Tensor(input_tensor))

predicts = post_process(torch_outputs)
classes = cfg.dataset.class_list
detections = [
    tensor_to_kwimage(yolo_annot_tensor, classes=classes).numpy()
    for yolo_annot_tensor in predicts]


# Convert to onnx
device = torch.device('cpu')
dummy_input = torch.randn(1, 3, 640, 640).to(device)  # Adjust image size as needed
torch.onnx.export(
    model,                             # The loaded YOLO model
    dummy_input,                       # Example input tensor
    "yolov9.onnx",                     # Output ONNX file
    export_params=True,                # Store trained weights
    opset_version=12,                  # ONNX opset version
    do_constant_folding=True,          # Optimize the graph
    input_names=['input'],             # Input name
    output_names=['output'],           # Output name
    dynamic_axes={
        'input': {0: 'batch_size'},    # Enable dynamic batch size
        'output': {0: 'batch_size'}
    }
)


# ub.cmd('python -m onnxsim yolov9.onnx yolov9-simplified.onnx')
# ort_session = ort.InferenceSession("yolov9-simplified.onnx")

ort_session = ort.InferenceSession("yolov9.onnx")

# Simulate an image tensor
onnx_outputs = ort_session.run(None, {'input': input_tensor})

torch_walker = ub.IndexableWalker(torch_outputs)
onnx_walker = ub.IndexableWalker(onnx_outputs)


def walker_to_nx(walker):
    import networkx as nx
    graph = nx.DiGraph()

    # root
    node = tuple()
    v = walker.data
    graph.add_node(node, label=f'.: {type(v).__name__}')

    for p, v in walker:
        node = tuple(p)
        parent = node[0:-1]
        graph.add_node(node)
        graph.add_edge(parent, node)
        if not isinstance(v, (list, dict)):
            if hasattr(v, 'shape'):
                graph.nodes[node]['label'] = f'{node}: {type(v)}[{v.shape}]'
            else:
                graph.nodes[node]['label'] = f'{node}: {type(v)}'
        else:
            graph.nodes[node]['label'] = f'{node}: {type(v).__name__}'
    nx.write_network_text(graph)

print('Torch Output:')
walker_to_nx(torch_walker)
print('ONNX Output:')
walker_to_nx(onnx_walker)


onnx_outputs[0].shape

torch_outputs['Main'][0][0].shape

# Split the ONNX output back into its tuple-like structure
recon_outputs = {}
onnx_outputs_ = [torch.Tensor(a) for a in onnx_outputs]
recon_outputs['Main'] = list(ub.chunks(onnx_outputs_[0:9], chunksize=3))
recon_outputs['Aux'] = list(ub.chunks(onnx_outputs_[9:], chunksize=3))

recon_walker = ub.IndexableWalker(recon_outputs)
print('ONNX Recon Output:')
walker_to_nx(recon_walker)

predicts = post_process(recon_outputs)
classes = cfg.dataset.class_list
detections = [
    tensor_to_kwimage(yolo_annot_tensor, classes=classes).numpy()
    for yolo_annot_tensor in predicts]

# Now see: ~/code/shitspotter/tpl/scatspotter_app/explore.rst
