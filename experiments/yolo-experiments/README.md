Note: the YOLO experiments were not originally intended for inclusion in the
paper, but reviewers have requested them, and we do have trained models. As
such the documentation for training and evaluation might not contain enough
information to fully perform each step. However, we do have notes in the POC
(proof-of-concpet directory), and we will attempt to improve docs here.

~/code/shitspotter/dev/poc/train_yolo_shitspotter.sh


I know that I've published the YOLO models here:
 tree ~/code/shitspotter/shitspotter_dvc/models/yolo-v9
├── demo-predict-onnx.sh
├── demo-predict-torch.sh
├── shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt
├── shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.onnx
└── shitspotter-simple-v3-run-v06-train_config.yaml


So we will evaluate this as the primary model.
