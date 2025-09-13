# YOLO Experiments

For the 2025 paper, there were two YOLO experiments run, pretrained and from scratch.

The scripts for training are:
* [train_yolo_shitspotter_poc_pretrained_v1.sh](./train_yolo_shitspotter_poc_pretrained_v1.sh)
* [train_yolo_shitspotter_scratch_v5.sh](./train_yolo_shitspotter_scratch_v5.sh)

And for evaluation the scripts are:

* [run_yolo_experiments_v1.sh](./run_yolo_experiments_v1.sh)
* [run_yolo_scratch_experiments_v5.sh](./run_yolo_scratch_experiments_v5.sh)


Reproducing these experiments requires a bit of setup that should be inferable from the information in this repo, but raise an issue if you want to reproduce these and need help.

---

Old notes:

Note: the YOLO experiments were not originally intended for inclusion in the
paper, but reviewers have requested them, and we do have trained models. As
such the documentation for training and evaluation might not contain enough
information to fully perform each step. However, we do have notes in the POC
(proof-of-concept directory), and we will attempt to improve docs here.

~/code/shitspotter/dev/poc/train_yolo_shitspotter.sh


```
I know that I've published the YOLO models here:
 tree ~/code/shitspotter/shitspotter_dvc/models/yolo-v9
├── demo-predict-onnx.sh
├── demo-predict-torch.sh
├── shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt
├── shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.onnx
└── shitspotter-simple-v3-run-v06-train_config.yaml
```

So we will evaluate this as the primary model.
