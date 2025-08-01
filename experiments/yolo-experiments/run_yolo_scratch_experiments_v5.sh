

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_yolo_scratch_evals/test_imgs121_6cb3b6ff
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()'
        matrix:
            yolo_pred.src:
                - '$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip'
            yolo_pred.model_config:
                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0097-step=000294-trainlosstrain_loss=16.305.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0163-step=000492-trainlosstrain_loss=13.247.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0216-step=000651-trainlosstrain_loss=11.828.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0453-step=001362-trainlosstrain_loss=7.986.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0497-step=001494-trainlosstrain_loss=8.029.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0288-step=000867-trainlosstrain_loss=9.719.ckpt.ckpt
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --print_varied=0 \
    --run=1

# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_yolo_scratch_evals/test_imgs121_6cb3b6ff
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=0 \
    --rois=None \
    --io_workers=0 \
    --eval_nodes="
        - detection_evaluation
    " \
    --stdout_report="
        top_k: 10
        per_group: null
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 1
        show_csv: 0
    "


export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_yolo_scratch_evals/vali_imgs691_99b22ad0
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()'
        matrix:
            yolo_pred.src:
                - '$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.kwcoco.zip'
            yolo_pred.model_config:

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0097-step=000294-trainlosstrain_loss=16.305.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0163-step=000492-trainlosstrain_loss=13.247.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0216-step=000651-trainlosstrain_loss=11.828.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0453-step=001362-trainlosstrain_loss=7.986.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0497-step=001494-trainlosstrain_loss=8.029.ckpt.ckpt

                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-yolo-train_imgs5747_1e73d54f-v5/train/shitspotter-yolo-train_imgs5747_1e73d54f-v5/lightning_logs/version_0/checkpoints-copy/epoch=0288-step=000867-trainlosstrain_loss=9.719.ckpt.ckpt

    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux --skip_existing=0 \
    --print_varied=0 \
    --run=1


# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_yolo_scratch_evals/vali_imgs691_99b22ad0
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=0 \
    --rois=None \
    --io_workers=0 \
    --eval_nodes="
        - detection_evaluation
    " \
    --stdout_report="
        top_k: 10
        per_group: null
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 0
        show_csv: 0
    "
