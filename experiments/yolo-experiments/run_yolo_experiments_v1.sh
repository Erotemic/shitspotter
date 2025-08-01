~/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt
~/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-train_config.yaml

# This run definately worked at some point:
# /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v06/train/shitspotter-simple-v3-run-v06/train_config.yaml



ls /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v0*/train/*/
fd train_config /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v0*/train/*/
fd checkpoints /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v0*/train/*/

ls /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v0*/train/*/train_config.yaml -l
ls /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-*/train/mit-yolo-*/lightning_logs/*/checkpoints/*.ckpt

/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/train_config.yaml
/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/train_config.yaml



python -m shitspotter.other.predict_yolo \
    --src /home/joncrall/code/shitspotter/shitspotter_dvc/test_imgs30_d8988f8c.kwcoco.zip \
    --dst ~/data/dvc-repos/shitspotter_expt_dvc/_yolov9_evals/test/pred3.kwcoco.zip \
    --checkpoint /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v06/train/shitspotter-simple-v3-run-v06/lightning_logs/version_2/checkpoints/epoch=0497-step=001992-trainlosstrain_loss=4.483.ckpt.ckpt \
    --model_config /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter-simple-v3-run-v06/train/shitspotter-simple-v3-run-v06/train_config.yaml

python -m shitspotter.other.predict_yolo \
    --src /home/joncrall/code/shitspotter/shitspotter_dvc/test_imgs30_d8988f8c.kwcoco.zip \
    --dst ~/data/dvc-repos/shitspotter_expt_dvc/_yolov9_evals/test/pred2.kwcoco.zip \
    --checkpoint ~/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt \
    --model_config ~/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-train_config.yaml


python -m shitspotter.other.predict_yolo \
    --src /home/joncrall/code/shitspotter/shitspotter_dvc/test_imgs30_d8988f8c.kwcoco.zip \
    --dst ~/data/dvc-repos/shitspotter_expt_dvc/_yolov9_evals/test/pred2.kwcoco.zip \
    --checkpoint /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/checkpoints/epoch=0198-step=025273-trainlosstrain_loss=0.019.ckpt.ckpt \
    --model_config /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml


export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_yolo_evals/test_imgs121_6cb3b6ff
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()'
        matrix:
            yolo_pred.src:
                - '$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip'
            yolo_pred.model_config:
                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/checkpoints/epoch=0198-step=025273-trainlosstrain_loss=0.019.ckpt.ckpt
                - config: $HOME/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-train_config.yaml
                  checkpoint: $HOME/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt
                - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/train_config.yaml
                  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/lightning_logs/version_0/checkpoints/epoch=0049-step=000200-trainlosstrain_loss=7.173.ckpt.ckpt
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0," --tmux_workers=1 \
    --backend=tmux --skip_existing=1 \
    --print_varied=0 \
    --run=0

# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_yolo_evals/test_imgs121_6cb3b6ff
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




#
#
# -- VALIDATION DATASET
#
python -c "if 1:
    import pathlib
    import yaml
    from collections import defaultdict

    base = pathlib.Path('/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs')

    import glob

    config_glob = list(glob.glob('/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v0*/train/*/train_config.yaml'))
    ckpt_glob   = list(glob.glob('/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-*/train/mit-yolo-*/lightning_logs/*/checkpoints/*.ckpt'))

    # Index configs by (run_name, subdir)
    config_lookup = {}
    import ubelt as ub
    for cfg in config_glob:
        cfg = ub.Path(cfg)
        try:
            run_name = cfg.parts[-4]  # mit-yolo-v0*
            subdir   = cfg.parts[-2]  # typically same or similar to run_name
            config_lookup[(run_name, subdir)] = cfg
        except Exception:
            pass

    items = []
    for ckpt in ckpt_glob:
        ckpt = ub.Path(ckpt)
        try:
            run_name = ckpt.parts[-7]  # mit-yolo-*
            subdir   = ckpt.parts[-5]  # second part of path under 'train'
            key = (run_name, subdir)
            if 'last.ckpt' not in str(ckpt):
                config = config_lookup.get(key)
                if config:
                    items.append({'config': str(config), 'checkpoint': str(ckpt)})
        except Exception:
            print('error')
            pass
    print(yaml.dump(items, sort_keys=False))
"
#
echo "
    - config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
      checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/checkpoints/epoch=0198-step=025273-trainlosstrain_loss=0.019.ckpt.ckpt

    - config: $HOME/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-train_config.yaml
      checkpoint: $HOME/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt

    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/lightning_logs/version_14/checkpoints/epoch=0097-step=001568-trainlosstrain_loss=6.754.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/lightning_logs/version_0/checkpoints/epoch=0049-step=000200-trainlosstrain_loss=7.173.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/lightning_logs/version_0/checkpoints/epoch=0056-step=000228-trainlosstrain_loss=6.887.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/lightning_logs/version_0/checkpoints/epoch=0123-step=000496-trainlosstrain_loss=7.197.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/lightning_logs/version_0/checkpoints/epoch=0122-step=000492-trainlosstrain_loss=6.865.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v04/train/mit-yolo-v02/lightning_logs/version_0/checkpoints/epoch=0118-step=000476-trainlosstrain_loss=7.321.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/lightning_logs/version_14/checkpoints/epoch=0079-step=001280-trainlosstrain_loss=6.654.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/lightning_logs/version_14/checkpoints/epoch=0087-step=001408-trainlosstrain_loss=6.313.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/lightning_logs/version_14/checkpoints/epoch=0067-step=001088-trainlosstrain_loss=6.255.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/lightning_logs/version_14/checkpoints/epoch=0097-step=001568-trainlosstrain_loss=6.754.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/lightning_logs/version_14/checkpoints/epoch=0050-step=000816-trainlosstrain_loss=6.406.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/lightning_logs/version_11/checkpoints/epoch=0001-step=000032-trainlosstrain_loss=6.186.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v03/train/mit-yolo-v02/lightning_logs/version_11/checkpoints/epoch=0000-step=000016-trainlosstrain_loss=5.943.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_20/checkpoints/epoch=0257-step=065274-trainlosstrain_loss=0.000.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_20/checkpoints/epoch=0307-step=077924-trainlosstrain_loss=0.000.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_20/checkpoints/epoch=0070-step=017963-trainlosstrain_loss=0.000.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_20/checkpoints/epoch=0233-step=059202-trainlosstrain_loss=0.000.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_20/checkpoints/epoch=0092-step=023529-trainlosstrain_loss=0.000.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_13/checkpoints/epoch=0035-step=018216-trainlosstrain_loss=0.106.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_13/checkpoints/epoch=0029-step=015180-trainlosstrain_loss=0.113.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_13/checkpoints/epoch=0005-step=003036-trainlosstrain_loss=0.114.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_13/checkpoints/epoch=0031-step=016192-trainlosstrain_loss=0.107.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v01/lightning_logs/version_13/checkpoints/epoch=0032-step=016698-trainlosstrain_loss=0.086.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_4/checkpoints/epoch=0000-step=000253-trainlosstrain_loss=5.875.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_0/checkpoints/epoch=0000-step=000253-trainlosstrain_loss=5.890.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_0/checkpoints/epoch=0001-step=000506-trainlosstrain_loss=5.674.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/checkpoints/epoch=0306-step=038989-trainlosstrain_loss=0.023.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/checkpoints/epoch=0160-step=020447-trainlosstrain_loss=0.028.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/checkpoints/epoch=0218-step=027813-trainlosstrain_loss=0.021.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/checkpoints/epoch=0330-step=042037-trainlosstrain_loss=0.023.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_5/checkpoints/epoch=0198-step=025273-trainlosstrain_loss=0.019.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_2/checkpoints/epoch=0392-step=099429-trainlosstrain_loss=0.006.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_2/checkpoints/epoch=0208-step=052877-trainlosstrain_loss=0.010.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_2/checkpoints/epoch=0231-step=058696-trainlosstrain_loss=0.016.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_2/checkpoints/epoch=0190-step=048323-trainlosstrain_loss=0.018.ckpt.ckpt
    #- config: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/train_config.yaml
    #  checkpoint: /data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/mit-yolo-v01/train/mit-yolo-v02/lightning_logs/version_2/checkpoints/epoch=0138-step=035167-trainlosstrain_loss=0.017.ckpt.ckpt
" > chosen_config.yaml

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.other.yolo_pipeline.yolo_evaluation_pipeline()'
        matrix:
            yolo_pred.src:
                - '$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.kwcoco.zip'
            yolo_pred.model_config: chosen_config.yaml
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1," --tmux_workers=2 \
    --backend=tmux --skip_existing=0 \
    --print_varied=0 \
    --run=0


# NOTE: need to ignore
# /home/joncrall/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-train_config.yaml
# because its trained on a bigger dataset

# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0
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



# Best Result
# /data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0/eval/flat/detection_evaluation/detection_evaluation_id_7112cd0c

# TODO: eval_dets needs to have an option for outputing a confusion
# visualization for detection.
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0/eval/flat/detection_evaluation/detection_evaluation_id_7112cd0c/.pred/yolo_pred/yolo_pred_id_1525e7de/pred.kwcoco.zip


#!/bin/bash
# See Also:
# /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0/pred/flat/yolo_pred/yolo_pred_id_1525e7de
python -m kwcoco eval \
    --true_dataset=/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.kwcoco.zip \
    --pred_dataset=/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0/pred/flat/yolo_pred/yolo_pred_id_1525e7de/pred.kwcoco.zip \
    --out_dpath=/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0/eval/flat/detection_evaluation/detection_evaluation_id_7112cd0c \
    --out_fpath=/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0/eval/flat/detection_evaluation/detection_evaluation_id_7112cd0c/detect_metrics.json \
    --confusion_fpath=/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0/eval/flat/detection_evaluation/detection_evaluation_id_7112cd0c/confusion.kwcoco.zip \
    --draw=True \
    --iou_thresh=0.5


geowatch visualize /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_2025_rebutal_yolo_evals/vali_imgs691_99b22ad0/eval/flat/detection_evaluation/detection_evaluation_id_7112cd0c/confusion.kwcoco.zip --smart --draw_imgs=False --draw_anns=True --max_dim=512 --draw_chancode=False --draw_header=False
#--channels="red|green|blue,red|green|blue" --role_order="true,pred"


# Next steps:
# stack visualizations to add results from dino and yolo to paper
# evaluate tuned grounding dino
