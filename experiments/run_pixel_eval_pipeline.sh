#!/bin/bash
__doc__="
Setup:

Ensure you have an sdvc registery for the test dataset and packaged your checkpoints.
"

setup(){
    sdvc registery add --path "$HOME"/data/dvc-repos/shitspotter_dvc --name shitspotter_data --tags shitspotter_data
    sdvc registery add --path "$HOME"/data/dvc-repos/shitspotter_expt_dvc --name shitspotter_expt --tags shitspotter_expt
    sdvc registery list
    DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
    DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")

    python -m geowatch.mlops.manager "status" --expt_dvc_dpath "$DVC_EXPT_DPATH"
    python -m geowatch.mlops.manager "list" --expt_dvc_dpath "$DVC_EXPT_DPATH"
    python -m geowatch.mlops.manager "repackage checkpoints" --expt_dvc_dpath "$DVC_EXPT_DPATH"
}
export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")

test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"

#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs228_20928c8c.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_evals



# TODO: script to subselect models based on train-time validation metrics.


## Use geowatch mlops to define a grid of bash jobs to run evaluation over
## specified models
#python -m geowatch.mlops.schedule_evaluation \
#    --params="
#        pipeline: 'shitspotter.pipelines.heatmap_evaluation_pipeline()'
#        matrix:
#            heatmap_pred.package_fpath:
#                - $DVC_DATA_DPATH/models/shitspotter_scratch_v025-version_2-epoch=1277-step=005112-val_loss=0.600.ckpt.pt
#            heatmap_pred.test_dataset:
#                - $VALI_FPATH
#            heatmap_eval.workers: 4
#            heatmap_eval.draw_heatmaps: True
#            heatmap_eval.draw_curves: True
#    " \
#    --root_dpath="$EVAL_PATH" \
#    --devices="0,1" --tmux_workers=2 \
#    --backend=serial --skip_existing=0 \
#    --run=0


echo "
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_0/checkpoints/epoch=0000-step=000043-val_loss=0.008.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_0/checkpoints/epoch=0002-step=000129-val_loss=0.008.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_2/checkpoints/epoch=0004-step=000215-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_2/checkpoints/epoch=0005-step=000258-val_loss=0.016.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_2/checkpoints/epoch=0012-step=000559-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_3/checkpoints/epoch=0017-step=012294-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_3/checkpoints/epoch=0024-step=017075-val_loss=0.016.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_3/checkpoints/epoch=0026-step=018441-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_3/checkpoints/epoch=0034-step=023905-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_fromv28_newdata_20240615_v1/lightning_logs/version_3/checkpoints/epoch=0037-step=025954-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v2/lightning_logs/version_0/checkpoints/epoch=0041-step=057372-val_loss=0.022.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v2/lightning_logs/version_0/checkpoints/epoch=0062-step=086058-val_loss=0.022.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v2/lightning_logs/version_0/checkpoints/epoch=0063-step=087424-val_loss=0.021.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v2/lightning_logs/version_0/checkpoints/epoch=0065-step=090156-val_loss=0.022.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v2/lightning_logs/version_0/checkpoints/epoch=0072-step=099718-val_loss=0.022.ckpt.pt
- $DVC_DATA_DPATH/models/shitspotter_from_v027_halfres_v028-epoch=0121-step=000488-val_loss=0.005.ckpt.pt
- $DVC_DATA_DPATH/models/shitspotter_from_v027_halfres_v028-epoch=0179-step=000720-val_loss=0.005.ckpt.pt
- $DVC_DATA_DPATH/models/shitspotter_scratch_v025-version_2-epoch=1277-step=005112-val_loss=0.600.ckpt.pt
" > "$HOME"/code/shitspotter/experiments/models.yaml

echo "
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_3/checkpoints/epoch=0019-step=027320-val_loss=0.031.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_3/checkpoints/epoch=0020-step=028686-val_loss=0.031.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_3/checkpoints/epoch=0021-step=030052-val_loss=0.033.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_3/checkpoints/epoch=0025-step=035516-val_loss=0.033.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_3/checkpoints/epoch=0026-step=036882-val_loss=0.030.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_4/checkpoints/epoch=0080-step=110646-val_loss=0.022.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_4/checkpoints/epoch=0089-step=122940-val_loss=0.021.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_4/checkpoints/epoch=0092-step=127038-val_loss=0.021.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_4/checkpoints/epoch=0103-step=142064-val_loss=0.021.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_4/checkpoints/epoch=0105-step=144796-val_loss=0.022.ckpt.pt

- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_0/checkpoints/epoch=0023-step=032784-val_loss=0.026.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_0/checkpoints/epoch=0024-step=034150-val_loss=0.027.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_0/checkpoints/epoch=0026-step=036882-val_loss=0.025.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_0/checkpoints/epoch=0027-step=038248-val_loss=0.025.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_0/checkpoints/epoch=0028-step=039614-val_loss=0.026.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_1/checkpoints/epoch=0049-step=068300-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_1/checkpoints/epoch=0068-step=094254-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_1/checkpoints/epoch=0072-step=099718-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_1/checkpoints/epoch=0073-step=101084-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_1/checkpoints/epoch=0076-step=105182-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v5/lightning_logs/version_0/checkpoints/epoch=0056-step=077862-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v5/lightning_logs/version_0/checkpoints/epoch=0068-step=094254-val_loss=0.019.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v5/lightning_logs/version_0/checkpoints/epoch=0070-step=096986-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v5/lightning_logs/version_0/checkpoints/epoch=0072-step=099718-val_loss=0.016.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v5/lightning_logs/version_0/checkpoints/epoch=0078-step=107914-val_loss=0.019.ckpt.pt

- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v6/lightning_logs/version_0/checkpoints/epoch=0061-step=084692-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v6/lightning_logs/version_0/checkpoints/epoch=0067-step=092888-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v6/lightning_logs/version_0/checkpoints/epoch=0073-step=101084-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v6/lightning_logs/version_0/checkpoints/epoch=0074-step=102450-val_loss=0.016.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v6/lightning_logs/version_0/checkpoints/epoch=0076-step=105182-val_loss=0.018.ckpt.pt

- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_0/checkpoints/epoch=0007-step=010928-val_loss=0.041.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_0/checkpoints/epoch=0008-step=012294-val_loss=0.045.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_0/checkpoints/epoch=0010-step=015026-val_loss=0.040.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_0/checkpoints/epoch=0012-step=017758-val_loss=0.037.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_0/checkpoints/epoch=0013-step=019124-val_loss=0.038.ckpt.pt

- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v2/lightning_logs/version_0/checkpoints/last.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v3/lightning_logs/version_4/checkpoints/last.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_0/checkpoints/last.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v5/lightning_logs/version_0/checkpoints/last.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v6/lightning_logs/version_0/checkpoints/last.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_0/checkpoints/last.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/last.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_0/checkpoints/epoch=0063-step=087424-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_0/checkpoints/epoch=0067-step=092888-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_0/checkpoints/epoch=0072-step=099718-val_loss=0.018.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_0/checkpoints/epoch=0073-step=101084-val_loss=0.017.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/epoch=0076-step=105182-val_loss=0.020.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/epoch=0078-step=107914-val_loss=0.016.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/epoch=0081-step=112012-val_loss=0.020.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/epoch=0084-step=116110-val_loss=0.020.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v7/lightning_logs/version_1/checkpoints/epoch=0089-step=122940-val_loss=0.019.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_0/checkpoints/epoch=0022-step=031418-val_loss=0.027.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_0/checkpoints/epoch=0030-step=042346-val_loss=0.026.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_0/checkpoints/epoch=0032-step=045078-val_loss=0.026.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_0/checkpoints/epoch=0033-step=046444-val_loss=0.026.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_0/checkpoints/epoch=0034-step=047810-val_loss=0.027.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_1/checkpoints/epoch=0036-step=050542-val_loss=0.024.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_1/checkpoints/epoch=0049-step=068300-val_loss=0.023.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_1/checkpoints/epoch=0057-step=079228-val_loss=0.024.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_1/checkpoints/epoch=0069-step=095620-val_loss=0.022.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_1/checkpoints/epoch=0073-step=101084-val_loss=0.023.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_2/checkpoints/epoch=0093-step=128404-val_loss=0.033.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_2/checkpoints/epoch=0094-step=129770-val_loss=0.033.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_2/checkpoints/epoch=0096-step=132502-val_loss=0.032.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_2/checkpoints/epoch=0097-step=133868-val_loss=0.032.ckpt.pt
- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v8/lightning_logs/version_2/checkpoints/epoch=0103-step=142064-val_loss=0.034.ckpt.pt
" > "$HOME"/code/shitspotter/experiments/models.yaml



# specified models
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
#WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs228_20928c8c.kwcoco.zip
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_evals
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.heatmap_evaluation_pipeline()'
        #pipeline: 'shitspotter.pipelines.polygon_evaluation_pipeline()'
        matrix:
            heatmap_pred.package_fpath:
                # - $HOME/code/shitspotter/experiments/first_chosen_eval_batch1.yaml
                 - $HOME/code/shitspotter/experiments/models.yaml
                # - $DVC_DATA_DPATH/models/shitspotter_scratch_v025-version_2-epoch=1277-step=005112-val_loss=0.600.ckpt.pt
                #- $HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/shitspotter_scratch_20240618_noboxes_v4/lightning_logs/version_1/checkpoints/epoch=0076-step=105182-val_loss=0.018.ckpt.pt
            heatmap_pred.test_dataset:
                - $VALI_FPATH
            heatmap_eval.workers: 1
            heatmap_eval.draw_heatmaps: 0
            heatmap_eval.draw_curves: True
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1," --tmux_workers=1 \
    --backend=tmux --skip_existing=1 \
    --run=1


# Simple no-dependency result readout
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_evals
python -c "if 1:
    import ubelt as ub
    eval_fpaths = list(ub.Path('$EVAL_PATH/eval/flat/heatmap_eval').glob('*/pxl_eval.json'))
    import json
    datas = [json.loads(p.read_text()) | {'path': p} for p in ub.ProgIter(eval_fpaths, desc='load')]

    datas = sorted(datas, key=lambda r: r['nocls_measures']['ap'])

    for data in datas:
        root_dir = data['meta']['info'][-2]['properties']['extra']['fit_config']['trainer']['default_root_dir']

        package_fpath = ub.Path(data['meta']['info'][-2]['properties']['config']['package_fpath'])

        if 'noboxes_v' in ub.Path(root_dir).name:
            print(data['nocls_measures']['ap'], root_dir, data['path'])
"


# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_evals
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.heatmap_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/full_aggregate" \
    --resource_report=1 \
    --io_workers=0 \
    --eval_nodes="
        - heatmap_eval
    " \
    --stdout_report="
        top_k: 10
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 0
        show_csv: 0
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 2
        max_variations: 40
        min_support: 1
        params_of_interest:
            - resolved_params.heatmap_pred_fit.model.init_args.arch_name
            - resolved_params.heatmap_pred_fit.model.init_args.perterb_scale
            - resolved_params.heatmap_pred_fit.optimizer.init_args.lr
            - resolved_params.heatmap_pred_fit.optimizer.init_args.weight_decay
            - resolved_params.heatmap_pred_fit.trainer.default_root_dir
            - params.heatmap_pred.package_fpath
    "


# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_evals
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.heatmap_evaluation_pipeline()' \
    --target "
        - $EVAL_PATH
    " \
    --output_dpath="$EVAL_PATH/aggregate" \
    --resource_report=1 \
    --io_workers=0 \
    --eval_nodes="
        - heatmap_eval
    " \
    --stdout_report="
        top_k: 10
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 0
        show_csv: 0
    " \
    --plot_params="
        enabled: 1
        stats_ranking: 0
        min_variations: 2
        max_variations: 40
        min_support: 1
        params_of_interest:
            - resolved_params.heatmap_pred_fit.model.init_args.arch_name
            - resolved_params.heatmap_pred_fit.model.init_args.perterb_scale
            - resolved_params.heatmap_pred_fit.optimizer.init_args.lr
            - resolved_params.heatmap_pred_fit.optimizer.init_args.weight_decay
            - resolved_params.heatmap_pred_fit.trainer.default_root_dir
            - params.heatmap_pred.package_fpath
    " \
    --query="
    df['resolved_params.heatmap_pred_fit.trainer.default_root_dir'].apply(lambda p: str(p).split('/')[-1]).str.contains('noboxes')
    " \
    --custom_query="
        new_eval_type_to_aggregator = {}
        for key, agg in eval_type_to_aggregator.items():
            chosen_idxs = []
            for group_id, group in agg.table.groupby('resolved_params.heatmap_pred_fit.trainer.default_root_dir'):
                group['metrics.heatmap_eval.salient_AP'].argsort()
                keep_idxs = group['metrics.heatmap_eval.salient_AP'].sort_values()[-5:].index
                chosen_idxs.extend(keep_idxs)
            new_agg = agg.filterto(index=chosen_idxs)
            rich.print(f'Special filter {key} filtered to {len(new_agg)}/{len(agg)} rows')
            new_eval_type_to_aggregator[key] = new_agg

        if 1:
            new_agg.table
            new_agg.table.search_columns('lr')
            new_agg.table.search_columns('resolved_params.heatmap_pred_fit')
            new_agg.table.search_columns('resolved_params.heatmap_pred_fit.lr_scheduler.init_args.max_lr')
            new_agg.table['resolved_params.heatmap_pred_fit.lr_scheduler.init_args.max_lr']

            subcols = [
                'resolved_params.heatmap_pred_fit.trainer.default_root_dir',
                'resolved_params.heatmap_pred_fit.optimizer.init_args.lr',
                'resolved_params.heatmap_pred_fit.optimizer.init_args.weight_decay',
                'resolved_params.heatmap_pred_fit.model.init_args.perterb_scale',
                'metrics.heatmap_eval.salient_AP',
                'metrics.heatmap_eval.salient_AUC',
            ]
            new_agg.table[subcols]

            from geowatch.utils.util_pandas import pandas_shorten_columns, pandas_condense_paths

            chosen_idxs = []
            for group_id, group in agg.table.groupby('resolved_params.heatmap_pred_fit.trainer.default_root_dir'):
                group['metrics.heatmap_eval.salient_AP'].argsort()
                keep_idxs = group['metrics.heatmap_eval.salient_AP'].sort_values()[-1:].index
                chosen_idxs.extend(keep_idxs)

            table = new_agg.table.safe_drop(['resolved_params.heatmap_pred_fit.trainer.callbacks'], axis=1)
            varied = table.varied_value_counts(min_variations=2)
            print(list(varied.keys()))

            subtable = new_agg.table.loc[chosen_idxs, subcols]
            subtable = pandas_shorten_columns(subtable)
            subtable['default_root_dir'] = pandas_condense_paths(subtable['default_root_dir'])[0]
            subtable = subtable.sort_values(['salient_AP'], ascending=False)
            print(subtable.to_latex(index=False))
    "
