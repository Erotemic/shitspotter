#export CUDA_VISIBLE_DEVICES=0,1
#DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
#DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
#test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
#test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"
#KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
#TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs5747_1e73d54f.mscoco.json
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.mscoco.json

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_evals/test_imgs121_6cb3b6ff
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.other.grounding_dino_pipeline.grounding_dino_evaluation_pipeline()'
        matrix:
            grounding_dino_pred.src:
                - '$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip'
            grounding_dino_pred.classes:
                - '[poop]'
                - '[dogpoop]'
                - '[feces]'
                - '[dogfeces]'
                - '[excrement]'
                - '[droppings]'
                - '[turd]'
                - '[stool]'
                - '[caninefeces]'
                - '[animalfeces]'
                - '[petwaste]'
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1," --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --run=1

# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_evals/test_imgs121_6cb3b6ff
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.other.grounding_dino_pipeline.grounding_dino_evaluation_pipeline()' \
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


# --- Run on validation data

export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags="shitspotter_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
test -e "$DVC_EXPT_DPATH" || echo "CANNOT FIND EXPT"
test -e "$DVC_DATA_DPATH" || echo "CANNOT FIND DATA"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_evals/vali_imgs691_99b22ad0
python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.other.grounding_dino_pipeline.grounding_dino_evaluation_pipeline()'
        matrix:
            grounding_dino_pred.src:
                - '$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.kwcoco.zip'
            grounding_dino_pred.classes:
                - '[poop]'
                - '[dogpoop]'
                - '[feces]'
                - '[dogfeces]'
                - '[excrement]'
                - '[droppings]'
                - '[turd]'
                - '[stool]'
                - '[caninefeces]'
                - '[animalfeces]'
                - '[petwaste]'
            grounding_dino_pred.force_classname:
                - poop
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0,1," --tmux_workers=2 \
    --backend=tmux --skip_existing=1 \
    --run=1

# Result aggregation and reporting
DVC_EXPT_DPATH=$(geowatch_dvc --tags="shitspotter_expt")
EVAL_PATH=$DVC_EXPT_DPATH/_shitspotter_2025_rebutal_evals/vali_imgs691_99b22ad0
python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.other.grounding_dino_pipeline.grounding_dino_evaluation_pipeline()' \
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



#python -m shitspotter.other.predict_grounding_dino \
#    --src "$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip" \
#    --dst "$EVAL_PATH/preds/grounding_dino/test_imgs121_6cb3b6ff/prompt-poop/pred.kwcoco.zip" \
#    --classes "[poop]"

#kwcoco eval_detections \
#    --true_dataset "$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip" \
#    --pred_dataset "$EVAL_PATH/preds/grounding_dino/test_imgs121_6cb3b6ff/prompt-poop/pred.kwcoco.zip" \
#    --out_dpath "$EVAL_PATH/preds/grounding_dino/test_imgs121_6cb3b6ff/prompt-poop/detection_evaluation"

## ---
#python -m shitspotter.other.predict_grounding_dino \
#    --src "$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip" \
#    --dst "$EVAL_PATH/preds/grounding_dino/test_imgs121_6cb3b6ff/prompt-feces/pred.kwcoco.zip" \
#    --force_classname="poop" \
#    --classes "[feces]"

#kwcoco eval_detections \
#    --true_dataset "$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip" \
#    --pred_dataset "$EVAL_PATH/preds/grounding_dino/test_imgs121_6cb3b6ff/prompt-feces/pred.kwcoco.zip" \
#    --out_dpath "$EVAL_PATH/preds/grounding_dino/test_imgs121_6cb3b6ff/prompt-feces/detection_evaluation"
## ---



#--------

python -m shitspotter.other.predict_grounding_dino \
    --src "$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.mscoco.json" \
    --dst "$EVAL_PATH/preds/grounding_dino/vali_imgs691_99b22ad0/prompt-poop/pred.kwcoco.zip" \
    --classes "[poop]"

kwcoco eval_detections \
    --true_dataset "$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.kwcoco.zip" \
    --pred_dataset "$EVAL_PATH/preds/grounding_dino/vali_imgs691_99b22ad0/prompt-poop/pred.kwcoco.zip" \
    --out_dpath "$EVAL_PATH/preds/grounding_dino/vali_imgs691_99b22ad0/prompt-poop/detection_evaluation"
# ---

