VIT GeoWATCH Experiments
========================

These experiments use the Author's GeoWATCH system to explore high resolution
segmentation of the dataset.

For our 2025 paper, the training script that produced the model in the main
paper was `train_toothbrush_scratch_noboxes_v7.sh <./train_toothbrush_scratch_noboxes_v7.sh>`_.

The models were evaluated with

`run_pixel_eval_on_test_pipeline.sh <./run_pixel_eval_on_test_pipeline.sh>`_. and
`run_pixel_eval_on_vali_pipeline.sh <./run_pixel_eval_on_vali_pipeline.sh>`_.

Reproducing these experiments requires a bit of setup that should be inferable from the information in this repo, but raise an issue if you want to reproduce these and need help.
