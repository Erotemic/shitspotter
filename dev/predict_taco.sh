# https://github.com/jeremy-rico/litter-detection

new_pyenv_venv  ultralytics
pip install ultralytics


cd "$HOME"/code
git clone https://github.com/Erotemic/litter-detection
git clone https://github.com/pedropro/TACO.git
git clone https://github.com/dbash/zerowaste.git

ls "$HOME"/code/litter-detection/runs/detect/train/yolov8m_100epochs

python -c "if 1:
    import ubelt as ub
    package_fpath = ub.Path('~/code/litter-detection/runs/detect/train/yolov8m_100epochs/weights/best.pt').expand()
    kwargs = {}
    kwargs['package_fpath'] = package_fpath
    cmdline = 0

    kwargs['src'] = '/home/joncrall/code/shitspotter/shitspotter_dvc/vali.kwcoco.zip'
"

mkdir -p /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_tmp/test-ultralytics-taco
python -m shitspotter.cli.predict_ultralytics \
    /home/joncrall/code/shitspotter/shitspotter_dvc/vali.kwcoco.zip \
    /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_tmp/test-ultralytics-taco/pred.kwcoco.zip \
    --package_fpath ~/code/litter-detection/runs/detect/train/yolov8m_100epochs/weights/best.pt


cd ~/code/zerowaste

# train deeplab on ZeroWaste data
python deeplab/train_net.py \
    --config-file ~/code/zerowaste/deeplab/configs/zerowaste_config.yaml \
    --dataroot /path/to/zerowaste/data/ \
    (optional) \
    --resume OUTPUT_DIR /deeplab/outputs/*experiment_name* \
    (optional) \
    MODEL.WEIGHTS /path/to/checkpoint.pth

# train Mask R-CNN on ZeroWaste\TACO-zerowaste data
#python maskrcnn/train_net.py --config-file maskrcnn/configs/*config*.yaml (optional, only use if trained on TACO-zerowaste) --taco --dataroot /path/to/zerowaste/data/ (optional) --resume OUTPUT_DIR /maskrcnn/outputs/*experiment_name* (optional) --MODEL.WEIGHTS /path/to/checkpoint.pth

# train ReCo on ZeroWasteAug data
#python reco_aug/train_sup.py --dataset zerowaste --num_labels 0 --seed 1


# train deeplab on ZeroWaste data
python deeplab/train_net.py --config-file deeplab/configs/zerowaste_config.yaml --dataroot /path/to/zerowaste/data/ (optional) --resume OUTPUT_DIR /deeplab/outputs/*experiment_name* (optional) MODEL.WEIGHTS /path/to/checkpoint.pth

