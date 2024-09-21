__doc__="
This installs detectron in development mode and applies a patch needed for our
dataset.

References:
    https://colab.research.google.com/drive/1DIk7bDpdZDkTTZyJbPADZklcbZKr1xkn#scrollTo=DvVulbjZcTdp
    https://detectron2.readthedocs.io/en/latest/tutorials/install.html
"
#python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install detectron2 from source
cd "$HOME"/code
git clone https://github.com/facebookresearch/detectron2.git
cd "$HOME"/code/detectron2
git checkout ebe8b45437f86395352ab13402ba45b75b4d1ddb
cd "$HOME"/code/detectron2

cd "$HOME"/code/detectron2
echo '
diff --git a/detectron2/data/detection_utils.py b/detectron2/data/detection_utils.py
index 8d6173e..4b7bc19 100644
--- a/detectron2/data/detection_utils.py
+++ b/detectron2/data/detection_utils.py
@@ -135,6 +135,7 @@ def _apply_exif_orientation(image):
     Returns:
         (PIL.Image): the PIL image with exif orientation applied, if applicable
     """
+    return image
     if not hasattr(image, "getexif"):
         return image
' | git apply

python -m pip install -e .



ls "$HOME"/data/dvc-repos/shitspotter_dvc
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
TRAIN_FPATH=$DVC_DATA_DPATH/train_imgs5747_1e73d54f.kwcoco.zip
VALI_FPATH=$DVC_DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip
TRAIN_FPATH2=$DVC_DATA_DPATH/train_imgs5747_1e73d54f.mscoco.json
VALI_FPATH2=$DVC_DATA_DPATH/vali_imgs691_99b22ad0.mscoco.json

kwcoco conform --legacy=True --src "$TRAIN_FPATH" --dst "$TRAIN_FPATH2"
kwcoco conform --legacy=True --src "$VALI_FPATH" --dst "$VALI_FPATH2"
