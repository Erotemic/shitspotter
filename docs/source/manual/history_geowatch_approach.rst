Historical GeoWATCH algorithm
=============================

The first reasonable models trained were around done ~2023 with the `geowatch
<https://gitlab.kitware.com/computer-vision/geowatch>`_ framework on the
annotations existant at the time. These were VIT-based pixel segmentation
models that used far to much resolution. These models were superceded by
MaskRCNN, YOLOv9, GroundingDINO, and others. This blurb was originally part of
the README and is now being preserved here.


Update: 2023-10-15

The `geowatch <https://gitlab.kitware.com/computer-vision/geowatch>`_ framework
is being used to train initial models on the small set of annotations.


Initial train and validation batches look like this:

.. image:: https://i.imgur.com/Nfk8XbE.jpg


.. image:: https://i.imgur.com/YHfl0Wd.jpg


An example prediction from an initial model on a full validation image is:

.. image:: https://i.imgur.com/ya4jnAO.jpg


Clearly there is still more work to do, but training a deep network is an art,
and I have full confidence that a high quality model is possible. The training
batches are starting to fit the data, but the validation batches shows that
there is still a clear generalization gap, but this is only the very start of
training and the hyper-parameters are untuned.


The current train validation split is defined in the ``make_splits.py`` file.
Only "before" images with annotations are currently considered. The "after"
images and "negative" will be taken into account when they are properly
associated with the "before" images in the kwcoco metadata. The early images
before 2021 are used for validation, whereas everything else is used for
training. Contributor data is also currently held out and can serve as a test
set once annotations are placed.


Update 2024-03-31: Recent results from model ``shitspotter_from_v027_halfres_v028-epoch=0179-step=000720-val_loss=0.005.ckpt.pt`` have been quite good. These have quantitatively been measured against the ``vali_imgs228_20928c8c.kwcoco.zip`` variant of the validation dataset. The precision recall and ROC curves for pixelwise binary poop/no-poop classification are:


.. image:: https://i.imgur.com/rgGjAda.png

And the corresponding threshold versus F1, G1, and MCC is:

.. image:: https://i.imgur.com/vay6TEP.png

Qualitatively some cherry-picked success cases in challenging images look like:


.. image:: https://i.imgur.com/oWPg4CE.jpeg

There still are false positives and false negatives in some of the more
challenging images, but the algorithm is now accurate enough where it can be
used, and it will continue to improve.

