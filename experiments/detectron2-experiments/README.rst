Detectron2 Experiments
======================

This directory contains experiments using detectron2 training system and models.

Instructions to setup detectron are in: `setup_detectron.sh <./setup_detectron.sh>`_

Experiment Layout
-----------------

* `train_maskrcnn_v1.py <./train_maskrcnn_v1.py>`_ - Trains using defaults similar to TACO


Detectron2 Pain Points
----------------------

* Dataset registration, I typically don't like systems where items need to be pre-registered, but this isn't the end of the world.

* Trainer seems to ignore images without annotations. This means it cannot use
  after or negative images as hard cases to mitigate false positives.

* EXIF respected by default whereas kwcoco does not handle this.

* Configs require knowing the path to the code directory instead of having a
  resource directory that can be accessed programatically
