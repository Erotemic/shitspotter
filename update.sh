#!/bin/bash
__doc__="
This script updates the main kwcoco files based on labelme annotations and
produces the data splits.
"

echo "
To add a new cohort of data use

* Plug phone into computer.

* In USB preferences, enable 'File trasfer / Android Auto'.

* Run code to transfer and organize new images

.. code::

    python -m shitspotter.phone_manager

This will transfer data into a staging area.

* Add manual annotations with labelme (or maybe bootstrap with existing models? todo)
"

# The gather script
python -m shitspotter.gather

# The train/vali splits
python -m shitspotter.make_splits

# The matching script
python -m shitspotter.matching autofind_pair_hueristic

# The plots script
python -m shitspotter.plots update_analysis_plots
