#!/bin/bash

# In the main uncompressed IPFS data repo
cd "$HOME"/code/shitspotter/shitspotter_dvc
kwcoco union data.kwcoco.json test.kwcoco.zip --dst full.kwcoco.zip
kwcoco modify_categories --keep poop --src full.kwcoco.zip --dst full-poop-only.kwcoco.zip

kwcoco stats full-poop-only.kwcoco.zip

kwcoco plot_stats full-poop-only.kwcoco.zip --dst_dpath full_plots \
        --options "
            split_y: 20
            all_polygons:
                facecolor: 'baby shit brown'
            "


python -c "if 1:
    import kwcoco
    dset = kwcoco.CocoDataset('full-poop-only.kwcoco.zip')
    whs = list(map(tuple, map(sorted, zip(*dset.images().lookup(['width', 'height']).values()))))
    histo = ub.dict_hist(whs)
    print('Image Size Histogram')
    print(ub.urepr(histo))
"
