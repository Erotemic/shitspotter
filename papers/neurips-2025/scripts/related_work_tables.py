# https://github.com/AgaMiko/waste-datasets-review
import rich
import ubelt as ub
import kwcoco
import pandas as pd
from kwcoco.cli import coco_plot_stats

"""
cd /data/joncrall/dvc-repos/MSHIT/
kwcoco union --src train.kwcoco.zip  verification.kwcoco.zip --dst trainval.kwcoco.zip
kwcoco union --src test.kwcoco.zip train.kwcoco.zip  verification.kwcoco.zip --dst all.kwcoco.zip
kwcoco plot_stats trainval.kwcoco.zip

kwcoco union --src \
    /data/joncrall/dvc-repos/mscoco2017/coco2017/annotations/instances_train2017.json \
    /data/joncrall/dvc-repos/mscoco2017/coco2017/annotations/instances_val2017.json \
    --dst /data/joncrall/dvc-repos/mscoco2017/coco2017/annotations/instances_trainval2017.json
kwcoco stats /data/joncrall/dvc-repos/mscoco2017/coco2017/annotations/instances_trainval2017.json
kwcoco plot_stats /data/joncrall/dvc-repos/mscoco2017/coco2017/annotations/instances_trainval2017.json


cd /home/joncrall/data/dvc-repos/TACO/data
kwcoco stats annotations.json
kwcoco plot_stats /data/joncrall/dvc-repos/mscoco2017/coco2017/annotations/instances_trainval2017.json

cd /data/store/data/ImageNet/ILSVRC
kwcoco union --src train.kwcoco.zip  val.kwcoco.zip --dst ilsvrc2017_full.kwcoco.zip
kwcoco stats ilsvrc2017_full.kwcoco.zip


kwcoco union test.kwcoco.zip train.kwcoco.zip val.kwcoco.zip --dst cityscapes_all.kwcoco.zip


cd $HOME/code/shitspotter/shitspotter_dvc
kwcoco union data.kwcoco.json test.kwcoco.zip --dst full.kwcoco.zip
kwcoco modify_categories --keep poop --src full.kwcoco.zip --dst full-poop-only.kwcoco.zip
kwcoco stats full-poop-only.kwcoco.zip
kwcoco plot_stats full-poop-only.kwcoco.zip --dst_dpath full_plots \
        --options "
            split_y: 20
            all_polygons:
                facecolor: 'baby shit brown'
            "

        # --plots all_polygons \


SeeAlso:

    # TODO
    /home/joncrall/data/dvc-repos/SpotGarbage/
    /home/joncrall/data/dvc-repos/MJU-Waste-v1/

    # Shown
    /home/joncrall/data/dvc-repos/TACO/

"""

rows = []
rows.extend([
    {
        'name': r'ImageNet LSVRC2017 \cite{ILSVRC15}',
        'size': '166GB',
        'n_anns': 695776,
        'n_imgs': 594546,
        'n_videos': 0,
        'n_cats': 1000,
        'n_tracks': 0,
        'images_with_eq0_anns': 0,
        'images_with_ge1_anns': 594546,
        'frac_images_with_ge1_anns': 1.0,
        'frac_images_with_eq0_anns': 0.0,
        'median_area_image_size': (500, 374),
        'most_frequent_image_size': '4080 x 5935',
        'median_box_rt_area': 239.02301144450507,
        'median_box_dsize': (276.0, 276.0),
        'median_sseg_rt_area': 239.02301144450507,
        'median_sseg_box_dsize': (276.0, 276.0),
        'median_sseg_obox_dsize': (276.0, 207.0),
        'table_buildtime_seconds': 192.92026164399977,
        'coco_fpath': '/data/joncrall/dvc-repos/ImageNet/ILSVRC/ilsvrc2017_full.kwcoco.zip',
        'annot_type': 'box',
    },
    {
        'name': r'MSCOCO 2017 \cite{lin_microsoft_2014}',
        'size': '50GB',
        'n_anns': 896782,
        'n_imgs': 123287,
        'n_videos': 0,
        'n_cats': 80,
        'n_tracks': 0,
        'images_with_eq0_anns': 1069,
        'images_with_ge1_anns': 122218,
        'frac_images_with_ge1_anns': 0.9913291750143973,
        'frac_images_with_eq0_anns': 0.008670824985602699,
        'median_area_image_size': (428, 640),
        'most_frequent_image_size': '640 x 640',
        'median_box_rt_area': 57.44600247188659,
        'median_box_dsize': (92.49, 92.49),
        'table_buildtime_seconds': 141.64544131500043,
        'coco_fpath': '/data/joncrall/dvc-repos/mscoco2017/coco2017/annotations/instances_trainval2017.json',
        'annot_type': 'polygon',
    },
    {
        'name': r'CityScapes \cite{cordts2015cityscapes}',
        'size': '78GB',
        'coco_fpath': '/data/torrents/Cityscapes/cityscapes_all.kwcoco.zip',
        'n_anns': 287465,
        'n_imgs': 5000,
        'n_videos': 27,
        'n_cats': 40,
        'n_tracks': 0,
        'images_with_eq0_anns': 0,
        'images_with_ge1_anns': 5000,
        'frac_images_with_ge1_anns': 1.0,
        'frac_images_with_eq0_anns': 0.0,
        'median_area_image_size': (2048, 1024),
        'most_frequent_image_size': '2048 x 1024',
        'median_box_rt_area': 50.84289527554464,
        'median_box_dsize': (47.0, 47.0),
        'table_buildtime_seconds': 37.033869935999974,
        'annot_type': 'polygon',
    },
    {
        'name': r'ZeroWaste \cite{bashkirova_zerowaste_2022}',
        'size': '10GB',
        'coco_fpath': '/data/joncrall/dvc-repos/ZeroWaste/zerowaste-f/splits_final_deblurred/all.kwcoco.zip',
        'n_anns': 26766,
        'n_imgs': 4503,
        'n_videos': 0,
        'n_cats': 4,
        'n_tracks': 0,
        'images_with_eq0_anns': 86,
        'images_with_ge1_anns': 4417,
        'frac_images_with_ge1_anns': 0.9809016211414613,
        'frac_images_with_eq0_anns': 0.019098378858538753,
        'median_area_image_size': (1920, 1080),
        'most_frequent_image_size': '1920 x 1080',
        'median_box_rt_area': 270.9623959150051,
        'median_box_dsize': (237.3, 237.3),
        'median_sseg_rt_area': 200.03687160121257,
        'median_sseg_box_dsize': (237.3, 237.3),
        'median_sseg_obox_dsize': (303.660400390625, 224.81057739257812),
        'annot_type': 'polygon',
    },
    {
        'name': r'TACO \cite{proenca_taco_2020}',
        'n_anns': 4784,
        'n_imgs': 1500,
        'n_videos': 0,
        'n_cats': 60,
        'n_tracks': 0,
        'images_with_eq0_anns': 0,
        'images_with_ge1_anns': 1500,
        'frac_images_with_ge1_anns': 1.0,
        'frac_images_with_eq0_anns': 0.0,
        'median_area_image_size': (2448, 3264),
        'most_frequent_image_size': '6000 x 4000',
        'median_box_rt_area': 171.49927113547741,
        'median_box_dsize': (172.0, 172.0),
        'median_sseg_rt_area': 119.37336386313322,
        'median_sseg_box_dsize': (172.0, 172.0),
        'median_sseg_obox_dsize': (168.45718383789062, 155.00772094726562),
        'table_buildtime_seconds': 1.5557497509998939,
        'size': '17GB',
        # 'median_area_image_size': '428 x 640',
        'coco_fpath': '/home/joncrall/data/dvc-repos/TACO/data/annotations.json',
        'annot_type': 'polygon',
    },
    {
        'name': r'TrashCanV1 \cite{hong2020trashcansemanticallysegmenteddatasetvisual}',
        'size': '0.61GB',
        'n_anns': 12128,
        'n_imgs': 7212,
        'n_videos': 425,
        'n_cats': 22,
        'n_tracks': 0,
        'images_with_eq0_anns': 129,
        'images_with_ge1_anns': 7083,
        'frac_images_with_ge1_anns': 0.9821131447587355,
        'frac_images_with_eq0_anns': 0.01788685524126456,
        'median_area_image_size': (480, 270),
        'most_frequent_image_size': '480 x 360',
        'median_box_rt_area': 78.36698494330498,
        'median_box_dsize': (82.99607843137255, 82.99607843137255),
        'median_sseg_rt_area': 54.217707546620964,
        'median_sseg_box_dsize': (82.99607843137255, 82.99607843137255),
        'median_sseg_obox_dsize': (87.39673614501953, 69.20834350585938),
        'coco_fpath': '/home/joncrall/data/dvc-repos/TrashCan-v1/trashcan_instance_trainval.kwcoco.zip',
        'annot_type': 'polygon',
    },

    {
        'name': r'SpotGarbage-GINI \cite{mittal2016spotgarbage}',
        'size': '1.5GB',
        'n_anns': 337,
        'n_imgs': 2512,
        'n_videos': 0,
        'n_cats': 1,
        'n_tracks': 0,
        'images_with_eq0_anns': 2175,
        'images_with_ge1_anns': 337,
        'frac_images_with_ge1_anns': 0.13415605095541402,
        'frac_images_with_eq0_anns': 0.865843949044586,
        'median_area_image_size': (754, 754),
        'most_frequent_image_size': '8000 x 11000',
        'median_box_rt_area': 355.5123063974017,
        'median_box_dsize': (211.0, 211.0),
        'median_sseg_rt_area': 355.5123063974017,
        'median_sseg_box_dsize': (211.0, 211.0),
        'median_sseg_obox_dsize': (599.0, 211.0),
        'table_buildtime_seconds': 0.30256412199742044,
        'kwcoco_loadtime_seconds': 0.011875015999976313,
        'coco_fpath': '/data/joncrall/dvc-repos/SpotGarbage-GINI/all.kwcoco.zip',
        'annot_type': 'classification',
    },
    {
        'name': r'UAVVaste \cite{rs13050965}',
        'size': '2.9GB',
        'n_anns': 3718,
        'n_imgs': 772,
        'n_videos': 0,
        'n_cats': 1,
        'n_tracks': 0,
        'images_with_eq0_anns': 0,
        'images_with_ge1_anns': 772,
        'frac_images_with_ge1_anns': 1.0,
        'frac_images_with_eq0_anns': 0.0,
        'median_area_image_size': (3840, 2160),
        'most_frequent_image_size': '4056 x 3040',
        'median_box_rt_area': 72.24956747275377,
        'median_box_dsize': (58.0, 58.0),
        'median_sseg_rt_area': 55.00454526673228,
        'median_sseg_box_dsize': (58.0, 58.0),
        'median_sseg_obox_dsize': (95.25048065185547, 36.03748321533203),
        'table_buildtime_seconds': 1.2786736369998835,
        'kwcoco_loadtime_seconds': 0.030174073999660322,
        'coco_fpath': '/data/joncrall/dvc-repos/UAVVaste/annotations/annotations.json',
        'annot_type': 'polygon',
    },
    {
        'name': r'MSHIT \cite{mshit_2020}',
        'n_anns': 2348,
        'n_imgs': 769,
        'n_cats': 2,
        'size': '4GB',
        'median_area_image_size': (960, 540),
        'median_sseg_rt_area': 99.0,
        'notes': 'object of interest is plastic poop toy',
        'url': 'https://www.kaggle.com/datasets/mikian/dog-poop',
        'coco_fpath': '/data/joncrall/dvc-repos/MSHIT/all.kwcoco.zip',
        'annot_type': 'box',
    },
    {
        'name': '``ScatSpotter\'\' (ours)',
        'size': '61GB',
        'n_imgs': 9_296,
        'n_anns': 6_594,
        'n_cats': 1,
        'median_area_image_size': (4032, 3024),
        'median_sseg_rt_area': 87.0,
        'notes': 'real picture of real poop',
        'coco_fpath': '/home/joncrall/code/shitspotter/shitspotter_dvc/full-poop-only.kwcoco.json',
        'annot_type': 'polygon',
    },
])

for row in rows:
    if 'median_box_rt_area' in row and 'median_sseg_rt_area' not in row:
        row['median_sseg_rt_area'] = row['median_box_rt_area']


if 0:
    memo_coco = ub.memoize(kwcoco.CocoDataset)
    for row in rows:
        if 'coco_fpath' in row:
            coco_fpath = row['coco_fpath']
            dset = memo_coco(coco_fpath)
            info = coco_plot_stats.build_stats_data(dset)
            scalar_stats, tables_data, nonsaved_data, dataframes = info
            scalar_stats = ub.udict(scalar_stats)
            row = ub.udict(row)
            common = set(row & scalar_stats)
            diff = ub.IndexableWalker(row & common).diff(scalar_stats & common)
            rich.print(f'scalar_stats = {ub.urepr(scalar_stats, nl=-1)}')
            rich.print(f'row = {ub.urepr(row, nl=2)}')
            if diff['similarity'] != 1.0:
                print(f'diff = {ub.urepr(diff, nl=2)}')
            print('---')

table = pd.DataFrame(rows)

predisplay_table = table[[
    'name', 'n_cats',
    'n_imgs',
    'n_anns',
    'median_area_image_size',
    'median_sseg_rt_area',
    'size',
    'annot_type',
]]

predisplay_table['n_cats'] = predisplay_table['n_cats'].apply(lambda x: int(x) if not pd.isna(x) else '-')
predisplay_table['n_imgs'] = predisplay_table['n_imgs'].apply(lambda x: int(x) if not pd.isna(x) else '-')
predisplay_table['n_anns'] = predisplay_table['n_anns'].apply(lambda x: int(x) if not pd.isna(x) else '-')
predisplay_table['median_sseg_rt_area'] = predisplay_table['median_sseg_rt_area'].round(1).astype(int)


def humanize_num(num):
    chunks = list(ub.chunks(str(num)[::-1], chunksize=3))
    chunks = [''.join(c) for c in chunks]
    recon = ','.join(chunks)
    final = recon[::-1]
    return final


def humanize_imagesize(tup):
    a = humanize_num(tup[0])
    b = humanize_num(tup[1])
    return f'{a} \\times {b}'
    ...

predisplay_table['n_anns'] = predisplay_table['n_anns'].apply(humanize_num)
predisplay_table['n_imgs'] = predisplay_table['n_imgs'].apply(humanize_num)
predisplay_table['n_cats'] = predisplay_table['n_cats'].apply(humanize_num)
predisplay_table['median_area_image_size'] = predisplay_table['median_area_image_size'].apply(humanize_imagesize)

mapping = {
    'name': 'Name',
    'size': 'Size',
    'n_cats': 'Categories',
    'n_imgs': 'Images',
    'n_anns': 'Annotations',
    'median_area_image_size': 'Image Size',
    'median_sseg_rt_area': 'Annot Size',
    'annot_type': 'Annot Type',
}
display_table = predisplay_table.rename(mapping, axis=1)
rich.print(display_table)

print(display_table.to_latex())

display_table = display_table.set_index('Name', drop=True)

latex_text = display_table.style.to_latex(
    hrules=True,
    label='tab:related_datasets',
    position='t',
    environment='table*',
    column_format='llrrrcrrl',
    caption=ub.paragraph(
        '''
        Related Datasets.
        Image Size is the pixel width and height of the image with the median area.
        Annot Size is the median sqrt(area) in pixels of the annotation polygon or box.
        The Size column refers to the amount of bytes needed download the
        entire dataset.
        Annot Type refers to if the dataset is annotated with bounding boxes,
        image-level classification labels, or polygon segmentations.
        Of the datasets in this table, ours has the highest image resolution
        and the smallest annotation size relative to that resolution.
        Of the waste related datasets, ours is among the largest, and it is
        currently the largest publicly available poop detection dataset that we
        are aware of.
        '''))
# text = latex_text.replace('{table}', '{table*}')
# Fix caption
lines = latex_text.split('\n')
for idx, line in enumerate(lines):
    if line.startswith('\\caption'):
        cap_pos = idx
    if line.startswith('\\label'):
        label_pos = idx
    if line.startswith('\\end{tabular}'):
        tab_pos = idx

start_part = lines[:cap_pos]
cap_part = lines[cap_pos:label_pos + 1]
data_part = lines[label_pos + 1:tab_pos + 1]
end_part = lines[tab_pos + 1:]
text = '\n'.join((start_part + data_part + cap_part + end_part))
print(text)
