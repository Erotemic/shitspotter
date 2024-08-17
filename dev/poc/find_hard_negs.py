import ubelt as ub
import pandas as pd
path = ub.Path('perimage_table.json')
table = pd.read_json(path, orient='table')

subtable = table[table['realpos_total'] > 0]

ordered = subtable.sort_values('salient_max_f1_fpr', ascending=False)

dpath = ub.Path('/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301')
img_dpath = (dpath / 'heatmaps/_loose_images')
link_dpath = (dpath / 'heatmaps/_by_fpr').ensuredir()
for _, row in ordered.iloc[0:100].iterrows():
    gid = row['true_gid']
    fpr = row['salient_max_f1_fpr']
    img_fpath = list(img_dpath.glob(f'*-{gid}.jpg'))[0]

    link_fpath = link_dpath / f'fpr-{fpr:08.4f}-gid-{gid:05d}.jpg'
    ub.symlink(img_fpath, link_fpath)

    # link_fpath.symlink_to(img_fpath)
