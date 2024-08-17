import kwimage
import ubelt as ub

figure_dpath = ub.Path('$HOME/code/shitspotter/papers/application-2024/figures').expand()

# Test Images (with best test model)
paths = '''
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-15.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-8.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-10.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-23.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-26.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-30.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-24.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-1.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-6.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689/heatmaps/_loose_images/_loose_images-None-4.jpg
'''.strip().split(chr(10))
images = [kwimage.imread(p) for p in paths if p and not p.startswith('#')]
canvas = kwimage.stack_images(images, axis=1, resize='larger', pad=100, bg_value='white')
dpath = ub.Path('/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689')
fpath = dpath / 'test_heatmaps_with_best_test_model.jpg'
canvas = kwimage.imresize(canvas, max_dim=4096)
print(f'canvas.shape={canvas.shape}')
kwimage.imwrite(fpath, canvas)
fpath.copy(figure_dpath / fpath.name, overwrite=True)


# Validation Images (with best validation model)
paths = '''
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-5513.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-12.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-113.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-156.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-99.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-90.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-5460.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-177.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-18.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-121.jpg
/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74/heatmaps/_loose_images/_loose_images-None-74.jpg
'''.strip().split(chr(10))
images = [kwimage.imread(p) for p in paths if p and not p.startswith('#')]
canvas = kwimage.stack_images(images, axis=1, resize='larger', pad=100, bg_value='white')
dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74')
fpath = dpath / 'vali_heatmaps_with_best_vali_model.jpg'
canvas = kwimage.imresize(canvas, max_dim=4096)
print(f'canvas.shape={canvas.shape}')
kwimage.imwrite(fpath, canvas)
fpath.copy(figure_dpath / fpath.name, overwrite=True)


# Test Images (with best validation model)
paths = '''
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-15.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-8.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-10.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-23.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-26.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-30.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-24.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-1.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-6.jpg
/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533/heatmaps/_loose_images/_loose_images-None-4.jpg
'''.strip().split(chr(10))
images = [kwimage.imread(p) for p in paths if p and not p.startswith('#')]
canvas = kwimage.stack_images(images, axis=1, resize='larger', pad=100, bg_value='white')
dpath = ub.Path('/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_0f613533')
fpath = dpath / 'test_heatmaps_with_best_vali_model.jpg'
canvas = kwimage.imresize(canvas, max_dim=4096)
print(f'canvas.shape={canvas.shape}')
kwimage.imwrite(fpath, canvas)
fpath.copy(figure_dpath / fpath.name, overwrite=True)


#### MISS THAT NEED FIXES
# assets/poop-2023-07-01-T160318/PXL_20230418_211636033.jpg
# assets/poop-2023-08-22-T202656/PXL_20230723_143928317.jpg
# file:///home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-321.jpg
#file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0041-gid-04261.jpg
# file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0008-gid-03315.jpg
#file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0009-gid-01353.jpg
# file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0012-gid-04304.jpg
# file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0018-gid-04289.jpg
# file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1885.jpg
# file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-2275.jpg
# file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-6357.jpg
# file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-6538.jpg


# Train Images (with best validation model)
# Number of missed annotations found: 2
dpath = ub.Path('/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/')
paths = '''

# Bee Snoot
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1316.jpg

# Hard FP
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1450.jpg

# Bee in muzzle
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-2456.jpg

# HARD TP, minor FP, BIG PILE OF LEAFS, lefas
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1862.jpg

# TP, minor FP, leafs
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1785.jpg

# TP, minor FP, rain, leafs
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1757.jpg

# True positive, minor false positive, leafs
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1726.jpg

# True positive, medium false positive, nighttime
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1520.jpg

# Medium to moderate false positives, leafs
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1527.jpg

# medium FP stick
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-3298.jpg

# hard FP stick
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0009-gid-02091.jpg

# Hard Pine Cone
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0009-gid-05295.jpg

# strong FP, cones, leaf, TP,
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-3396.jpg

# med FP, weird road thing
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4673.jpg

# med FP, leaf
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4678.jpg

# strong FP, leaf
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4711.jpg

# Bee Shadow, strong TP concrete
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4920.jpg

# weak TP,
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5019.jpg

# FP leaf, concrete
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5054.jpg

# TP, Strong FP, stick
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5145.jpg

# Strong FP, weak TP (HARD CASE)
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5157.jpg


# Snoot is a hard false positive
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0045-gid-05519.jpg

# Hard FP in a tulip
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0087-gid-04287.jpg

# Very hard leaf FP, with weak TP
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0030-gid-05061.jpg

# Stairs FP
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0008-gid-06517.jpg

# Poop Everywhere
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0010-gid-05841.jpg

# Dark Region, Snow Log
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0011-gid-03766.jpg

# Bee and Honey Blury
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-631.jpg

# Bee, good picture, nighttime, True positive
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-871.jpg

# Honey, good picture, night time, false positive
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-873.jpg

# Bee back head
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-3179.jpg

# Bee Face,
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-2294.jpg

# Easy Positive, Similar to hard Test Case
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4114.jpg
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4201.jpg

# Strong false positives
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4301.jpg


# ITSSSS GRATE!
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4889.jpg

# Fog, kinda hard
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5022.jpg

# Difficult leaf case
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5025.jpg
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5026.jpg
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5027.jpg

# Hard Rain, Stick
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5031.jpg

# Hard Dark Case
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5244.jpg

# Roadie
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5597.jpg
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5799.jpg
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-6252.jpg

'''.strip().split(chr(10))

# Chosen
paths = '''

# Easy Positive, Similar to hard Test Case
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4201.jpg

# Bee, good picture, nighttime, True positive
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-871.jpg

# TP, minor FP, leafs
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1785.jpg

# HARD TP, minor FP, BIG PILE OF LEAFS, lefas
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-1862.jpg

# Roadie
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5799.jpg

# Bee and Honey Blury
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-631.jpg

# hard FP stick
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0009-gid-02091.jpg

# Hard Pine Cone
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0009-gid-05295.jpg

# Strong FP, weak TP (HARD CASE)
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-5157.jpg

# Snoot is a hard false positive
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0045-gid-05519.jpg

# Strong false positives
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_loose_images/_loose_images-None-4301.jpg

# Hard FP in a tulip
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0087-gid-04287.jpg

# Very hard leaf FP, with weak TP
file:///data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/heatmaps/_by_fpr/fpr-000.0030-gid-05061.jpg

'''.strip().split(chr(10))


def fixup_path(p):
    prefix = 'file://'
    if p.startswith(prefix):
        p = p[len(prefix):]
    return p

paths = [fixup_path(p) for p in paths if p and not p.startswith('#')]
print(len(paths))
images = [
    kwimage.imread(p)
    # kwimage.imresize(kwimage.imread(p), max_dim=1024)
    for p in ub.ProgIter(paths) if p and not p.startswith('#')
]
canvas = kwimage.stack_images(images, axis=1, resize='larger', pad=100, bg_value='white')
fpath = dpath / 'train_heatmaps_with_best_vali_model.jpg'
canvas = kwimage.imresize(canvas, max_dim=8096)
print(f'canvas.shape={canvas.shape}')
kwimage.imwrite(fpath, canvas)
fpath.copy(figure_dpath / fpath.name, overwrite=True)
