import kwimage
import ubelt as ub

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
images = [kwimage.imread(p) for p in paths if p]
canvas = kwimage.stack_images(images, axis=1, resize='larger', pad=100, bg_value='white')
dpath = ub.Path('/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_test_evals/eval/flat/heatmap_eval/heatmap_eval_id_37a99689')
fpath = dpath / 'test_result_heatmaps.jpg'
canvas = kwimage.imresize(canvas, max_dim=4096)
print(f'canvas.shape={canvas.shape}')
kwimage.imwrite(fpath, canvas)


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
images = [kwimage.imread(p) for p in paths if p]
canvas = kwimage.stack_images(images, axis=1, resize='larger', pad=100, bg_value='white')
dpath = ub.Path('/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_6b952f74')
fpath = dpath / 'vali_result_heatmaps.jpg'
canvas = kwimage.imresize(canvas, max_dim=4096)
print(f'canvas.shape={canvas.shape}')
kwimage.imwrite(fpath, canvas)
