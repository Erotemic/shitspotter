import kwcoco
from collections import Counter
import ubelt as ub
import shitspotter
import kwplot
kwplot.autompl()

dset = shitspotter.open_shit_coco()
test_dset = kwcoco.CocoDataset(dset.fpath.parent / 'test.kwcoco.zip')
# Test set is distinct.
assert not set(dset.images().lookup('file_name')) & set(test_dset.images().lookup('file_name'))


def quick_stats():
    combo_dset = kwcoco.CocoDataset.union(dset, test_dset)
    poop_cid = combo_dset.index.name_to_cat['poop']['id']
    aids_with_poop = [ann['id'] for ann in combo_dset.annots().objs if ann.get('category_id') == poop_cid]
    gids_with_poop = sorted(set(combo_dset.annots(aids_with_poop).lookup('image_id')))

    stat_table = []
    stat_table.append({
        'index': 'Total',
        '# Images': combo_dset.n_images,
        '# Annots': combo_dset.n_annots,
    })
    stat_table.append({
        'index': 'With Poop',
        '# Images': len(gids_with_poop),
        '# Annots': len(aids_with_poop),
    })
    import pandas as pd
    df = pd.DataFrame(stat_table)
    print(df.to_string())


def draw_tagged_images():
    print(dset.annots().attribute_frequency())

    taghist = Counter()

    tag_to_imgs = ub.ddict(list)

    for ann in dset.annots().objs:
        desc = ann.get('description', None)
        if desc:
            tags = [t.strip().lower() for t in desc.split(';') if t.strip()]
            print(f'tags = {ub.urepr(tags, nl=1)}')
            taghist.update(tags)

            for t in tags:
                tag_to_imgs[t].append(ann['image_id'])

    print(f'taghist={taghist}')

    tags_of_interest = [
        'hidden',
        'camoflauged',
        'difficult',
        'hard',
        'leaf clutter',
        'confusor',
        'hardcase',
        'hard-to-see',
        'hard-to-see-case',
    ]

    outpath = ub.Path('~/code/shitspotter/papers/wacv_2026/presentation/tagged').expand()

    ordered = sorted(dset.images().objs, key=lambda x: x['datetime'])
    ordered_gids = [img['id'] for img in ordered]

    for tag in ub.ProgIter(tags_of_interest, desc='draw tags'):
        print(f'tag={tag}')
        gids = set(tag_to_imgs[tag])
        print(f'gids={gids}')
        dpath = (outpath / tag).ensuredir()

        for base_image_id in gids:
            print(f'base_image_id={base_image_id}')
            order_idx = ordered_gids.index(base_image_id)
            for offset in [0, 1, 2]:
                image_id = ordered_gids[order_idx + offset]
                print(f'image_id={image_id}')
                coco_img = dset.coco_image(image_id)
                delayed = coco_img.imdelay()
                imgonly_imdata = delayed.finalize()

                annots = coco_img.annots()
                flags = [cid is not None for cid in annots.lookup('category_id')]
                annots = annots.compress(flags)
                flags = [n == 'poop' for n in annots.category_names]
                annots = annots.compress(flags)
                imgann_imdata = annots.detections.draw_on(imgonly_imdata.copy())

                fpath1 = dpath / (coco_img.name + '_img.jpg')
                fpath2 = dpath / (coco_img.name + '_ann.jpg')

                import kwimage
                kwimage.imwrite(fpath1, imgonly_imdata)
                kwimage.imwrite(fpath2, imgann_imdata)


def draw_polygon_image():
    poop_cid = dset.index.name_to_cat['poop']['id']
    aids_with_poop = [ann['id'] for ann in dset.annots().objs if ann.get('category_id') == poop_cid]
    annots = dset.annots(aids_with_poop)
    import numpy as np
    idxs = np.argsort(annots.detections.data['boxes'].area.ravel()).ravel()
    annot_id = annots[idxs[-200]]
    image_id = dset.annots([annot_id]).objs[0]['image_id']
    coco_img = dset.coco_image(image_id)

    imdata_img = coco_img.imdelay().finalize()

    annots = coco_img.annots()
    imgdata_ann = annots.detections.draw_on(imdata_img.copy())
    kwplot.imshow(imdata_img)
    kwplot.imshow(imgdata_ann)




quick_stats()
draw_tagged_images()
