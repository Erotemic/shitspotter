import json
import ubelt as ub
from pathlib import Path
# from huggingface_hub import HfApi, HfFolder, Repository
# from kwcoco import CocoDataset
import webdataset as wds
from scriptconfig import DataConfig, Value
from PIL import Image
from io import BytesIO


class KwcocoToHFConfig(DataConfig):
    """
    Convert a KWCoco bundle (train/vali/test .kwcoco.zip files) to Hugging Face WebDataset format.
    """
    bundle_dir = Value('/data/joncrall/dvc-repos/shitspotter_dvc', help='Directory with train/vali/test .kwcoco.zip files')
    output_dir = Value('/data/joncrall/dvc-repos/shitspotter_dvc/webdataset_shards', help='Output dir for WebDataset .tar files')
    push_to_hub = Value(False, isflag=True, help='Push to Hugging Face hub (not implemented)')
    hf_repo = Value('erotemic/shitspotter', help='Optional HF repo (e.g. erotemic/shitspotter)')


def convert_dset(coco_fpath, out_tar):
    import kwcoco
    dset = kwcoco.CocoDataset(coco_fpath)
    print(f"[INFO] Loaded {coco_fpath}: {len(dset.images())} images")

    with wds.ShardWriter(str(out_tar)) as sink:
        for img in dset.images():
            img_id = img['id']
            img_fpath = dset.get_image_fpath(img_id)
            anns = dset.index.imgid_to_aids[img_id]
            annots = [dset.anns[aid] for aid in anns]
            key = f"{img_id:08d}"

            try:
                with Image.open(img_fpath) as im:
                    buf = BytesIO()
                    im.save(buf, format='JPEG')
                    img_bytes = buf.getvalue()
            except Exception as ex:
                print(f"[ERROR] Failed to load image {img_fpath}: {ex}")
                continue

            sample = {
                "__key__": key,
                "jpg": img_bytes,
                "json": json.dumps({
                    "id": img_id,
                    "file_name": img['file_name'],
                    "width": img['width'],
                    "height": img['height'],
                    "annotations": annots
                })
            }
            sink.write(sample)

    # print(f"[INFO] Saved {split}.tar")

    # # Save categories.json once
    # cats = dset.dataset.get("categories", [])
    # with open(output_dir / "categories.json", "w") as f:
    #     json.dump(cats, f, indent=2)


def main():
    config = KwcocoToHFConfig.cli()
    print(f'config = {ub.urepr(config, nl=1)}')

    import kwcoco
    bundle_dir = Path(config.bundle_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'vali', 'test']
    categories_saved = False

    for split in splits:
        coco_fpath = bundle_dir / f"{split}.kwcoco.zip"
        out_tar = output_dir / f"{split}.tar"
        convert_dset(coco_fpath, out_tar)

    print(f"[DONE] All splits converted. Output: {output_dir}")

    if config.push_to_hub:
        # Upload all to HF
        upload_to_hub(config.hf_repo, config.output_dir)

if __name__ == "__main__":
    KwcocoToHFConfig.main()


if __name__ == "__main__":
    main()
