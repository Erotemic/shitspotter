#!/usr/bin/env python3
# /// script
# dependencies = [
#   "Pillow",
#   "huggingface_hub",
#   "kwcoco",
#   "kwutil",
#   "scriptconfig",
#   "ubelt",
#   "webdataset",
# ]
# requires-python = ">=3.11"
# ///
r"""
Convert a KWCoco dataset with train/vali/test splits to Hugging Face WebDataset format.

Example usage (locally):

    python kwcoco_to_hf_webdataset.py \
        --bundle_dir /data/joncrall/dvc-repos/shitspotter_dvc \
        --output_dir /data/joncrall/dvc-repos/shitspotter_dvc/webdataset_shards \
        --hf_repo erotemic/shitspotter

References:
    https://huggingface.co/datasets/erotemic/shitspotter
    https://discuss.huggingface.co/t/help-making-object-detection-dataset/152344
    https://discuss.huggingface.co/t/generating-croissant-metadata-for-custom-image-dataset/150255
    https://chatgpt.com/c/680be71a-4a0c-8002-a31e-bd9c17b5ac05

Example:
    >>> # Demo of full conversion
    >>> from kwcoco_to_hf_webdataset import *  # NOQA
    >>> import ubelt as ub
    >>> import kwcoco
    >>> dpath = ub.Path.appdir('kwcoco/demo/hf-convert').ensuredir()
    >>> full_dset = kwcoco.CocoDataset.demo('shapes32')
    >>> full_dset.reroot(absolute=True)
    >>> # Create splits
    >>> split_names = ['train', 'validation', 'test']
    >>> imgid_chunks = list(ub.chunks(full_dset.images(), nchunks=3))
    >>> for split_name, gids in zip(split_names, imgid_chunks):
    >>>     sub_dset.fpath = dpath / (split_name + '.kwcoco.zip')
    >>>     sub_dset.dump()
    >>> # Call conversion script
    >>> config = KwcocoToHFConfig(
    >>>     bundle_dir=dpath,
    >>>     output_dir=dpath / 'webds',
    >>>     hf_repo=None,
    >>>     #hf_repo='erotemic/shapes',
    >>> )
    >>> KwcocoToHFConfig.main(argv=False, **config)
    >>> # Test conversion can be read by a torch dataloader
    >>> check_webdataset_as_torch(dpath / 'webds/train/*.tar')
    >>> # xdoctest: +REQUIRES(--upload)
    >>> # Test upload
    >>> hf_repo = 'erotemic/shapes'
    >>> upload_to_hub(hf_repo, config.bundle_dir, config.output_dir)
"""

import json
import kwcoco
import kwutil
import os
import ubelt as ub
import webdataset
from PIL import Image
from huggingface_hub import HfApi, upload_file
from io import BytesIO
import scriptconfig as scfg


class KwcocoToHFConfig(scfg.DataConfig):
    """
    Convert a KWCoco bundle (train/vali/test .kwcoco.zip files) to Hugging Face WebDataset format.
    """

    bundle_dir = scfg.Value(
        None,
        help=ub.paragraph(
            """
            Directory with train/vali/test .kwcoco.zip files
            """
        ),
    )
    output_dir = scfg.Value(
        None,
        help=ub.paragraph(
            """
            Output dir for WebDataset .tar files
            """
        ),
    )
    hf_repo = scfg.Value(
        None,
        help=ub.paragraph(
            """
            If specified, push to this huggingface repo.
            (e.g. erotemic/shitspotter)
            """
        ),
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        bundle_dir = ub.Path(config.bundle_dir)
        output_dir = ub.Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = ["train", "validation", "test"]
        categories_out = output_dir / "categories.json"

        for split in splits:
            coco_fpath = bundle_dir / f"{split}.kwcoco.zip"
            out_tar = output_dir / f"{split}.tar"
            if not coco_fpath.exists():
                raise Exception(f"Missing {split} split at {coco_fpath}")

        for split in splits:
            coco_fpath = bundle_dir / f"{split}.kwcoco.zip"
            out_tar = output_dir / f"{split}/{split}-%06d.tar"
            categories_out = output_dir / f"{split}_categories.json"
            convert_coco_to_webdataset(coco_fpath, out_tar, categories_out)

        if config.hf_repo is not None:
            hf_repo = config.hf_repo
            upload_to_hub(hf_repo, bundle_dir, output_dir)


def check_webdataset_as_torch(urls):
    """
    Args:
        urls (str):
            glob pattern matching the tar files or directory containing them.
    """
    # Once converted, test that we can use a pytorch dataloader:
    import webdataset as wds
    import torch
    from torchvision.transforms import ToTensor
    import kwutil

    urls = list(map(os.fspath, kwutil.util_path.coerce_patterned_paths(urls, expected_extension='.tar')))
    print(f'urls = {ub.urepr(urls, nl=1)}')
    assert urls

    # decode to PIL, then map PIL→Tensor
    dset = (
        wds.WebDataset(urls)
        .decode("pil")
        .to_tuple("jpg", "json")
        .map_tuple(ToTensor(), lambda meta: meta)
    )
    loader = torch.utils.data.DataLoader(dset.batched(2))
    for imgs, metas in loader:
        # imgs is a list of torch.Tensors, metas is a list of dicts
        print(imgs[0].shape, metas[0])
        break


def convert_coco_to_webdataset(coco_dset, out_tar, categories_out=None):
    """
    Convert a coco dataset to a webdataset suitable for huggingface.

    Args:
        coco_dset (str | PathLike | CocoDataset):
            path to the coco dataset or the coco datset itself.

        out_tar (str | PathLike): this is the patterned path
            to write sharded tar files to.

        categories_out (str | PathLike | None):
            if True, write out the category json file to this path

    Example:
        >>> from kwcoco_to_hf_webdataset import *  # NOQA
        >>> import ubelt as ub
        >>> import kwcoco
        >>> dpath = ub.Path.appdir('kwcoco/test/hf-convert').ensuredir()
        >>> coco_dset = kwcoco.CocoDataset.demo('shapes8')
        >>> out_tar = dpath / f"test_wds/test-wds-%06d.tar"
        >>> categories_out = dpath / f"test_wds_categories.json"
        >>> urls = written_files = convert_coco_to_webdataset(coco_dset, out_tar, categories_out)
        >>> check_webdataset_as_torch(urls)
    """
    dset = kwcoco.CocoDataset.coerce(coco_dset)
    print(f"[INFO] Loaded {coco_dset}")

    if categories_out and not categories_out.exists():
        cats = dset.dataset.get("categories", [])
        categories_out.write_text(json.dumps(cats, indent=2))
        print(f"[INFO] Wrote categories.json with {len(cats)} categories")

    ub.Path(out_tar).parent.ensuredir()
    sink = webdataset.ShardWriter(pattern=str(out_tar), maxcount=1000)

    dset.conform(legacy=True)

    written_files = ub.oset()

    pman = kwutil.ProgressManager()
    with pman:
        coco_images = dset.images().coco_images
        prog_iter = pman.progiter(coco_images, desc=f"Processing {dset.tag}")
        for coco_img in prog_iter:
            image_id = coco_img.img["id"]
            img_path = coco_img.image_filepath()
            img_pil = Image.open(img_path).convert("RGB")

            # Save image to bytes
            img_bytes = BytesIO()
            img_pil.save(img_bytes, format="jpeg")
            img_bytes = img_bytes.getvalue()

            # Convert annots to basic JSON-serializable format

            # Attempt to make dataset object detection ready.
            # https://huggingface.co/docs/datasets/v2.14.5/en/object_detection
            objects = {
                "area": [],
                "bbox": [],
                "category": [],
                "id": [],
            }
            for ann in coco_img.annots().objs:
                objects["area"].append(int(ann["area"]))
                objects["bbox"].append(ann["bbox"])
                objects["category"].append(ann["category_id"])
                objects["id"].append(ann["id"])

            anns = []
            for ann in coco_img.annots().objs:
                anns.append(
                    {
                        "bbox": ann["bbox"],
                        "category_id": ann["category_id"],
                        "segmentation": ann.get("segmentation", None),
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )

            # Save JSON metadata
            sample = {
                "__key__": str(image_id),
                "jpg": img_bytes,
                # "image_id": image_id,
                # "width": coco_img.img["width"],
                # "height": coco_img.img["height"],
                # "objects": objects,
                "json": json.dumps(
                    {
                        "id": image_id,
                        "image_id": image_id,
                        "file_name": os.path.basename(img_path),
                        "width": coco_img.img["width"],
                        "height": coco_img.img["height"],
                        "objects": objects,
                        "annotations": anns,
                    }
                ),
            }

            sink.write(sample)
            written_files.append(sink.fname)

    sink.close()
    written_files = list(written_files)
    print(f"Saved {written_files}")
    return written_files


def upload_to_hub(hf_repo, bundle_dir, output_dir):
    api = HfApi()  # NOQA
    output_dir = ub.Path(output_dir)

    for file in output_dir.glob("*/**.tar"):
        print(f"[UPLOAD] Uploading {file.name} to {hf_repo}")
        upload_file(
            path_or_fileobj=str(file),
            path_in_repo=str(file.relative_to(bundle_dir)),
            repo_id=hf_repo,
            repo_type="dataset",
        )
    for categories_file in output_dir.glob("*categories.json"):
        upload_file(
            path_or_fileobj=str(categories_file),
            path_in_repo=str(categories_file.relative_to(bundle_dir)),
            repo_id=hf_repo,
            repo_type="dataset",
        )


if __name__ == "__main__":
    KwcocoToHFConfig.main()
