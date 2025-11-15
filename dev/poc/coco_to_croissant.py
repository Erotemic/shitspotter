# pip install mlcroissant

#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CocoToCroissantCLI(scfg.DataConfig):
    src = scfg.Value(None, help='path to coco file')

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Ignore:
            import shitspotter
            fpath = shitspotter.util.find_shit_coco_fpath()

        Example:
            >>> # xdoctest: +SKIP
            >>> from coco_to_croissant import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict()
            >>> cls = CocoToCroissantCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        import kwcoco
        dset = kwcoco.CocoDataset.coerce(fpath)


def build_croissant_metadata(dset, output_path="dataset_description.json"):
    import mlcroissant as mlc
    import json
    from pathlib import Path
    from mlcroissant import Dataset
    from mlcroissant import FileObject, Field

    coco = dset.dataset
    coco_json_path = dset.fpath

    # Basic dataset info
    dataset_name = coco.get("info", {}).get("description", "COCO-style Dataset")
    dataset_description = coco.get("info", {}).get("description", "Converted from COCO format")
    dataset_url = coco.get("info", {}).get("url", "")
    license_info = coco.get("licenses", [{}])[0]

    info = coco.get("info", {})
    licenses = coco.get("licenses", [{}])

    # References:
    # https://github.com/mlcommons/croissant/blob/main/python/mlcroissant/recipes/bounding-boxes.ipynb

    dataset = Dataset(
        name=info.get("description", "COCO-style Dataset"),
        description=info.get("description", "COCO-style dataset converted to Croissant format"),
        version=info.get("version", "1.0"),
        citation=licenses[0].get("name", ""),
        data=[
            FileObject(
                name="coco_annotations",
                source=JsonSource(url=str(coco_json_path)),
                fields=[
                    Field(name="image_id", path=mlc.Source("$.images[*].id")),
                    Field(name="file_name", path=JsonPath("$.images[*].file_name")),
                    Field(name="width", path=JsonPath("$.images[*].width")),
                    Field(name="height", path=JsonPath("$.images[*].height")),
                    Field(name="annotation_id", path=JsonPath("$.annotations[*].id")),
                    Field(name="annotation_image_id", path=JsonPath("$.annotations[*].image_id")),
                    Field(name="category_id", path=JsonPath("$.annotations[*].category_id")),
                    Field(name="bbox", path=JsonPath("$.annotations[*].bbox")),
                    Field(name="segmentation", path=JsonPath("$.annotations[*].segmentation")),
                    Field(name="category_name", path=JsonPath("$.categories[*].name")),
                    Field(name="category_supercategory", path=JsonPath("$.categories[*].supercategory")),
                ]
            )
        ]
    )

    dataset = Dataset(
        name=dataset_name,
        description=dataset_description,
        url=dataset_url or None,
        citation=license_info.get("name", ""),
        version=coco.get("info", {}).get("version", "1.0"),
        keywords=["coco", "images", "annotations"],
        data=[FileObject(
            name="coco_annotations",
            source=JsonLSource(
                url=coco_json_path,
            ),
            fields=fields
        )]
    )

    # Write to disk
    with open(output_path, "w") as out:
        json.dump(dataset.model_dump(mode="json"), out, indent=2)
    print(f"Wrote Croissant metadata to: {output_path}")


__cli__ = CocoToCroissantCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/dev/poc/coco_to_croissant.py
        python -m coco_to_croissant
    """
    __cli__.main()
