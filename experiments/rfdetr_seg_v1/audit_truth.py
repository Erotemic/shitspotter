#!/usr/bin/env python
"""Write a reproducible category census and paths for semantic edge cases."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from driver import load_config, sha256_file


def main():
    import kwcoco

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--src", default=None, help="defaults to configured train split")
    parser.add_argument("--dst", default=None, help="JSON output; defaults beside this script")
    args = parser.parse_args()
    config = load_config(args.config)
    src = Path(args.src or config["splits"]["train"]).resolve()
    dst = Path(args.dst or Path(__file__).with_name("dataset_category_census.json")).resolve()
    dset = kwcoco.CocoDataset.coerce(str(src))

    counts = Counter()
    edge = defaultdict(list)
    inspect_names = {"unknown", "ignore", "unkown", "residue", "residual", "background"}
    for ann in dset.annots().objs:
        cid = ann.get("category_id")
        cat = dset.cats.get(cid) if cid is not None else None
        name = cat.get("name") if cat else None
        key = name if name is not None else "__uncategorized__"
        counts[key] += 1
        if name in inspect_names or name is None:
            gid = ann["image_id"]
            source = Path(dset.get_image_fpath(gid)).resolve()
            edge[key].append({
                "annotation_id": ann.get("id"),
                "image_id": gid,
                "source_image": str(source),
                "labelme_json": str(source.with_suffix(".json")),
                "labelme_json_exists": source.with_suffix(".json").is_file(),
                "bbox": ann.get("bbox"),
                "has_segmentation": ann.get("segmentation") is not None,
            })

    report = {
        "schema": "shitspotter.rfdetr.truth_census.v1",
        "source_kwcoco": str(src),
        "source_sha256": sha256_file(src),
        "totals": {
            "images": dset.n_images,
            "annotations": dset.n_annots,
            "declared_categories": len(dset.cats),
            "images_with_annotations": len({ann["image_id"] for ann in dset.annots().objs}),
            "images_without_annotations": dset.n_images - len({ann["image_id"] for ann in dset.annots().objs}),
        },
        "category_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "truth_semantics": config["truth_semantics"],
        "semantic_edge_cases": dict(edge),
    }
    dst.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(dst)


if __name__ == "__main__":
    main()
