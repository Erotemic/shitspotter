#!/usr/bin/env python
"""Build a ranked truth-QA queue from one completed mining round.

This is intentionally separate from mining/finalization: reviewing hard negatives
is a human gate before they are trusted as training negatives for the next round.
The generated review artifacts never mutate canonical truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from driver import load_config, require_kdk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--per-source", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    require_kdk(config)
    from kwcoco_detector_kit.data.review_mine import ReviewMineConfig, run

    root = Path(config["paths"]["output_root"])
    round_index = int(args.round_index)
    mining_dir = root / "rounds" / f"round{round_index}" / "mining"
    candidate_index = root / "candidates" / "train_negative_candidates"
    num_shards = int(config["mining"]["num_shards"])
    ledgers = [mining_dir / f"rank{rank}.mine_ledger.json" for rank in range(num_shards)]
    missing = [path for path in ledgers if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "mining must complete before review; missing ledgers: "
            + ", ".join(map(str, missing))
        )

    hard_negatives = mining_dir / "hard_negatives.kwcoco.zip"
    selected_candidates = hard_negatives.with_suffix(".selected_candidates.json")
    if not selected_candidates.is_file():
        raise FileNotFoundError(
            f"mine-finalize must produce {selected_candidates} before truth review"
        )

    policy = config.get("mining_review", {})
    review_dpath = mining_dir / "review"
    review_cfg = ReviewMineConfig.cli(argv=False, data={
        "candidate_index": str(candidate_index),
        "ledgers": [str(path) for path in ledgers],
        "selected_candidates": str(selected_candidates),
        "dst_dpath": str(review_dpath),
        "top_n": int(args.top_n if args.top_n is not None else policy.get("top_n", 200)),
        "per_source": int(args.per_source if args.per_source is not None else policy.get("per_source", 3)),
        "min_score": float(args.min_score if args.min_score is not None else policy.get("min_score", config["mining"]["score_thresh"])),
        "make_previews": not args.no_previews,
        "preview_max_dim": int(policy.get("preview_max_dim", 1000)),
        "context_margin": float(policy.get("context_margin", 0.20)),
    })
    queue_path = Path(run(review_cfg))

    # ShitSpotter's canonical manual annotations are LabelMe JSON sidecars next
    # to source images.  Write a compact annotation-target table even when the
    # sidecar does not exist yet: a missing sidecar can itself be the correction
    # needed for a false-negative source image.
    queue_doc = json.loads(queue_path.read_text())
    target_tsv = review_dpath / "annotation_targets.tsv"
    fields = [
        "rank", "max_score", "source_gid", "source_image", "labelme_json",
        "labelme_json_exists", "tile_id", "top_bbox_xyxy_in_source",
        "review_status", "review_note",
    ]
    with target_tsv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for item in queue_doc["items"]:
            source_image = Path(item["source_fpath"])
            labelme_json = source_image.with_suffix(".json")
            writer.writerow({
                "rank": item["rank"],
                "max_score": item["max_score"],
                "source_gid": item["source_gid"],
                "source_image": str(source_image),
                "labelme_json": str(labelme_json),
                "labelme_json_exists": labelme_json.is_file(),
                "tile_id": item["tile_id"],
                "top_bbox_xyxy_in_source": json.dumps(item.get("top_bbox_xyxy_in_source")),
                "review_status": item.get("review_status", "unreviewed"),
                "review_note": item.get("review_note", ""),
            })

    print()
    print("ShitSpotter truth-review gate")
    print(f"  browser: file://{review_dpath / 'index.html'}")
    print(f"  queue:   {review_dpath / 'review_queue.tsv'}")
    print(f"  targets: {target_tsv}")
    print(f"  kdoc:    {review_dpath / 'review.kwcoco.zip'}")
    print("If a hard negative is actually poop, edit/create the listed canonical")
    print("LabelMe sidecar, regenerate the source KWCoco manifest, and invalidate/rebuild")
    print("the candidate index and any pools derived from the old truth before round 1.")


if __name__ == "__main__":
    main()
