#!/usr/bin/env python
"""Thin ShitSpotter wrapper around KDK source-space truth-aware review."""
from __future__ import annotations

import argparse
from pathlib import Path

from driver import load_config, require_kdk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--true", dest="true_fpath", default=None)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--target-iou-thresh", type=float, default=0.50)
    args = parser.parse_args()

    config = load_config(args.config)
    require_kdk(config)
    from kwcoco_detector_kit.data.truth_review import (
        PredictionReviewConfig,
        build_prediction_review,
    )

    semantics = config["truth_semantics"]
    review_config = PredictionReviewConfig.cli(argv=False, data={
        "true": args.true_fpath or config["splits"]["train"],
        "pred": args.pred,
        "dst_dpath": args.dst,
        "target_categories": ",".join(semantics["target_categories"]),
        "ignore_categories": ",".join(semantics["ignore_categories"]),
        "uncategorized_annotation_policy": semantics["uncategorized_annotation_policy"],
        "default_non_target_policy": semantics["default_non_target_policy"],
        "unclassified_category_policy": semantics["unclassified_category_policy"],
        "min_score": args.min_score,
        "top_n": args.top_n,
        "target_iou_thresh": args.target_iou_thresh,
    })
    build_prediction_review(review_config)


if __name__ == "__main__":
    main()
