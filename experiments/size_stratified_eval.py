#!/usr/bin/env python3
"""
Post-hoc size-stratified detection AP from saved eval predictions.

The kit's eval writes detect_metrics.json with only ``area_range=all``. This
re-scores the SAVED predictions with COCO small/medium/large area ranges — no
re-inference, no retrain — so we can test whether a resolution bump actually
helped *small* poops (EVALUATION_ROADMAP.md test #2).

It re-runs kwcoco's ``CocoEvaluator`` (the same scorer ``python -m kwcoco
eval`` uses inside the kit) over each model's ``true_bbox_only.kwcoco.zip`` +
``pred_boxes_bbox_only.kwcoco.zip``, at iou_thresh=0.5, for area ranges
[all, small, medium, large].

MUST run in the host kwcoco env (e.g. uvpy3.13.13), not the VM's bare python.

Usage:
    # default: compare v10 pico@416 vs v11 pico@640 baseline
    python experiments/size_stratified_eval.py

    # explicit candidate eval dirs (each must hold the *_bbox_only.kwcoco.zip)
    python experiments/size_stratified_eval.py \
        --candidate v10_pico416=/media/joncrall/flash1/kcd-ssd/v10/eval/deimv2_pico_416x416_multiscale_320_512 \
        --candidate v11_pico640=/media/joncrall/flash1/kcd-ssd/v11/baseline/eval/deimv2_pico_640x640_multiscale_512_768

    # or a single true/pred pair
    python experiments/size_stratified_eval.py --true <true.kwcoco.zip> --pred <pred.kwcoco.zip> --name mymodel
"""
import json
import sys
from pathlib import Path

AREA_RANGES = ["all", "small", "medium", "large"]
IOU_THRESH = 0.5

DEFAULT_CANDIDATES = {
    "v10_pico416": "/media/joncrall/flash1/kcd-ssd/v10/eval/deimv2_pico_416x416_multiscale_320_512",
    "v11_pico640": "/media/joncrall/flash1/kcd-ssd/v11/baseline/eval/deimv2_pico_640x640_multiscale_512_768",
}


def _resolve_pair(eval_dir: Path):
    """Find (true, pred) bbox-only kwcoco files under a candidate eval dir."""
    true = eval_dir / "true_bbox_only.kwcoco.zip"
    pred = eval_dir / "pred_boxes_bbox_only.kwcoco.zip"
    if not pred.exists():
        pred = eval_dir / "pred_boxes.kwcoco.zip"
    if not true.exists() or not pred.exists():
        raise FileNotFoundError(f"missing true/pred kwcoco under {eval_dir}")
    return true, pred


def _extract_ap(single_result):
    """Pull the bbox AP out of a kwcoco CocoSingleResult, defensively."""
    m = getattr(single_result, "nocls_measures", None)
    if m is None and isinstance(single_result, dict):
        m = single_result.get("nocls_measures")
    if m is None:
        return None
    try:
        return float(m["ap"])
    except Exception:
        return float(getattr(m, "ap", float("nan")))


def score_one(name, true_fpath, pred_fpath):
    from kwcoco.coco_evaluator import CocoEvaluator
    cfg = {
        "true_dataset": str(true_fpath),
        "pred_dataset": str(pred_fpath),
        "area_range": AREA_RANGES,
        "iou_thresh": [IOU_THRESH],
        "draw": False,
    }
    evaler = CocoEvaluator(cfg)
    results = evaler.evaluate()
    # results.results maps reskey -> CocoSingleResult; reskey looks like
    # 'area_range=small,iou_thresh=0.5'.
    by_area = {}
    res_map = getattr(results, "results", results)
    items = res_map.items() if hasattr(res_map, "items") else []
    for reskey, single in items:
        for ar in AREA_RANGES:
            if f"area_range={ar}," in (reskey + ","):
                by_area[ar] = _extract_ap(single)
    return {"name": name, "true": str(true_fpath), "pred": str(pred_fpath), "ap_by_area": by_area}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", action="append", default=[],
                   help="name=eval_dir; repeatable. Overrides the default v10-vs-v11 comparison.")
    p.add_argument("--true", help="single true kwcoco (with --pred, --name)")
    p.add_argument("--pred", help="single pred kwcoco")
    p.add_argument("--name", default="model")
    p.add_argument("--out", help="optional JSON output path")
    args = p.parse_args(argv)

    jobs = []
    if args.true and args.pred:
        jobs.append((args.name, Path(args.true), Path(args.pred)))
    elif args.candidate:
        for spec in args.candidate:
            name, _, d = spec.partition("=")
            jobs.append((name, *_resolve_pair(Path(d))))
    else:
        for name, d in DEFAULT_CANDIDATES.items():
            jobs.append((name, *_resolve_pair(Path(d))))

    rows = []
    for name, true_fpath, pred_fpath in jobs:
        print(f"[size-eval] scoring {name}: {pred_fpath.name}", file=sys.stderr)
        rows.append(score_one(name, true_fpath, pred_fpath))

    # Pretty table.
    hdr = f"{'model':<18} " + " ".join(f"AP_{a:<7}" for a in AREA_RANGES)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = []
        for a in AREA_RANGES:
            v = r["ap_by_area"].get(a)
            cells.append(f"{v:.4f} " if isinstance(v, float) else f"{'n/a':<7} ")
        print(f"{r['name']:<18} " + " ".join(c.rstrip().ljust(9) for c in cells))

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
