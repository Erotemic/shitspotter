#!/usr/bin/env python
"""Root-cause the v13 in-loop eval crash from a coco_eval_crash_dump_*.pkl.

The patched DEIM CocoEvaluator (coco_eval.py) dumps the EXACT offending batch
(raw predictions + prepared results + img_ids + faster_coco_eval version) when
the per-batch eval throws. This reads that pickle and tries to pinpoint WHY
loadRes/evaluate fell over -- without guessing.

Usage (host or container; needs faster_coco_eval only to re-run loadRes):

    python experiments/mobile_app_training_v13/analyze_eval_crash.py \
        /media/joncrall/flash1/kcd-ssd/v13b/repro_eval_crash/coco_eval_crash_dump_pid*.pkl
"""
import argparse
import glob
import math
import pickle
import sys


def _is_bad_number(x):
    try:
        return math.isnan(float(x)) or math.isinf(float(x))
    except (TypeError, ValueError):
        return True  # non-numeric where a number is expected


def describe_results(results):
    """Look for the usual loadRes landmines in the prepared detection list."""
    if not results:
        print("  results is empty/falsy ->", repr(results))
        return
    print(f"  n_results = {len(results)}")
    print("  first result:", results[0])
    bad_bbox, bad_score, bad_id, bad_imgid, weird_types = [], [], [], [], []
    for k, r in enumerate(results):
        bbox = r.get("bbox")
        if bbox is None or len(bbox) != 4 or any(_is_bad_number(v) for v in bbox):
            bad_bbox.append((k, bbox))
        if _is_bad_number(r.get("score", 0.0)):
            bad_score.append((k, r.get("score")))
        iid = r.get("image_id")
        if not isinstance(iid, (int,)) and not _looks_int(iid):
            bad_imgid.append((k, type(iid).__name__, iid))
        cid = r.get("category_id")
        if not isinstance(cid, (int,)) and not _looks_int(cid):
            bad_id.append((k, type(cid).__name__, cid))
        for key in ("bbox", "score", "image_id", "category_id"):
            v = r.get(key)
            if hasattr(v, "dtype") or type(v).__module__ not in ("builtins",):
                weird_types.append((k, key, type(v).__module__ + "." + type(v).__name__))
    for label, items in [("bad/NaN/inf bbox", bad_bbox), ("bad score", bad_score),
                         ("non-int category_id", bad_id), ("non-int image_id", bad_imgid),
                         ("non-builtin types (numpy/torch leak)", weird_types)]:
        if items:
            print(f"  !! {label}: {len(items)} e.g. {items[:5]}")
    if not any([bad_bbox, bad_score, bad_id, bad_imgid, weird_types]):
        print("  no obvious degenerate field found in results")


def _looks_int(x):
    try:
        int(x)
        return True
    except (TypeError, ValueError):
        return False


def try_reproduce(payload):
    """Re-run loadRes/evaluate on the dumped batch to confirm + localize."""
    try:
        from faster_coco_eval import COCO, COCOeval_faster
        import faster_coco_eval as fce
    except Exception as e:
        print("  cannot import faster_coco_eval here:", e)
        return
    print("  faster_coco_eval here =", getattr(fce, "__version__", "?"),
          "| dumped under =", payload.get("faster_coco_eval_version"))
    results = payload.get("results")
    # We don't have coco_gt in the dump (too big); the crash is in loadRes ->
    # loadAnns building the DT index, which only needs `results`. Build a tiny
    # COCO with just the referenced images/cat to localize the offending entry.
    img_ids = set(payload.get("img_ids") or [])
    cat_ids = set(r.get("category_id") for r in (results or []))
    gt = {
        "images": [{"id": i, "width": 99999, "height": 99999} for i in img_ids],
        "categories": [{"id": c, "name": str(c)} for c in cat_ids if c is not None],
        "annotations": [],
    }
    # Bisect: which single result makes loadRes throw?
    def loads(rs):
        c = COCO(gt)
        c.loadRes(rs)
    try:
        loads(results)
        print("  loadRes(results) did NOT throw here -- crash may need real coco_gt"
              " or a version diff (dumped vs local).")
        return
    except Exception as e:
        print("  reproduced loadRes failure:", type(e).__name__, e)
    for k, r in enumerate(results):
        try:
            loads([r])
        except Exception as e:
            print(f"  >> offending single result #{k}: {r}")
            print(f"     raises {type(e).__name__}: {e}")
            return
    print("  no single result reproduced it alone -> interaction across results"
          " (e.g. duplicate ids).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?", help="path or glob to coco_eval_crash_dump_*.pkl")
    args = ap.parse_args()

    pattern = args.dump or "/media/joncrall/flash1/kcd-ssd/v13b/repro_eval_crash/coco_eval_crash_dump_pid*.pkl"
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("no dump found at:", pattern); sys.exit(2)
    path = paths[-1]
    print("=== analyzing:", path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    print("exception :", payload.get("exception_type"), "-", payload.get("exception_str"))
    print("iou_type  :", payload.get("iou_type"))
    print("img_ids   :", (payload.get("img_ids") or [])[:10],
          f"(n={len(payload.get('img_ids') or [])})")
    print("\n--- prepared results ---")
    describe_results(payload.get("results"))
    print("\n--- reproduce + localize ---")
    try_reproduce(payload)
    print("\n--- original traceback ---")
    print(payload.get("traceback"))


if __name__ == "__main__":
    main()
