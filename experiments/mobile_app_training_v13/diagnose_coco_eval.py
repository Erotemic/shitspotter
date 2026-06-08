#!/usr/bin/env python
"""Ground-truth diagnostic for the v13 in-loop eval crash.

The relaunch died in DEIM's per-batch eval with::

    File ".../faster_coco_eval/core/coco.py", line 278, in loadAnns
        return [self.anns[i] for i in ids]
    TypeError: 'int' object does not support the context manager protocol

A clean faster_coco_eval builds ``self.anns`` as a plain dict, whose subscript
cannot run a ``with`` statement — so this is NOT a simple "1.7.x is broken"
version bump. This script prints what the CONTAINER actually has so we stop
guessing: the installed version/file, and whether ``cocoDt.anns`` is really a
dict in this environment (or some object whose __getitem__ does ``with <int>``).

Run inside the image the failing run used:

    reproduce/in_docker.sh python experiments/mobile_app_training_v13/diagnose_coco_eval.py
"""
import traceback


def main():
    import faster_coco_eval as f
    print("faster_coco_eval VERSION:", getattr(f, "__version__", "<none>"))
    print("faster_coco_eval FILE   :", f.__file__)

    from faster_coco_eval import COCO, COCOeval_faster
    from faster_coco_eval.core import coco as coco_mod
    print("coco.py FILE            :", coco_mod.__file__)

    # Is loadAnns line 278 the `self.anns[i]` listcomp here too?
    import inspect
    src = inspect.getsource(COCO.loadAnns)
    print("\nloadAnns source in this env:\n" + src)

    gt = {
        "images": [{"id": 1, "width": 64, "height": 64, "file_name": "x.jpg"}],
        "categories": [{"id": 1, "name": "poop"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [8, 8, 16, 16], "area": 256, "iscrowd": 0}],
    }
    c = COCO(gt)
    dt = c.loadRes([{"image_id": 1, "category_id": 1,
                     "bbox": [8, 8, 16, 16], "score": 0.9}])
    print("type(coco_gt.anns):", type(c.anns).__name__)
    print("type(cocoDt.anns) :", type(dt.anns).__name__)

    ce = COCOeval_faster(c, iouType="bbox",
                         print_function=lambda *a, **k: None, separate_eval=True)
    ce.cocoDt = dt
    ce.params.imgIds = [1]
    try:
        ce.evaluate()
        print("\nevaluate(): OK on synthetic input -> the crash is data-shaped, "
              "not a blanket version break. Capture the real cocoDt at failure.")
    except Exception as e:
        print("\nevaluate() RAISED:", type(e).__name__, e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
