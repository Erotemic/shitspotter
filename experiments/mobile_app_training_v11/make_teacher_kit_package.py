#!/usr/bin/env python3
"""
Repackage the foundation_detseg_v3 OGDino detector into a KIT-NATIVE package
that `kwcoco-detector-kit pseudo-label` / `predict_kwcoco` can consume.

WHY: the v9/v11 distill path pointed `pseudo-label` at a foundation-format
package (`backend: opengroundingdino_sam2`, `detector.config_fpath/…`), but
`predict_kwcoco` only understands the kit-native format — it reads
`manifest["trainer"]`, `materialize_workdir(manifest["artifacts"])`, then
`OGDinoTrainer.build_predictor(workdir)` which expects
`generated_configs/ogdino_cfg.py` + `checkpoint*.pth` + `policy.json`. The two
formats were never reconciled because v9 never actually ran (KeyError('trainer')).

This shim builds the kit-native layout from the v3 pipeline's
`selected_detector_checkpoint.yaml`:

    <out_pkg>/
      package.yaml                      # trainer: opengroundingdino, artifacts, ...
      checkpoint0000.pth -> <ckpt>      # symlink; materialize copy2's the content
      generated_configs/ogdino_cfg.py -> <cfg>
      policy.json                       # {"label_list": ["poop"]}

Detector-only (no SAM2): the student is a box detector, so pipeline=detector_only.

⚠️ UNVERIFIED on the VM (no GPU/OGDino here). Run on the host. Risks to watch:
  - the OGDino config_fpath must be loadable by groundingdino `load_model`
    (it is what the v3 pipeline trained with, so it should be);
  - `$KCD_OPENGROUNDINGDINO_REPO_DPATH` must point at the OGDino repo when
    `pseudo-label` runs (the predictor imports groundingdino lazily);
  - the label text prompt is `label_list` joined → "poop ." .

Usage:
    python make_teacher_kit_package.py \
        --selected_yaml /data/joncrall/dvc-repos/shitspotter_expt_dvc/foundation_detseg_v3/v9/selected_detector_checkpoint.yaml \
        --out_pkg /media/joncrall/flash1/kcd-ssd/v11/data/teacher_kit_package \
        --label poop
"""
import argparse
import json
import os
from pathlib import Path

import yaml


def _link_or_copy(src: Path, dst: Path):
    """Prefer a symlink (cheap for big checkpoints); fall back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected_yaml", required=True,
                   help="v3 pipeline selected_detector_checkpoint.yaml")
    p.add_argument("--out_pkg", required=True, help="output kit package directory")
    p.add_argument("--label", default="poop", help="single class / text prompt (default: poop)")
    p.add_argument("--score_thresh", type=float, default=0.30)
    p.add_argument("--nms_iou_thresh", type=float, default=0.50)
    args = p.parse_args(argv)

    sel = yaml.safe_load(Path(args.selected_yaml).read_text())
    cfg_src = Path(sel["detector_config_fpath"]).expanduser()
    ckpt_src = Path(sel["selected_detector_checkpoint_fpath"]).expanduser()
    for f in (cfg_src, ckpt_src):
        if not f.exists():
            raise FileNotFoundError(f"referenced artifact missing: {f}")

    out = Path(args.out_pkg).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # Lay out the kit-native workdir-able package.
    _link_or_copy(cfg_src, out / "generated_configs" / "ogdino_cfg.py")
    _link_or_copy(ckpt_src, out / "checkpoint0000.pth")
    (out / "policy.json").write_text(json.dumps({"label_list": [args.label]}, indent=2))

    manifest = {
        "format_version": 1,
        "trainer": "opengroundingdino",
        "pipeline": "detector_only",
        "category_names": [args.label],
        "artifacts": {
            "checkpoint": "checkpoint0000.pth",
            "train_config": "generated_configs/ogdino_cfg.py",
            "policy": "policy.json",
        },
        "postprocess": {
            "score_thresh": float(args.score_thresh),
            "nms_iou_thresh": float(args.nms_iou_thresh),
        },
        "metadata": {
            "name": "v11_ogdino_teacher_kitpkg",
            "source_selected_yaml": str(args.selected_yaml),
            "detector_test_simplified_ap": sel.get("detector_test_simplified_ap", ""),
            "note": "Repackaged foundation OGDino detector into kit-native format "
                    "for pseudo-label/predict_kwcoco (bbox-only teacher).",
        },
    }
    (out / "package.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"wrote kit-native OGDino teacher package -> {out}")
    print(f"  config:     {cfg_src}")
    print(f"  checkpoint: {ckpt_src}")
    print(f"  label/prompt: {args.label!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
