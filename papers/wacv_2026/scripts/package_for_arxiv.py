#!/usr/bin/env python3
"""
package_for_arxiv.py

Minimal arXiv submission packager for WACV 2026 paper, with aggressive image shrinking
while keeping *exactly one* smallest staged variant per referenced figure.

Default workflow command (yours):
  python scripts/package_for_arxiv.py --oxipng --png-to-jpg --png-jpg-mode ffmpeg \
    --jpeg-quality 80 --verbose --out-dir _arxiv_submission_compress5

Key features
------------
- Only copies reachable TeX (via \\input/\\include), local TeX assets in paper root, and referenced figures.
- PNG handling:
    * If PNG > --png-threshold-mb: consider:
        - oxipng candidate
        - ffmpeg JPG candidate at high-quality settings (q:v 2, yuvj444p)
      choose smallest, keep only winner.
    * If PNG <= threshold: try oxipng and keep only if smaller.
- JPEG handling:
    * If JPEG > --jpeg-threshold-mb and --recompress-jpegs: try Pillow recompress to --jpeg-quality and keep if smaller.
- EXTRA RULE (requested):
    After PNG->JPG conversion, if resulting JPG is still > jpeg threshold,
    re-run PNG->JPG conversion using the *target* JPEG compression settings
    (Pillow quality = --jpeg-quality), and keep smallest among all JPG candidates.
- Parallelism:
    Uses ThreadPoolExecutor to run per-figure transformation work in parallel.
    TeX rewrite happens after all chosen outputs are known.
- Reporting:
    Output table sorted by NEW size descending (largest offenders first).

Dependencies
------------
- oxipng (recommended)
- ffmpeg (recommended, for PNG->JPG)
- Pillow (optional; required for JPEG recompress and for PNG->JPG "target quality" fallback)

Example:
python scripts/package_for_arxiv.py --oxipng --png-to-jpg --png-jpg-mode ffmpeg --jpeg-quality 80 --verbose --out-dir _arxiv_submission_compress6 --jobs 2
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# -------------------------
# Regex helpers
# -------------------------

TEX_COMMENT_RE = re.compile(r"(?<!\\)%.*$")
TEX_INPUT_RE = re.compile(r"""\\(?:input|include)\s*\{\s*([^\{\}]+?)\s*\}""")
TEX_GRAPHICS_RE = re.compile(r"""\\includegraphics(?:\s*\[[^\]]*\])?\s*\{\s*([^\{\}]+?)\s*\}""")
TEX_BIB_RE = re.compile(r"""\\bibliography\s*\{\s*([^\{\}]+?)\s*\}""")

COMMON_GRAPHIC_EXTS = [".pdf", ".png", ".jpg", ".jpeg", ".eps"]


# -------------------------
# Config / records
# -------------------------

@dataclass(frozen=True)
class BuildPlan:
    paper_dir: Path
    entry_tex: Path
    out_dir: Path
    make_zip: bool
    zip_path: Optional[Path]
    dry_run: bool
    verbose: bool
    jobs: int

    # PNG policy
    oxipng: bool
    oxipng_level: int
    oxipng_extra: bool

    png_to_jpg: bool
    png_threshold_bytes: int
    png_jpg_mode: str          # auto|ffmpeg|pillow
    ffmpeg_qv: int
    ffmpeg_pix_fmt: str
    pillow_png_to_jpg_quality: int  # "high quality" baseline if pillow is used directly

    # JPEG policy
    recompress_jpegs: bool
    jpeg_threshold_bytes: int
    jpeg_quality: int  # DEFAULT 80


@dataclass
class TransformRecord:
    kind: str
    source_rel: Path
    chosen_rel: Path
    old_bytes: int
    new_bytes: int
    note: str


# -------------------------
# Small utils
# -------------------------

def strip_tex_comments(text: str) -> str:
    return "\n".join(TEX_COMMENT_RE.sub("", line) for line in text.splitlines())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.1f} {u}" if u != "B" else f"{int(v)} {u}"
        v /= 1024
    return f"{n} B"


def pct_saved(old: int, new: int) -> float:
    if old <= 0:
        return 0.0
    return 100.0 * (old - new) / old


def safe_copy2(src: Path, dst: Path, verbose: bool, dry_run: bool) -> None:
    if verbose:
        print(f"[copy] {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_manifest(out_dir: Path, relpaths: List[Path], dry_run: bool) -> None:
    manifest = out_dir / "MANIFEST.txt"
    lines = ["# Files included in arXiv bundle (relative paths)\n"]
    lines += [p.as_posix() + "\n" for p in sorted(set(relpaths))]
    if dry_run:
        return
    manifest.write_text("".join(lines), encoding="utf-8")


def make_zip_from_dir(out_dir: Path, zip_path: Path, dry_run: bool, verbose: bool) -> None:
    if verbose:
        print(f"[zip] creating: {zip_path}")
    if dry_run:
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = str(zip_path.with_suffix(""))
    shutil.make_archive(base_name, "zip", root_dir=out_dir)


def print_transform_table(records: List[TransformRecord]) -> None:
    if not records:
        print("\nNo transforms applied.")
        return

    # Largest offenders first (sorted by new size, descending)
    recs = sorted(records, key=lambda r: r.new_bytes, reverse=True)

    print("\nImage size report (sorted by NEW size; biggest offenders first):")
    header = f"{'kind':>18}  {'old':>10}  {'new':>10}  {'saved':>10}  {'%':>6}  {'chosen staged path':<70}  note"
    print(header)
    print("-" * len(header))
    for r in recs:
        saved = r.old_bytes - r.new_bytes
        pct = pct_saved(r.old_bytes, r.new_bytes)
        print(
            f"{r.kind:>18}  "
            f"{human_bytes(r.old_bytes):>10}  "
            f"{human_bytes(r.new_bytes):>10}  "
            f"{human_bytes(saved):>10}  "
            f"{pct:>5.1f}%  "
            f"{r.chosen_rel.as_posix():<70}  {r.note}"
        )


def have_oxipng() -> bool:
    return shutil.which("oxipng") is not None


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def have_pillow() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except Exception:
        return False


def replace_all_trailing_suffixes(path: Path, old_suffix: str, new_suffix: str) -> Path:
    """
    Replace ALL trailing occurrences of old_suffix with new_suffix (case-insensitive).
      foo.png.png -> foo.jpg
    """
    p = path
    while p.suffix.lower() == old_suffix.lower():
        p = p.with_suffix("")
    return p.with_suffix(new_suffix)


# -------------------------
# TeX dependency resolution
# -------------------------

def rel_to_root(root: Path, fpath: Path) -> Path:
    try:
        return fpath.resolve().relative_to(root.resolve())
    except Exception:
        return Path("external") / fpath.name


def resolve_tex_dep(paper_dir: Path, raw: str) -> Optional[Path]:
    raw = raw.strip()
    if not raw:
        return None

    p = Path(raw)
    candidates: List[Path] = []
    if p.suffix:
        candidates.append(p)
    else:
        candidates.append(Path(raw + ".tex"))
        candidates.append(p)

    for cand in candidates:
        a = (paper_dir / cand).resolve()
        if a.exists() and a.is_file():
            return a

    stem = p.name
    for match in paper_dir.rglob(stem + ".tex"):
        return match.resolve()

    return None


def resolve_graphic(paper_dir: Path, raw: str) -> Optional[Path]:
    raw = raw.strip()
    if not raw:
        return None

    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None

    base = (paper_dir / p).resolve()
    if base.exists():
        return base

    if not p.suffix:
        for ext in COMMON_GRAPHIC_EXTS:
            cand = (paper_dir / (raw + ext)).resolve()
            if cand.exists():
                return cand

    return None


def iter_tex_deps(paper_dir: Path, entry_tex: Path, verbose: bool = False) -> Set[Path]:
    to_visit = [entry_tex.resolve()]
    seen: Set[Path] = set()

    while to_visit:
        cur = to_visit.pop()
        if cur in seen:
            continue
        if not cur.exists():
            raise FileNotFoundError(f"Missing tex file: {cur}")
        seen.add(cur)

        text = strip_tex_comments(read_text(cur))
        for raw in TEX_INPUT_RE.findall(text):
            dep = resolve_tex_dep(paper_dir, raw)
            if dep is None:
                if verbose:
                    print(f"[warn] could not resolve tex include: {raw!r} (from {cur})")
                continue
            if dep not in seen:
                to_visit.append(dep)

    return seen


def gather_bibs(paper_dir: Path, tex_files: Iterable[Path], verbose: bool = False) -> Set[Path]:
    bibs: Set[Path] = set()
    for tex in tex_files:
        text = strip_tex_comments(read_text(tex))
        for raw_list in TEX_BIB_RE.findall(text):
            parts = [p.strip() for p in raw_list.split(",") if p.strip()]
            for part in parts:
                p = Path(part)
                cand = (paper_dir / (p if p.suffix else Path(part + ".bib"))).resolve()
                if cand.exists():
                    bibs.add(cand)
                else:
                    if verbose:
                        print(f"[warn] could not resolve bib: {part!r} (from {tex})")
    return bibs


def gather_graphics(paper_dir: Path, tex_files: Iterable[Path], verbose: bool = False) -> Set[Path]:
    graphics: Set[Path] = set()
    for tex in tex_files:
        text = strip_tex_comments(read_text(tex))
        for raw in TEX_GRAPHICS_RE.findall(text):
            g = resolve_graphic(paper_dir, raw)
            if g is None:
                if verbose:
                    print(f"[warn] could not resolve graphic: {raw!r} (from {tex})")
                continue
            graphics.add(g)
    return graphics


def gather_local_tex_assets(paper_dir: Path) -> Set[Path]:
    assets: Set[Path] = set()
    for ext in (".sty", ".cls", ".bst", ".bib"):
        for p in paper_dir.glob(f"*{ext}"):
            if p.is_file():
                assets.add(p.resolve())
    return assets


# -------------------------
# TeX rewriting
# -------------------------

def rewrite_tex_includegraphics(
    tex_text: str,
    paper_dir: Path,
    replace_map_by_resolved_rel: Dict[Path, Path],
    verbose: bool = False,
    tex_path_for_logs: Optional[Path] = None,
) -> str:
    """
    Resolve \\includegraphics paths against ORIGINAL paper_dir and replace if needed.
    This works even if staged originals were deleted (e.g., png->jpg).
    """
    full_re = re.compile(r"""\\includegraphics(?:\s*\[[^\]]*\])?\s*\{\s*([^\{\}]+?)\s*\}""")

    def repl(match: re.Match) -> str:
        raw = match.group(1)
        resolved_abs = resolve_graphic(paper_dir, raw)
        if resolved_abs is None:
            return match.group(0)

        resolved_rel = rel_to_root(paper_dir, resolved_abs)
        if resolved_rel not in replace_map_by_resolved_rel:
            return match.group(0)

        new_rel = replace_map_by_resolved_rel[resolved_rel]
        new_raw = new_rel.as_posix()

        if verbose:
            where = f" in {tex_path_for_logs.as_posix()}" if tex_path_for_logs else ""
            print(f"[tex]{where} includegraphics: {raw!r} -> {new_raw!r}")

        full = match.group(0)
        return full[: full.rfind("{") + 1] + new_raw + "}"

    return full_re.sub(repl, tex_text)


# -------------------------
# Image transforms (staging only)
# -------------------------

def oxipng_in_place(path: Path, level: int, extra: bool, verbose: bool, dry_run: bool) -> Tuple[bool, str]:
    if dry_run:
        return True, "dry-run"
    cmd = ["oxipng", f"-o{int(level)}", "--strip", "safe", str(path)]
    if extra:
        cmd.insert(1, "--zopfli")
    try:
        if verbose:
            print(f"[oxipng] {' '.join(cmd)}")
            proc = subprocess.run(cmd, check=False)
        else:
            proc = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode == 0:
            return True, "ok"
        return False, f"exit={proc.returncode}"
    except Exception as ex:
        return False, str(ex)


def png_to_jpg_ffmpeg(src_png: Path, dst_jpg: Path, qv: int, pix_fmt: str, verbose: bool, dry_run: bool) -> Tuple[bool, str]:
    if dry_run:
        return True, "dry-run"
    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src_png), "-q:v", str(int(qv)), "-pix_fmt", str(pix_fmt), str(dst_jpg)]
    try:
        if verbose:
            print(f"[ffmpeg] {' '.join(cmd)}")
            proc = subprocess.run(cmd, check=False)
        else:
            proc = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode == 0 and dst_jpg.exists():
            return True, "ok"
        return False, f"exit={proc.returncode}"
    except Exception as ex:
        return False, str(ex)


def png_to_jpg_pillow(src_png: Path, dst_jpg: Path, quality: int, verbose: bool, dry_run: bool) -> Tuple[bool, str]:
    try:
        from PIL import Image  # type: ignore
    except Exception as ex:
        return False, f"no pillow: {ex}"

    if verbose:
        print(f"[pillow] png->jpg q={quality}: {src_png} -> {dst_jpg}")
    if dry_run:
        return True, "dry-run"

    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_png) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(dst_jpg, format="JPEG", quality=int(quality), optimize=True, progressive=True)
    return True, "ok"


def recompress_jpeg_pillow(src_jpg: Path, dst_jpg: Path, quality: int, verbose: bool, dry_run: bool) -> Tuple[bool, str]:
    try:
        from PIL import Image  # type: ignore
    except Exception as ex:
        return False, f"no pillow: {ex}"

    if verbose:
        print(f"[pillow] jpg recompress q={quality}: {src_jpg} -> {dst_jpg}")
    if dry_run:
        return True, "dry-run"

    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_jpg) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(dst_jpg, format="JPEG", quality=int(quality), optimize=True, progressive=True)
    return True, "ok"


def _choose_png_to_jpg_backend(plan: BuildPlan) -> str:
    mode = plan.png_jpg_mode
    if mode == "auto":
        return "ffmpeg" if have_ffmpeg() else ("pillow" if have_pillow() else "none")
    if mode == "ffmpeg":
        return "ffmpeg" if have_ffmpeg() else "none"
    if mode == "pillow":
        return "pillow" if have_pillow() else "none"
    return "none"


def choose_png_variant_in_staging(
    staged_png_abs: Path,
    source_rel: Path,
    plan: BuildPlan,
) -> Tuple[Path, Path, List[TransformRecord]]:
    """
    Decide & materialize the single best (smallest) staged variant for a PNG.

    Returns (chosen_abs, chosen_rel, records). Ensures only chosen file remains.
    Handles .png.png properly by removing all trailing ".png" if JPG wins.

    Extra requested rule:
      After converting png->jpg (high quality), if jpg still > jpeg_threshold,
      also convert png->jpg with target jpeg compression settings (--jpeg-quality, via Pillow if available),
      and keep the smallest jpg.
    """
    records: List[TransformRecord] = []
    old_bytes = staged_png_abs.stat().st_size

    tmp_opt_png = staged_png_abs.with_name(staged_png_abs.name + ".__tmp_oxipng.png")
    tmp_hi_jpg = staged_png_abs.with_name(staged_png_abs.name + ".__tmp_frompng_hi.jpg")
    tmp_q_jpg = staged_png_abs.with_name(staged_png_abs.name + f".__tmp_frompng_q{plan.jpeg_quality}.jpg")

    best_abs = staged_png_abs
    best_rel = source_rel
    best_bytes = old_bytes
    best_note = "kept staged png"

    # Candidate: oxipng
    if plan.oxipng and have_oxipng():
        if not plan.dry_run:
            shutil.copy2(staged_png_abs, tmp_opt_png)
        ok, msg = oxipng_in_place(tmp_opt_png, plan.oxipng_level, plan.oxipng_extra, plan.verbose, plan.dry_run)
        if ok and (plan.dry_run or tmp_opt_png.exists()):
            opt_bytes = best_bytes if plan.dry_run else tmp_opt_png.stat().st_size
            records.append(
                TransformRecord(
                    kind="png_try_oxipng",
                    source_rel=source_rel,
                    chosen_rel=source_rel,
                    old_bytes=old_bytes,
                    new_bytes=opt_bytes,
                    note=f"candidate o{plan.oxipng_level}" + ("+zopfli" if plan.oxipng_extra else ""),
                )
            )
            if opt_bytes < best_bytes:
                best_abs, best_rel, best_bytes = tmp_opt_png, source_rel, opt_bytes
                best_note = f"oxipng smaller ({msg})"
        else:
            if plan.verbose:
                print(f"[warn] oxipng failed for {source_rel.as_posix()}: {msg}")

    # Candidate(s): png->jpg if png > threshold
    if plan.png_to_jpg and old_bytes > plan.png_threshold_bytes:
        backend = _choose_png_to_jpg_backend(plan)
        if backend == "none":
            if plan.verbose:
                print(f"[warn] png->jpg requested but no backend available for {source_rel.as_posix()}")
        else:
            # High-quality candidate (ffmpeg preferred / pillow fallback)
            if backend == "ffmpeg":
                ok, msg = png_to_jpg_ffmpeg(
                    staged_png_abs, tmp_hi_jpg, plan.ffmpeg_qv, plan.ffmpeg_pix_fmt, plan.verbose, plan.dry_run
                )
                note = f"ffmpeg -q:v {plan.ffmpeg_qv} -pix_fmt {plan.ffmpeg_pix_fmt} ({msg})"
            else:
                ok, msg = png_to_jpg_pillow(
                    staged_png_abs, tmp_hi_jpg, plan.pillow_png_to_jpg_quality, plan.verbose, plan.dry_run
                )
                note = f"pillow q={plan.pillow_png_to_jpg_quality} ({msg})"

            if ok and (plan.dry_run or tmp_hi_jpg.exists()):
                hi_bytes = best_bytes if plan.dry_run else tmp_hi_jpg.stat().st_size
                jpg_rel = replace_all_trailing_suffixes(source_rel, ".png", ".jpg")
                records.append(
                    TransformRecord(
                        kind="png_try_to_jpg_hi",
                        source_rel=source_rel,
                        chosen_rel=jpg_rel,
                        old_bytes=old_bytes,
                        new_bytes=hi_bytes,
                        note=f"candidate {note}",
                    )
                )
                if hi_bytes < best_bytes:
                    best_abs, best_rel, best_bytes = tmp_hi_jpg, jpg_rel, hi_bytes
                    best_note = f"jpg smaller (hi) ({note})"

                # EXTRA RULE: if hi jpg is still huge (> jpeg threshold), also try "target jpeg compression"
                # This requires Pillow (because ffmpeg uses q:v scale, not jpeg "quality" directly).
                if (not plan.dry_run) and hi_bytes > plan.jpeg_threshold_bytes and have_pillow():
                    ok2, msg2 = png_to_jpg_pillow(
                        staged_png_abs, tmp_q_jpg, plan.jpeg_quality, plan.verbose, plan.dry_run
                    )
                    note2 = f"pillow target q={plan.jpeg_quality} ({msg2})"
                    if ok2 and tmp_q_jpg.exists():
                        q_bytes = tmp_q_jpg.stat().st_size
                        records.append(
                            TransformRecord(
                                kind="png_try_to_jpg_q",
                                source_rel=source_rel,
                                chosen_rel=jpg_rel,
                                old_bytes=old_bytes,
                                new_bytes=q_bytes,
                                note=f"candidate {note2}",
                            )
                        )
                        if q_bytes < best_bytes:
                            best_abs, best_rel, best_bytes = tmp_q_jpg, jpg_rel, q_bytes
                            best_note = f"jpg smaller (target q) ({note2})"
                    elif plan.verbose:
                        print(f"[warn] target-q png->jpg failed for {source_rel.as_posix()}: {note2}")

            else:
                if plan.verbose:
                    print(f"[warn] png->jpg (hi) failed for {source_rel.as_posix()}: {note}")

    # Finalize: keep ONLY the winner
    if plan.dry_run:
        # No filesystem modifications; return logical winner
        records.append(
            TransformRecord(
                kind="png_choose",
                source_rel=source_rel,
                chosen_rel=best_rel,
                old_bytes=old_bytes,
                new_bytes=best_bytes,
                note=best_note,
            )
        )
        return staged_png_abs.parent / best_rel.name, best_rel, records

    # Cleanup helpers
    def _rm_if_exists(p: Path) -> None:
        if p.exists():
            p.unlink()

    # Winner is original staged PNG
    if best_abs == staged_png_abs:
        _rm_if_exists(tmp_opt_png)
        _rm_if_exists(tmp_hi_jpg)
        _rm_if_exists(tmp_q_jpg)
        records.append(
            TransformRecord(
                kind="png_keep",
                source_rel=source_rel,
                chosen_rel=source_rel,
                old_bytes=old_bytes,
                new_bytes=old_bytes,
                note="kept png",
            )
        )
        return staged_png_abs, source_rel, records

    # Winner is optimized PNG
    if best_abs == tmp_opt_png:
        shutil.move(str(tmp_opt_png), str(staged_png_abs))
        _rm_if_exists(tmp_hi_jpg)
        _rm_if_exists(tmp_q_jpg)
        new_bytes = staged_png_abs.stat().st_size
        records.append(
            TransformRecord(
                kind="png_oxipng",
                source_rel=source_rel,
                chosen_rel=source_rel,
                old_bytes=old_bytes,
                new_bytes=new_bytes,
                note="oxipng chosen",
            )
        )
        return staged_png_abs, source_rel, records

    # Winner is a JPG (tmp_hi_jpg or tmp_q_jpg)
    final_jpg_abs = replace_all_trailing_suffixes(staged_png_abs, ".png", ".jpg")
    if final_jpg_abs.exists():
        final_jpg_abs.unlink()
    shutil.move(str(best_abs), str(final_jpg_abs))

    # Delete staged PNG and other temps
    _rm_if_exists(staged_png_abs)
    _rm_if_exists(tmp_opt_png)
    # remove non-winner jpg temp if present
    if best_abs != tmp_hi_jpg:
        _rm_if_exists(tmp_hi_jpg)
    if best_abs != tmp_q_jpg:
        _rm_if_exists(tmp_q_jpg)

    new_bytes = final_jpg_abs.stat().st_size
    records.append(
        TransformRecord(
            kind="png_to_jpg",
            source_rel=source_rel,
            chosen_rel=best_rel,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            note=best_note,
        )
    )
    return final_jpg_abs, best_rel, records


def choose_jpeg_variant_in_staging(
    staged_jpg_abs: Path,
    source_rel: Path,
    plan: BuildPlan,
) -> Tuple[Path, Path, List[TransformRecord]]:
    """
    Optionally recompress JPEG in place if it wins by size.
    """
    records: List[TransformRecord] = []
    cur_bytes = staged_jpg_abs.stat().st_size

    if not plan.recompress_jpegs or cur_bytes <= plan.jpeg_threshold_bytes:
        return staged_jpg_abs, source_rel, records

    if not have_pillow():
        if plan.verbose:
            print(f"[warn] JPEG recompress requested but Pillow not available; keeping {source_rel.as_posix()}")
        return staged_jpg_abs, source_rel, records

    tmp = staged_jpg_abs.with_name(staged_jpg_abs.stem + f".__tmp_q{plan.jpeg_quality}.jpg")
    ok, msg = recompress_jpeg_pillow(staged_jpg_abs, tmp, plan.jpeg_quality, plan.verbose, plan.dry_run)
    if not ok:
        if plan.verbose:
            print(f"[warn] JPEG recompress failed for {source_rel.as_posix()}: {msg}")
        return staged_jpg_abs, source_rel, records

    if plan.dry_run:
        records.append(
            TransformRecord(
                kind="jpg_try_recompress",
                source_rel=source_rel,
                chosen_rel=source_rel,
                old_bytes=cur_bytes,
                new_bytes=cur_bytes,
                note=f"dry-run q={plan.jpeg_quality}",
            )
        )
        return staged_jpg_abs, source_rel, records

    if not tmp.exists():
        return staged_jpg_abs, source_rel, records

    new_bytes = tmp.stat().st_size
    records.append(
        TransformRecord(
            kind="jpg_try_recompress",
            source_rel=source_rel,
            chosen_rel=source_rel,
            old_bytes=cur_bytes,
            new_bytes=new_bytes,
            note=f"candidate q={plan.jpeg_quality}",
        )
    )

    if new_bytes < cur_bytes:
        staged_jpg_abs.unlink()
        shutil.move(str(tmp), str(staged_jpg_abs))
        records.append(
            TransformRecord(
                kind="jpg_recompress",
                source_rel=source_rel,
                chosen_rel=source_rel,
                old_bytes=cur_bytes,
                new_bytes=staged_jpg_abs.stat().st_size,
                note=f"chosen q={plan.jpeg_quality}",
            )
        )
    else:
        tmp.unlink()

    return staged_jpg_abs, source_rel, records


# -------------------------
# Build
# -------------------------

def default_paths() -> Tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    paper_dir = script_path.parent.parent
    entry_tex = paper_dir / "main.tex"
    return paper_dir, entry_tex


def build(plan: BuildPlan) -> None:
    paper_dir = plan.paper_dir.resolve()
    entry_tex = plan.entry_tex.resolve()
    out_dir = plan.out_dir.resolve()

    if plan.verbose:
        print(f"[info] paper_dir = {paper_dir}")
        print(f"[info] entry_tex = {entry_tex}")
        print(f"[info] out_dir   = {out_dir}")
        print(f"[info] jobs      = {plan.jobs}")
        print(f"[info] png_threshold = {human_bytes(plan.png_threshold_bytes)}")
        print(f"[info] oxipng={plan.oxipng} level={plan.oxipng_level} extra={plan.oxipng_extra} (available={have_oxipng()})")
        print(f"[info] png_to_jpg={plan.png_to_jpg} mode={plan.png_jpg_mode} (ffmpeg={have_ffmpeg()} pillow={have_pillow()})")
        print(f"[info] recompress_jpegs={plan.recompress_jpegs} jpeg_threshold={human_bytes(plan.jpeg_threshold_bytes)} jpeg_quality={plan.jpeg_quality}")

    tex_files = iter_tex_deps(paper_dir, entry_tex, verbose=plan.verbose)
    tex_files.add(entry_tex)

    bibs = gather_bibs(paper_dir, tex_files, verbose=plan.verbose)
    graphics_abs = gather_graphics(paper_dir, tex_files, verbose=plan.verbose)
    local_assets = gather_local_tex_assets(paper_dir)

    if not plan.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Path] = []
    all_records: List[TransformRecord] = []

    # Map from resolved original relpath (under paper_dir) -> chosen staged relpath
    chosen_map: Dict[Path, Path] = {}

    # 1) Stage TeX (copy; rewrite later)
    staged_tex_rels: List[Path] = []
    for tex_abs in sorted(tex_files):
        tex_rel = rel_to_root(paper_dir, tex_abs)
        dst = out_dir / tex_rel
        if plan.verbose:
            print(f"[tex] stage {tex_rel.as_posix()}")
        if not plan.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tex_abs, dst)
        manifest.append(tex_rel)
        staged_tex_rels.append(tex_rel)

    # 2) Stage local assets
    for asset_abs in sorted(bibs | local_assets):
        asset_rel = rel_to_root(paper_dir, asset_abs)
        safe_copy2(asset_abs, out_dir / asset_rel, plan.verbose, plan.dry_run)
        manifest.append(asset_rel)

    # 3) Stage graphics originals first (fast I/O), then process in parallel
    #    (we stage original name first so we can run tools on staged paths without touching sources)
    staged_fig_items: List[Tuple[Path, Path]] = []  # (source_abs, source_rel)
    for g_abs in sorted(graphics_abs):
        g_rel = rel_to_root(paper_dir, g_abs)
        dst_abs = out_dir / g_rel
        safe_copy2(g_abs, dst_abs, plan.verbose, plan.dry_run)
        staged_fig_items.append((g_abs, g_rel))

    # Worker for per-figure transformation
    def process_figure(g_abs: Path, g_rel: Path) -> Tuple[Path, Path, List[TransformRecord]]:
        staged_abs = out_dir / g_rel
        ext = g_abs.suffix.lower()
        if ext == ".png":
            return choose_png_variant_in_staging(staged_abs, g_rel, plan)
        if ext in (".jpg", ".jpeg"):
            return choose_jpeg_variant_in_staging(staged_abs, g_rel, plan)
        return staged_abs, g_rel, []  # pdf/eps/etc

    # Run transformations concurrently
    if plan.verbose:
        print(f"[info] processing {len(staged_fig_items)} figure(s) with ThreadPoolExecutor(jobs={plan.jobs})")

    with ThreadPoolExecutor(max_workers=max(1, plan.jobs)) as ex:
        futures = [ex.submit(process_figure, g_abs, g_rel) for g_abs, g_rel in staged_fig_items]
        for fut in as_completed(futures):
            chosen_abs, chosen_rel, records = fut.result()
            # Need original resolved rel to key mapping. That is g_rel (source relative path under paper_dir).
            # We can infer it from records[0].source_rel if present; otherwise chosen_rel itself.
            source_rel = records[0].source_rel if records else chosen_rel
            chosen_map[source_rel] = chosen_rel
            manifest.append(chosen_rel)
            all_records.extend(records)

    # 4) Rewrite staged TeX includegraphics based on ORIGINAL paper_dir resolution
    if plan.verbose:
        print(f"[info] rewriting staged TeX includegraphics based on {len(chosen_map)} mapping(s)")

    for tex_rel in staged_tex_rels:
        staged_tex_abs = out_dir / tex_rel
        text = read_text(staged_tex_abs) if not plan.dry_run else read_text(paper_dir / tex_rel)
        new_text = rewrite_tex_includegraphics(
            text,
            paper_dir=paper_dir,
            replace_map_by_resolved_rel=chosen_map,
            verbose=plan.verbose,
            tex_path_for_logs=tex_rel,
        )
        write_text(staged_tex_abs, new_text, plan.dry_run)

    # 5) Manifest + zip
    write_manifest(out_dir, manifest, plan.dry_run)
    if plan.make_zip:
        zip_path = plan.zip_path or out_dir.with_suffix(".zip")
        make_zip_from_dir(out_dir, zip_path, plan.dry_run, plan.verbose)

    # 6) Report (sorted by new size desc)
    # Keep one "final choice" record per source figure for offender view:
    # Prefer terminal records (png_to_jpg/png_oxipng/png_keep/jpg_recompress/jpg_keep).
    terminal_kinds = {"png_to_jpg", "png_oxipng", "png_keep", "jpg_recompress"}
    finals: Dict[Path, TransformRecord] = {}
    for r in all_records:
        if r.kind in terminal_kinds:
            finals[r.source_rel] = r
    # Some figures might have no terminal record; add synthetic entries based on chosen_map
    for src_rel, chosen_rel in chosen_map.items():
        if src_rel not in finals:
            staged_path = out_dir / chosen_rel
            size = staged_path.stat().st_size if (not plan.dry_run and staged_path.exists()) else 0
            finals[src_rel] = TransformRecord(
                kind="kept",
                source_rel=src_rel,
                chosen_rel=chosen_rel,
                old_bytes=size,
                new_bytes=size,
                note="no transform",
            )

    print_transform_table(list(finals.values()))
    if plan.verbose:
        print("[done]")


# -------------------------
# CLI
# -------------------------

def main(argv: Optional[List[str]] = None) -> None:
    paper_dir = (Path(__file__).resolve().parent.parent)
    entry_tex = paper_dir / "main.tex"

    ap = argparse.ArgumentParser(description="Package paper for arXiv with minimal figure variants + parallel processing.")
    ap.add_argument("--entry", type=Path, default=entry_tex, help="Path to main .tex file.")
    ap.add_argument("--paper-dir", type=Path, default=paper_dir, help="Paper root directory.")
    ap.add_argument("--out-dir", type=Path, default=paper_dir / "_arxiv_submission", help="Staging output directory.")
    ap.add_argument("--zip", action="store_true", help="Also create a zip from staging directory.")
    ap.add_argument("--zip-path", type=Path, default=None, help="Optional explicit zip path.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing outputs.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="Parallel workers for figure processing (default: ~half cores).")

    # PNG
    ap.add_argument("--oxipng", action="store_true", help="Try oxipng optimization on staged PNG candidates.")
    ap.add_argument("--oxipng-level", type=int, default=4, help="oxipng -oN (0-6). Default 4.")
    ap.add_argument("--oxipng-extra", action="store_true", help="Use --zopfli (slower).")

    ap.add_argument("--png-to-jpg", action="store_true",
                    help="If original PNG is larger than --png-threshold-mb, consider converting to JPG and pick smallest.")
    ap.add_argument("--png-threshold-mb", type=float, default=0.1,
                    help="PNG size threshold (MiB) above which JPG conversion is considered (default 0.1).")
    ap.add_argument("--png-jpg-mode", choices=["auto", "ffmpeg", "pillow"], default="auto",
                    help="PNG->JPG converter backend (default auto prefers ffmpeg).")
    ap.add_argument("--ffmpeg-qv", type=int, default=2, help="ffmpeg -q:v (lower=better). Default 2.")
    ap.add_argument("--ffmpeg-pix-fmt", type=str, default="yuvj444p", help="ffmpeg -pix_fmt. Default yuvj444p.")
    ap.add_argument("--pillow-png-to-jpg-quality", type=int, default=92,
                    help="Pillow JPEG quality when used for PNG->JPG baseline (default 92).")

    # JPEG recompress
    ap.add_argument("--recompress-jpegs", action="store_true",
                    help="Try recompressing staged JPEGs above threshold and keep smaller of original vs recompressed.")
    ap.add_argument("--jpeg-threshold-mb", type=float, default=1.0,
                    help="JPEG size threshold (MiB) above which recompress is tried (default 1.0).")
    ap.add_argument("--jpeg-quality", type=int, default=80,
                    help="Target JPEG quality for recompress, and for PNG->JPG 'target quality' retry if needed (default 80).")

    args = ap.parse_args(argv)

    plan = BuildPlan(
        paper_dir=args.paper_dir,
        entry_tex=args.entry,
        out_dir=args.out_dir,
        make_zip=bool(args.zip),
        zip_path=args.zip_path,
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
        jobs=max(1, int(args.jobs)),

        oxipng=bool(args.oxipng),
        oxipng_level=max(0, min(int(args.oxipng_level), 6)),
        oxipng_extra=bool(args.oxipng_extra),

        png_to_jpg=bool(args.png_to_jpg),
        png_threshold_bytes=int(float(args.png_threshold_mb) * 1024 * 1024),
        png_jpg_mode=str(args.png_jpg_mode),
        ffmpeg_qv=max(2, min(int(args.ffmpeg_qv), 31)),
        ffmpeg_pix_fmt=str(args.ffmpeg_pix_fmt),
        pillow_png_to_jpg_quality=max(50, min(int(args.pillow_png_to_jpg_quality), 95)),

        recompress_jpegs=bool(args.recompress_jpegs),
        jpeg_threshold_bytes=int(float(args.jpeg_threshold_mb) * 1024 * 1024),
        jpeg_quality=max(1, min(int(args.jpeg_quality), 95)),
    )

    build(plan)


if __name__ == "__main__":
    main()
