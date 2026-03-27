#!/usr/bin/env python3
"""
Inspect git submodules and report what is needed to normalize them for the
shitspotter workflow.

Run from the top-level of the superproject:

    python dev/inspect_submodules.py
    python dev/inspect_submodules.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=check,
    )


def git(args: List[str], cwd: Optional[Path] = None, check: bool = False) -> subprocess.CompletedProcess:
    return run(["git", *args], cwd=cwd, check=check)


def git_ok(args: List[str], cwd: Optional[Path] = None) -> bool:
    proc = git(args, cwd=cwd)
    return proc.returncode == 0


def git_out(args: List[str], cwd: Optional[Path] = None) -> str:
    proc = git(args, cwd=cwd)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def parse_gitmodules(repo_root: Path) -> List[Dict[str, str]]:
    proc = git(["config", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.(path|url|branch)$"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to parse .gitmodules:\n{proc.stderr}")

    pat = re.compile(r"^submodule\.(?P<name>.+?)\.(?P<key>path|url|branch)\s+(?P<val>.*)$")
    acc: Dict[str, Dict[str, str]] = {}
    for line in proc.stdout.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        key = m.group("key")
        val = m.group("val")
        acc.setdefault(name, {"name": name})
        acc[name][key] = val

    items = []
    for name, data in sorted(acc.items(), key=lambda kv: kv[1].get("path", kv[0])):
        items.append(data)
    return items


def parse_remotes(subrepo: Path) -> Dict[str, Dict[str, str]]:
    proc = git(["remote", "-v"], cwd=subrepo)
    remotes: Dict[str, Dict[str, str]] = {}
    if proc.returncode != 0:
        return remotes
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # name \t url (fetch)
        m = re.match(r"^(\S+)\s+(\S+)\s+\((fetch|push)\)$", line)
        if not m:
            continue
        name, url, kind = m.groups()
        remotes.setdefault(name, {})
        remotes[name][kind] = url
    return remotes


def remote_has_branch(subrepo: Path, remote: str, branch: str) -> Optional[bool]:
    proc = git(["ls-remote", "--heads", remote, branch], cwd=subrepo)
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def get_local_branches(subrepo: Path) -> List[str]:
    out = git_out(["for-each-ref", "--format=%(refname:short)", "refs/heads"], cwd=subrepo)
    return [line.strip() for line in out.splitlines() if line.strip()]


def get_current_branch(subrepo: Path) -> Optional[str]:
    out = git_out(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd=subrepo)
    return out or None


def get_head_sha(subrepo: Path) -> Optional[str]:
    out = git_out(["rev-parse", "HEAD"], cwd=subrepo)
    return out or None


def get_pinned_sha(repo_root: Path, path: str) -> Optional[str]:
    out = git_out(["ls-tree", "HEAD", path], cwd=repo_root)
    if not out:
        return None
    # format: 160000 commit <sha>\tpath
    parts = out.split()
    if len(parts) >= 3:
        return parts[2]
    return None


def get_branch_upstream(subrepo: Path, branch: str) -> Optional[str]:
    out = git_out(["rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}"], cwd=subrepo)
    return out or None


def summarize_url(url: str) -> str:
    if not url:
        return ""
    return url


def is_probable_fork_url(url: str) -> bool:
    return "github.com/Erotemic/" in url or "git@github.com:Erotemic/" in url


def inspect_submodule(repo_root: Path, item: Dict[str, str]) -> Dict[str, object]:
    name = item["name"]
    path = item["path"]
    subrepo = repo_root / path

    info: Dict[str, object] = {
        "name": name,
        "path": path,
        "exists": subrepo.exists(),
        "gitmodules_url": item.get("url"),
        "gitmodules_branch": item.get("branch"),
    }

    if not subrepo.exists():
        info["status"] = "missing"
        return info

    remotes = parse_remotes(subrepo)
    current_branch = get_current_branch(subrepo)
    local_branches = get_local_branches(subrepo)
    head_sha = get_head_sha(subrepo)
    pinned_sha = get_pinned_sha(repo_root, path)

    erotemic_remote_names = []
    for rname, cfg in remotes.items():
        fetch_url = cfg.get("fetch", "")
        push_url = cfg.get("push", "")
        if is_probable_fork_url(fetch_url) or is_probable_fork_url(push_url):
            erotemic_remote_names.append(rname)

    per_remote_shitspotter: Dict[str, Optional[bool]] = {}
    for rname in remotes:
        per_remote_shitspotter[rname] = remote_has_branch(subrepo, rname, "shitspotter")

    info.update(
        {
            "status": "ok",
            "remotes": remotes,
            "remote_names": sorted(remotes.keys()),
            "has_erotemic_remote": "Erotemic" in remotes,
            "erotemic_remote_names_with_erotemic_url": erotemic_remote_names,
            "current_branch": current_branch,
            "detached_head": current_branch is None,
            "local_branches": local_branches,
            "has_local_shitspotter_branch": "shitspotter" in local_branches,
            "on_shitspotter_branch": current_branch == "shitspotter",
            "shitspotter_upstream": get_branch_upstream(subrepo, "shitspotter") if "shitspotter" in local_branches else None,
            "remote_has_shitspotter_branch": per_remote_shitspotter,
            "head_sha": head_sha,
            "pinned_sha": pinned_sha,
            "head_matches_superproject": (head_sha == pinned_sha) if head_sha and pinned_sha else None,
        }
    )

    needs_fork = False
    reasons = []

    if "Erotemic" not in remotes:
        needs_fork = True
        reasons.append("missing_remote_named_Erotemic")

    if not any(v is True for v in per_remote_shitspotter.values()):
        needs_fork = True
        reasons.append("no_remote_advertises_shitspotter_branch")

    if not is_probable_fork_url(item.get("url", "")):
        reasons.append("gitmodules_not_pointing_to_Erotemic_fork")

    if "shitspotter" not in local_branches:
        reasons.append("missing_local_shitspotter_branch")

    if current_branch != "shitspotter":
        reasons.append("not_currently_on_shitspotter_branch")

    info["likely_needs_fork_or_setup"] = needs_fork
    info["action_reasons"] = reasons
    return info


def short_bool(val: Optional[bool]) -> str:
    if val is True:
        return "yes"
    if val is False:
        return "no"
    return "?"

def print_table(results: List[Dict[str, object]]) -> None:
    headers = [
        "path",
        "Erotemic_remote",
        "local_shitspotter",
        "on_shitspotter",
        "gitmodules_branch",
        "gitmodules_url_kind",
        "remote_shitspotter",
        "HEAD_eq_super",
        "needs_fork/setup",
    ]
    rows = []
    for r in results:
        remote_map = r.get("remote_has_shitspotter_branch", {}) or {}
        advertised = ",".join(f"{k}:{short_bool(v)}" for k, v in sorted(remote_map.items())) or "-"
        gm_url = str(r.get("gitmodules_url") or "")
        if "Erotemic/" in gm_url:
            gm_kind = "Erotemic"
        elif gm_url:
            gm_kind = "other"
        else:
            gm_kind = "-"
        rows.append([
            str(r.get("path", "")),
            short_bool(True if r.get("has_erotemic_remote") else False),
            short_bool(True if r.get("has_local_shitspotter_branch") else False),
            short_bool(True if r.get("on_shitspotter_branch") else False),
            str(r.get("gitmodules_branch") or "-"),
            gm_kind,
            advertised,
            short_bool(r.get("head_matches_superproject")),
            short_bool(True if r.get("likely_needs_fork_or_setup") else False),
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))

    print("\nDetailed notes:")
    for r in results:
        reasons = r.get("action_reasons", []) or []
        if reasons:
            print(f"- {r['path']}: {', '.join(reasons)}")
        else:
            print(f"- {r['path']}: looks aligned with target policy")

    print("\nSuggested next checks:")
    print("- If needs_fork/setup=yes and gitmodules_url_kind=other, you probably need an Erotemic fork or a .gitmodules update.")
    print("- If local_shitspotter=no but some remote says yes, create a local tracking branch.")
    print("- If Erotemic_remote=no but an Erotemic fork exists, add the remote inside that submodule.")
    print("- If HEAD_eq_super=no, the submodule checkout and superproject gitlink are out of sync.")


import ubelt as ub
import scriptconfig as scfg

class InspectSubmodulesConfig(scfg.DataConfig):
    """
    """
    json = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Emit machine-readable JSON instead of a table
            '''))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table")
    args = parser.parse_args()

    repo_root = Path.cwd()
    if not (repo_root / ".git").exists() and not (repo_root / ".gitmodules").exists():
        print("Run this from the top-level of the superproject.", file=sys.stderr)
        return 2
    if not (repo_root / ".gitmodules").exists():
        print("No .gitmodules file found.", file=sys.stderr)
        return 2

    items = parse_gitmodules(repo_root)
    results = [inspect_submodule(repo_root, item) for item in items]

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_table(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
