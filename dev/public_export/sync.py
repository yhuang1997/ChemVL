"""Checkout Tier-1 / Tier-2 manifest paths from a source git ref into the current worktree."""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set

import yaml

MANIFEST = Path(__file__).resolve().parents[2] / "docs/public_export/manifest.yaml"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists():
            return p
    raise RuntimeError("Cannot locate git repo root")


def _load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expand_glob(root: Path, pattern: str) -> List[str]:
    if pattern.startswith("."):
        p = root / pattern
        if p.is_file():
            return [pattern]
        return []
    if "**" in pattern:
        base = pattern.split("/**", 1)[0]
        base_path = root / base
        if not base_path.exists():
            return []
        out: List[str] = []
        if base_path.is_file():
            return [base]
        for p in base_path.rglob("*"):
            if p.is_file():
                out.append(p.relative_to(root).as_posix())
        return sorted(out)
    p = root / pattern
    if p.is_file():
        return [pattern]
    if p.is_dir():
        return sorted(
            x.relative_to(root).as_posix()
            for x in p.rglob("*")
            if x.is_file()
        )
    return []


def _matches_glob(rel: str, pattern: str) -> bool:
    if fnmatch.fnmatch(rel, pattern):
        return True
    if "**" in pattern:
        prefix = pattern.split("/**", 1)[0]
        if rel == prefix or rel.startswith(prefix + "/"):
            return True
    return False


def _matches_any(rel: str, patterns: Iterable[str]) -> bool:
    return any(_matches_glob(rel, pat) for pat in patterns)


def collect_tier1_paths(manifest: dict, root: Path) -> List[str]:
    paths: Set[str] = set()
    for item in manifest.get("tier1_cli", []):
        paths.add(item["path"])
    for item in manifest.get("tier1_tools", []):
        paths.add(item["path"])
    for glob_pat in manifest.get("tier1_support_globs", []):
        paths.update(_expand_glob(root, glob_pat))
    return sorted(paths)


def _git_ls_files(source: str, root: Path, pathspec: str) -> List[str]:
    cmd = ["git", "ls-tree", "-r", "--name-only", source, "--", pathspec]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _glob_to_pathspec(pattern: str) -> str:
    if pattern.startswith("."):
        return pattern
    if "**" in pattern:
        return pattern.split("/**", 1)[0]
    return pattern


def collect_tier2_paths(manifest: dict, source: str, root: Path) -> List[str]:
    paths: Set[str] = set()
    for pkg in manifest.get("tier2_packages", []):
        if pkg.get("status") != "included":
            continue
        pkg_paths: Set[str] = set()
        for glob_pat in pkg.get("globs", []):
            pathspec = _glob_to_pathspec(glob_pat)
            pkg_paths.update(_git_ls_files(source, root, pathspec))
        exclude = pkg.get("exclude_globs", [])
        for rel in pkg_paths:
            if exclude and _matches_any(rel, exclude):
                continue
            paths.add(rel)
    return sorted(paths)


def _is_excluded(rel: str, manifest: dict) -> bool:
    for name in manifest.get("exclude_root_scripts", []):
        if rel == name:
            return True
    for pat in manifest.get("exclude_tier3_globs", []):
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, pat.rstrip("/") + "/**"):
            return True
        if "/" in pat and rel.startswith(pat.rstrip("*").rstrip("/") + "/"):
            return True
    return False


def filter_checkout_paths(paths: Iterable[str], manifest: dict) -> List[str]:
    out: List[str] = []
    for rel in paths:
        if _is_excluded(rel, manifest):
            continue
        out.append(rel)
    return out


def git_checkout(source: str, paths: List[str], root: Path) -> int:
    if not paths:
        print("No paths to checkout.")
        return 0
    chunk = 200
    for i in range(0, len(paths), chunk):
        batch = paths[i : i + chunk]
        cmd = ["git", "checkout", source, "--", *batch]
        print("Running:", " ".join(cmd[:6]), f"... ({len(batch)} paths)")
        rc = subprocess.call(cmd, cwd=str(root))
        if rc != 0:
            return rc
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync manifest paths from a git ref.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--source", default="origin/main", help="Git ref to checkout from")
    parser.add_argument("--dry-run", action="store_true", help="List paths only")
    parser.add_argument("--apply", action="store_true", help="Run git checkout")
    parser.add_argument(
        "--tier2",
        action="store_true",
        help="Include tier2_packages with status=included",
    )
    parser.add_argument(
        "--tier2-only",
        action="store_true",
        help="Sync only included tier2 packages (skip tier1)",
    )
    args = parser.parse_args(argv)

    if not args.dry_run and not args.apply:
        parser.error("Specify --dry-run or --apply")

    root = _repo_root()
    manifest = _load_manifest(args.manifest)

    paths: Set[str] = set()
    if not args.tier2_only:
        paths.update(collect_tier1_paths(manifest, root))
    if args.tier2 or args.tier2_only:
        paths.update(collect_tier2_paths(manifest, args.source, root))

    ordered = filter_checkout_paths(sorted(paths), manifest)

    included_t2 = [
        pkg["id"]
        for pkg in manifest.get("tier2_packages", [])
        if pkg.get("status") == "included"
    ]

    print(f"manifest: {args.manifest}")
    print(f"repo root: {root}")
    print(f"source ref: {args.source}")
    if args.tier2 or args.tier2_only:
        print(f"tier2 included packages: {', '.join(included_t2) or '(none)'}")
    print(f"paths to sync: {len(ordered)}")
    for p in ordered[:40]:
        print(f"  {p}")
    if len(ordered) > 40:
        print(f"  ... and {len(ordered) - 40} more")

    if args.dry_run:
        return 0
    return git_checkout(args.source, ordered, root)


if __name__ == "__main__":
    raise SystemExit(main())
