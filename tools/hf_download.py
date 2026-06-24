#!/usr/bin/env python3
"""
Download and unpack ChemVL data from the Hugging Face **dataset** Hub.

**Status — work in progress:** the Hub dataset snapshot and this helper are committed
for convenience, but uploads / LFS objects may still be **incomplete** or flaky
(network, SSL). If download fails or stalls, use the **optional Quark mirror**
(time-limited) described in the root ``README.md`` until the Hub side is stable.

Subcommands:

- ``download`` — ``snapshot_download`` into ``CHEMVL_DATA_ROOT`` (raw Hub layout)
- ``unpack`` — extract ``archives/*.tar.zst`` per ``docs/data/hf_pack_manifest.json``

After **download + unpack**, training YAML paths such as
``data_root / "pretraining_datasets" / "10M-106mds" / "mds.csv"`` resolve when
``data_cfg.data_root`` equals ``CHEMVL_DATA_ROOT``.

See ``docs/data/HF_DATASET_CARD.md`` for the full after-unpack tree.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "yzhuang1997/chemvl-data"
DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "docs/data/hf_pack_manifest.json"


def _default_local_dir() -> str | None:
    raw = os.environ.get("CHEMVL_DATA_ROOT", "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    try:
        from utils.path_utils import get_data_root

        return str(get_data_root())
    except Exception:
        return None


def _data_root_path() -> Path:
    local = _default_local_dir()
    if not local:
        raise SystemExit(
            "ERROR: set CHEMVL_DATA_ROOT or pass --local-dir (your ChemVL data root)."
        )
    return Path(local)


def _load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _extract_tar_zst(archive: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(extract_to)], check=True)


def cmd_download(args: argparse.Namespace) -> int:
    local_dir = args.local_dir or _default_local_dir()
    if not local_dir:
        print(
            "ERROR: set CHEMVL_DATA_ROOT or pass --local-dir (your ChemVL data root).",
            file=sys.stderr,
        )
        return 2

    print(
        "[WIP] Hub snapshot/LFS may be incomplete or flaky; see README optional Quark mirror if needed.",
        flush=True,
    )
    kwargs = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "local_dir": local_dir,
        "revision": args.revision,
    }
    if args.allow_patterns:
        kwargs["allow_patterns"] = list(args.allow_patterns)
    path = snapshot_download(**kwargs)
    print(f"Snapshot at: {path}")
    print(f"Set CHEMVL_DATA_ROOT (and config paths) to: {os.path.abspath(local_dir)}")
    print(
        "\nNext: extract .tar.zst archives into the same layout:\n"
        "  python tools/hf_download.py unpack\n"
        "(Archives are under archives/; see docs/data/HF_DATA_MANIFEST.md)"
    )
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    data_root = _data_root_path()
    manifest = _load_manifest(args.manifest)

    for entry in manifest.get("archives", []):
        archive = data_root / entry["archive"]
        if not archive.is_file():
            print(f"[skip] {entry['id']}: {archive}")
            continue
        target = data_root / entry["extract_to"]
        print(f"[unpack] {entry['id']}: {archive} -> {target}/")
        _extract_tar_zst(archive, target)

    if not args.skip_finetuning_ckpts:
        rel = manifest.get("finetuning_ckpts_archive", "archives/checkpoints_finetuning.tar.zst")
        archive = data_root / rel
        if archive.is_file():
            target = data_root / "checkpoints"
            print(f"[unpack] finetuning_ckpts: {archive} -> {target}/")
            _extract_tar_zst(archive, target)
        else:
            print(f"[skip] finetuning_ckpts: {archive}")

    print(f"\nUnpack complete under: {data_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download Hub dataset snapshot into CHEMVL_DATA_ROOT")
    dl.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hub dataset id (default: {DEFAULT_REPO_ID})")
    dl.add_argument(
        "--local-dir",
        default=None,
        help="Target directory = CHEMVL_DATA_ROOT (default: env or utils.path_utils.get_data_root())",
    )
    dl.add_argument("--revision", default=None, help="Optional branch, tag, or commit hash")
    dl.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Limit snapshot to Hub paths matching this glob (repeatable)",
    )
    dl.set_defaults(_handler=cmd_download)

    up = sub.add_parser("unpack", help="Extract archives/*.tar.zst after download")
    up.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    up.add_argument("--skip-finetuning-ckpts", action="store_true")
    up.set_defaults(_handler=cmd_unpack)

    args = parser.parse_args(argv)
    return args._handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
