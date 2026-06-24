#!/usr/bin/env python3
"""
Batch driver for **MolMCL under ChemVL** (``utils/external/finetune_external.py``).

Drives **MoleculeNet** and **MoleculeACE** sweeps (``--dataset-list`` + ``--base-config``),
same layout as ``scripts/moleculeace_batch_run.py`` (datasets × runseeds, skip policy, _repro),
but defaults to the external finetune entry and optional dry-run echo of MolMCL yaml epochs/batch.

ChemVL-only batching stays in ``moleculeace_batch_run.py`` + ``moleculeace_finetune.py``.

Example::

    python scripts/external/molmcl_under_chemvl/batch_run_external.py \\
      --base-config configs/external/molmcl/molmcl_gps_moleculeace.external.json \\
      --dataset-list configs/external/molmcl/dataset_list_moleculeace30.example.txt \\
      --runseed-start 0 --runseed-end 1 \\
      --exp-name molmcl_under_chemvl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = str(_SCRIPT_DIR.parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.argparser import load_config  # noqa: E402
from utils.external.molmcl.moleculenet_io import discover_moleculenet_from_dataroot  # noqa: E402

from scripts.ablation_study_repro import append_run_jsonl, write_repro_bundle  # noqa: E402

SKIP_TRIPLES: Set[Tuple[str, int]] = set()
GROUP_ID = "molmcl_under_chemvl"


def validate_exp_name(name: str) -> str:
    s = name.strip()
    if not s:
        raise ValueError("exp_name must be non-empty (after stripping)")
    if ".." in s or "/" in s or "\\" in s:
        raise ValueError("exp_name must not contain '..' or path separators")
    if os.path.isabs(s):
        raise ValueError("exp_name must be a single relative label, not an absolute path")
    return s


def apply_exp_name(cfg: Dict[str, Any], exp_name: str) -> None:
    basic = cfg.setdefault("basic", {})
    orig = basic.get("log_dir_base", "results/finetuning")
    basic["log_dir_base"] = os.path.normpath(os.path.join(str(orig), exp_name))


def resolve_path(root: str, path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def _output_has_complete_test_result(cfg: Dict[str, Any], repo_root: str) -> bool:
    log_base = cfg.get("basic", {}).get("log_dir_base")
    version = cfg.get("basic", {}).get("version")
    ds_name = cfg.get("dataset", {}).get("dataset")
    want_seed = cfg.get("training", {}).get("runseed")
    if log_base is None or version is None or ds_name is None or want_seed is None:
        return False
    if not os.path.isabs(log_base):
        log_base = os.path.normpath(os.path.join(repo_root, log_base))
    ds_dir = os.path.join(log_base, str(version), str(ds_name))
    if not os.path.isdir(ds_dir):
        return False
    for entry in os.listdir(ds_dir):
        run_dir = os.path.join(ds_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        rpath = os.path.join(run_dir, "result.json")
        cpath = os.path.join(run_dir, "config.json")
        if not os.path.isfile(rpath) or not os.path.isfile(cpath):
            continue
        try:
            with open(cpath, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("training", {}).get("runseed") != want_seed:
                continue
            with open(rpath, encoding="utf-8") as f:
                res = json.load(f)
            if isinstance(res, dict) and "best_valid_on_test" in res:
                return True
        except (json.JSONDecodeError, OSError, TypeError):
            continue
    return False


def should_skip(
    dataset: str,
    runseed: int,
    merged_cfg: Dict[str, Any],
    repo_root: str,
    *,
    skip_if_existing_result: bool = True,
) -> Optional[str]:
    if (dataset, runseed) in SKIP_TRIPLES:
        return "SKIP_TRIPLES"
    if skip_if_existing_result and _output_has_complete_test_result(merged_cfg, repo_root):
        return "existing result.json (best_valid_on_test)"
    return None


def discover_datasets_from_dataroot(dataroot: str) -> List[str]:
    if not dataroot or not os.path.isdir(dataroot):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(dataroot)):
        sub = os.path.join(dataroot, name)
        if not os.path.isdir(sub):
            continue
        csv_path = os.path.join(sub, "processed", f"{name}_processed_ac.csv")
        if os.path.isfile(csv_path):
            out.append(name)
    return out


def load_dataset_list(path: str) -> List[str]:
    names: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(s)
    return names


def load_task_overlay(task_dir: Path, task_key: str) -> Dict[str, Any]:
    cfg_path = task_dir / f"{task_key}.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing task config: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    overlay: Dict[str, Any] = {}
    if raw.get("dataset"):
        ds = dict(raw["dataset"])
        ds.pop("dataroot", None)
        overlay["dataset"] = ds
    if raw.get("regression_scheduler") is not None:
        overlay["regression_scheduler"] = dict(raw["regression_scheduler"])
    return overlay


def run_finetune(
    python_exe: str,
    finetune_script: str,
    config_path: str,
    cwd: str,
    capture_output: bool,
) -> subprocess.CompletedProcess:
    command = [python_exe, finetune_script, "--config", config_path]
    print(f"Executing: {' '.join(command)}")
    if capture_output:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return subprocess.run(command, cwd=cwd)


def print_process_result(result: subprocess.CompletedProcess) -> None:
    if result.returncode == 0:
        print("Finished successfully.")
    else:
        print(f"Failed with exit code {result.returncode}.")
    if result.stdout:
        print("Output:")
        print(result.stdout)
    if result.stderr:
        print("Stderr:")
        print(result.stderr)


def repro_dir_for_cell(root: str, base_config: str, dataset: str, runseed: int, exp_name: str) -> Path:
    cfg = load_config([base_config])
    cfg.setdefault("dataset", {})["dataset"] = dataset
    cfg.setdefault("training", {})["runseed"] = runseed
    apply_exp_name(cfg, exp_name)
    log_base = cfg.get("basic", {}).get("log_dir_base", "results/finetuning")
    if not os.path.isabs(log_base):
        log_base = os.path.normpath(os.path.join(root, log_base))
    return Path(log_base) / "_repro"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--base-config",
        default="configs/external/molmcl/molmcl_gps_moleculeace.external.json",
        help="Base JSON (merged first).",
    )
    p.add_argument(
        "--dataset-list",
        default=None,
        help="Text file: one task key per line (overrides --datasets if set).",
    )
    p.add_argument(
        "--task-dir",
        default=None,
        help="Per-task JSON overlay dir (MoleculeNet batch). If set, merges configs/external/moleculenet/datasets/<key>.json.",
    )
    p.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset ids (used if --dataset-list unset).",
    )
    p.add_argument(
        "--discover",
        action="store_true",
        help="Discover dataset names from dataset.dataroot in base config (see docstring).",
    )
    p.add_argument("--runseed-start", type=int, default=0)
    p.add_argument("--runseed-end", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--capture-output", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--finetune-script",
        default=None,
        help="Default: utils/external/finetune_external.py under repo root.",
    )
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument(
        "--exp-name",
        default="default",
        metavar="STR",
        help="Nest outputs under basic.log_dir_base/STR/ (single segment).",
    )
    p.add_argument("--save-git-diff", action="store_true")
    return p.parse_args()


def resolve_datasets(args: argparse.Namespace, root: str, base_path: str) -> List[str]:
    if args.dataset_list:
        path = resolve_path(root, args.dataset_list)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--dataset-list not found: {path}")
        return load_dataset_list(path)
    ds_arg = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if ds_arg:
        return ds_arg
    if args.discover:
        base_cfg = load_config([base_path])
        dataroot = base_cfg.get("dataset", {}).get("dataroot")
        if not dataroot:
            raise ValueError("Base config missing dataset.dataroot; cannot --discover.")
        dataroot = resolve_path(root, str(dataroot))
        bench = str(base_cfg.get("dataset", {}).get("benchmark", "moleculeace")).lower()
        if bench == "moleculenet":
            mpp = str((base_cfg.get("dataset") or {}).get("moleculenet_mpp_subdir") or "MPP")
            found = discover_moleculenet_from_dataroot(dataroot, mpp_subdir=mpp)
            if not found:
                raise RuntimeError(
                    f"--discover (moleculenet) found no tasks under {dataroot!r} "
                    f"(expected …/{mpp}/classification|regression/<stem>/processed/<stem>_processed_ac.csv)."
                )
        else:
            found = discover_datasets_from_dataroot(dataroot)
            if not found:
                raise RuntimeError(
                    f"--discover found no datasets under {dataroot!r} with processed/*_processed_ac.csv"
                )
        return found
    raise ValueError("Provide --dataset-list, non-empty --datasets, or --discover.")


def main() -> int:
    args = parse_args()
    root = _REPO_ROOT
    finetune_script = args.finetune_script or os.path.join(root, "utils/external/finetune_external.py")
    if not os.path.isabs(finetune_script):
        finetune_script = os.path.normpath(os.path.join(root, finetune_script))
    base_path = resolve_path(root, args.base_config)
    if not os.path.isfile(base_path):
        print(f"Base config not found: {base_path}", file=sys.stderr)
        return 1

    try:
        exp_name = validate_exp_name(args.exp_name)
    except ValueError as e:
        print(f"Invalid --exp-name: {e}", file=sys.stderr)
        return 1

    try:
        datasets = resolve_datasets(args, root, base_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    repro_dir = repro_dir_for_cell(root, base_path, datasets[0], args.runseed_start, exp_name)
    if not args.dry_run:
        write_repro_bundle(repro_dir, sys.argv, root, save_git_diff=args.save_git_diff)

    task_dir: Optional[Path] = None
    if args.task_dir:
        task_dir = Path(resolve_path(root, args.task_dir))
        if not task_dir.is_dir():
            print(f"Task dir not found: {task_dir}", file=sys.stderr)
            return 1

    failures = 0
    for task_key in datasets:
        for runseed in range(args.runseed_start, args.runseed_end + 1):
            if task_dir is not None:
                try:
                    task_overlay = load_task_overlay(task_dir, task_key)
                except FileNotFoundError as e:
                    print(str(e), file=sys.stderr)
                    failures += 1
                    continue
                overlay = dict(task_overlay)
            else:
                overlay = {"dataset": {"dataset": task_key}}
            overlay.setdefault("training", {})["runseed"] = runseed
            fd, tmp_overlay = tempfile.mkstemp(
                prefix=f"molmcl_chemvl_overlay_{task_key}_{runseed}_", suffix=".json"
            )
            os.close(fd)
            try:
                with open(tmp_overlay, "w", encoding="utf-8") as f:
                    json.dump(overlay, f, indent=2)
                cfg = load_config([base_path, tmp_overlay])
                apply_exp_name(cfg, exp_name)

                skip_reason = should_skip(
                    task_key,
                    runseed,
                    cfg,
                    root,
                    skip_if_existing_result=not args.no_skip_existing,
                )
                if skip_reason is not None:
                    print(f"Skip {task_key} / runseed={runseed} ({skip_reason})")
                    continue

                ds_folder = cfg.get("dataset", {}).get("dataset", task_key)
                if args.dry_run:
                    tr = cfg.get("training", {})
                    mol_epochs, mol_bs = None, None
                    if (cfg.get("model") or {}).get("finetune_backend") in (
                        "molmcl_moleculeace",
                        "molmcl_moleculenet",
                    ):
                        try:
                            from utils.external.molmcl.molmcl_external_config import preview_molmcl_epochs_batch

                            mol_epochs, mol_bs = preview_molmcl_epochs_batch(cfg)
                        except (OSError, ValueError, TypeError, KeyError, ImportError):
                            mol_epochs, mol_bs = None, None
                    print(
                        f"[dry-run] task_key={task_key} dataset={ds_folder} runseed={runseed} "
                        f"version={cfg['basic'].get('version')} split={cfg['dataset'].get('split')} "
                        f"log_dir_base={cfg['basic'].get('log_dir_base')} "
                        f"molmcl_yaml_epochs={mol_epochs} molmcl_yaml_batch={mol_bs} "
                        f"task_type={cfg['dataset'].get('task_type')}"
                    )
                    continue

                fd2, tmp_merged = tempfile.mkstemp(
                    prefix=f"molmcl_chemvl_merged_{task_key}_{runseed}_", suffix=".json"
                )
                os.close(fd2)
                try:
                    with open(tmp_merged, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=4)
                    proc = run_finetune(
                        args.python,
                        finetune_script,
                        tmp_merged,
                        cwd=root,
                        capture_output=args.capture_output,
                    )
                    if args.capture_output:
                        print_process_result(proc)
                    if proc.returncode != 0:
                        failures += 1
                    if not args.dry_run:
                        append_run_jsonl(
                            repro_dir,
                            {
                                "group_id": GROUP_ID,
                                "task_key": task_key,
                                "dataset": ds_folder,
                                "runseed": runseed,
                                "base_config": os.path.relpath(base_path, root)
                                if base_path.startswith(root)
                                else base_path,
                                "exit_code": proc.returncode,
                            },
                        )
                finally:
                    try:
                        os.unlink(tmp_merged)
                    except OSError:
                        pass
            finally:
                try:
                    os.unlink(tmp_overlay)
                except OSError:
                    pass

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
