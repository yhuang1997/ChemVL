"""Rebuild per-dataset ``summary.csv`` from ablation finetune run directories."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SUMMARY_FIELDS = [
    "runseed",
    "timestamp",
    "exit_code",
    "best_valid_on_test",
    "best_train_on_test",
    "best_valid",
    "best_valid_epoch",
    "best_train_epoch",
    "best_train_loss",
    "epochs",
    "batch_size",
    "lr",
    "run_dir",
]


def resolve_log_base(cfg: Dict[str, Any], repo_root: str) -> str:
    log_base = cfg.get("basic", {}).get("log_dir_base", "results/finetuning")
    if not os.path.isabs(log_base):
        log_base = os.path.normpath(os.path.join(repo_root, log_base))
    return log_base


def dataset_dir_from_cfg(cfg: Dict[str, Any], repo_root: str) -> Path:
    log_base = resolve_log_base(cfg, repo_root)
    version = cfg.get("basic", {}).get("version")
    dataset = cfg.get("dataset", {}).get("dataset")
    if version is None or dataset is None:
        raise ValueError("cfg missing basic.version or dataset.dataset")
    return Path(log_base) / str(version) / str(dataset)


def dataset_summary_path(cfg: Dict[str, Any], repo_root: str) -> Path:
    return dataset_dir_from_cfg(cfg, repo_root) / "summary.csv"


def newest_run_dir(parent: str, after_ts: float) -> Optional[str]:
    if not os.path.isdir(parent):
        return None
    best: Optional[Tuple[float, str]] = None
    for name in os.listdir(parent):
        p = os.path.join(parent, name)
        if not os.path.isdir(p):
            continue
        mt = os.path.getmtime(p)
        if mt >= after_ts - 2.0:
            if best is None or mt > best[0]:
                best = (mt, p)
    return best[1] if best else None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _scalar(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return ""


def _row_from_run_dir(run_dir: Path, *, exit_code: int = 0) -> Optional[Dict[str, Any]]:
    cfg = _read_json(run_dir / "config.json")
    result = _read_json(run_dir / "result.json")
    if cfg is None:
        return None

    runseed = cfg.get("training", {}).get("runseed")
    if runseed is None:
        return None

    training = cfg.get("training") or {}
    row: Dict[str, Any] = {
        "runseed": int(runseed),
        "timestamp": run_dir.name,
        "exit_code": int(exit_code),
        "best_valid_on_test": "",
        "best_train_on_test": "",
        "best_valid": "",
        "best_valid_epoch": "",
        "best_train_epoch": "",
        "best_train_loss": "",
        "epochs": training.get("epochs", ""),
        "batch_size": training.get("batch_size", ""),
        "lr": training.get("lr", ""),
        "run_dir": str(run_dir),
        "_mtime": run_dir.stat().st_mtime,
    }
    if result is not None:
        row["best_valid_on_test"] = _scalar(result.get("best_valid_on_test"))
        row["best_train_on_test"] = _scalar(result.get("best_train_on_test"))
        row["best_valid"] = _scalar(result.get("best_valid"))
        row["best_valid_epoch"] = _scalar(result.get("best_valid_epoch"))
        row["best_train_epoch"] = _scalar(result.get("best_train_epoch"))
        row["best_train_loss"] = _scalar(result.get("best_train_loss"))
        if exit_code == 0:
            row["exit_code"] = 0
    return row


def scan_dataset_runs(dataset_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not dataset_dir.is_dir():
        return rows
    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "result.json").is_file():
            continue
        row = _row_from_run_dir(entry, exit_code=0)
        if row is not None:
            rows.append(row)
    return rows


def pick_latest_per_runseed(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seed: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        runseed = row.get("runseed")
        if runseed is None:
            continue
        prev = by_seed.get(int(runseed))
        if prev is None or row["_mtime"] > prev["_mtime"]:
            by_seed[int(runseed)] = row
    return [by_seed[k] for k in sorted(by_seed)]


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def refresh_dataset_summary(
    cfg: Dict[str, Any],
    repo_root: str,
    *,
    exit_code: int = 0,
    after_ts: Optional[float] = None,
) -> Path:
    """Rescan ``{version}/{dataset}/`` and rewrite ``summary.csv``."""
    dataset_dir = dataset_dir_from_cfg(cfg, repo_root)
    summary_path = dataset_dir / "summary.csv"
    rows = scan_dataset_runs(dataset_dir)

    if exit_code != 0 and after_ts is not None:
        failed_dir = newest_run_dir(str(dataset_dir), after_ts)
        if failed_dir is not None and not os.path.isfile(os.path.join(failed_dir, "result.json")):
            failed_row = _row_from_run_dir(Path(failed_dir), exit_code=exit_code)
            if failed_row is not None:
                rows.append(failed_row)

    latest_rows = pick_latest_per_runseed(rows)
    for row in latest_rows:
        row.pop("_mtime", None)
    write_summary_csv(summary_path, latest_rows)
    return summary_path
