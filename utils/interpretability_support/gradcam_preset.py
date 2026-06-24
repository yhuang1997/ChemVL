"""YAML preset helpers for moleculeace_interpretability Grad-CAM batch runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from utils.interpretability_support.moleculeace_gradcam_batch import (
    MoleculeAceGradcamConfig,
    run_moleculeace_gradcam,
)

RUN_PATH_KEYS = frozenset({"log_dir", "ckpt"})
CKPT_VARIANTS = frozenset({"valid_best", "train_best", "ckpt"})


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in RUN_PATH_KEYS:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_paths_from_run_entry(
    dataset: str,
    run_entry: Dict[str, Any],
    global_cfg: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Resolve log_dir and ckpt from explicit paths in the run entry."""
    log_dir_raw = run_entry.get("log_dir")
    if not log_dir_raw:
        raise ValueError(f"{dataset}: missing runs[{dataset!r}].log_dir")
    log_dir = Path(str(log_dir_raw)).expanduser().resolve()

    ckpt_raw = run_entry.get("ckpt")
    if ckpt_raw:
        ckpt = Path(str(ckpt_raw)).expanduser().resolve()
    else:
        variant = str(run_entry.get("ckpt_variant") or global_cfg.get("ckpt_variant", "valid_best"))
        if variant not in CKPT_VARIANTS:
            raise ValueError(
                f"{dataset}: ckpt_variant must be one of {sorted(CKPT_VARIANTS)}, got {variant!r}"
            )
        ckpt = log_dir / f"{variant}.pth"

    return log_dir, ckpt


def validate_gradcam_paths(dataset: str, log_dir: Path, ckpt: Path) -> Optional[str]:
    if not (log_dir / "config.json").is_file():
        return f"missing config.json under log_dir: {log_dir}"
    if not ckpt.is_file():
        return f"missing checkpoint: {ckpt}"
    return None


def load_gradcam_preset(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Preset must be a mapping: {path}")
    if data.get("version") != 1:
        raise ValueError(f"Unsupported preset version (expected 1): {path}")
    return data


def dataset_order(preset: Dict[str, Any], runs: Dict[str, Any]) -> List[str]:
    explicit = preset.get("datasets")
    if explicit:
        return [str(d) for d in explicit]
    return list(runs.keys())


def run_aggregate_summary(repo_root: Path, output_base: Path) -> None:
    agg_script = (
        repo_root / "scripts/experiments/moleculeace_interpretability/run_aggregate_summary.py"
    )
    if not agg_script.is_file() or not output_base.is_dir():
        return
    print("\n--- Aggregating target_summary.csv ---")
    agg_argv = [sys.executable, str(agg_script), "--root", str(output_base)]
    agg_result = subprocess.run(agg_argv, cwd=str(repo_root))
    if agg_result.returncode != 0:
        print("Warning: run_aggregate_summary.py failed", file=sys.stderr)


def run_gradcam_preset(
    preset_path: Path,
    *,
    repo_root: Path,
    dry_run: bool = False,
    only_datasets: Optional[List[str]] = None,
) -> int:
    preset = load_gradcam_preset(preset_path)
    global_cfg: Dict[str, Any] = dict(preset.get("global") or {})
    runs: Dict[str, Dict[str, Any]] = dict(preset.get("runs") or {})

    if not runs:
        print(
            "ERROR: preset must define 'runs' with per-dataset log_dir and ckpt paths",
            file=sys.stderr,
        )
        return 2

    datasets = dataset_order(preset, runs)
    missing_in_runs = [d for d in datasets if d not in runs]
    if missing_in_runs:
        print(f"ERROR: datasets not in runs: {missing_in_runs}", file=sys.stderr)
        return 2

    if only_datasets:
        only_set = set(only_datasets)
        datasets = [d for d in datasets if d in only_set]
        if not datasets:
            print(f"ERROR: no datasets match --only-datasets {only_datasets}", file=sys.stderr)
            return 2

    output_base = Path(
        str(global_cfg.get("output_dir", "results/moleculeace_interpretability"))
    )
    if not output_base.is_absolute():
        output_base = (repo_root / output_base).resolve()
    else:
        output_base = output_base.resolve()

    stop_on_error = os.environ.get("STOP_ON_ERROR", "0") == "1"
    ok: List[str] = []
    failed: List[Tuple[str, str]] = []

    print(f"Preset: {preset_path} ({preset.get('name', preset_path.stem)})")
    print(f"Output base: {output_base}")
    print(f"Datasets: {len(datasets)}")
    if dry_run:
        print("DRY RUN\n")

    for dataset in datasets:
        run_entry = dict(runs[dataset])

        try:
            log_dir, ckpt = resolve_paths_from_run_entry(dataset, run_entry, global_cfg)
        except ValueError as exc:
            print(f"SKIP {dataset}: {exc}", file=sys.stderr)
            failed.append((dataset, str(exc)))
            if stop_on_error:
                break
            continue

        err = validate_gradcam_paths(dataset, log_dir, ckpt)
        if err:
            print(f"SKIP {dataset}: {err}", file=sys.stderr)
            failed.append((dataset, err))
            if stop_on_error:
                break
            continue

        config = MoleculeAceGradcamConfig.from_preset_merge(
            global_cfg,
            run_entry,
            log_dir=log_dir,
            ckpt=ckpt,
            output_dir=output_base,
            preset_path=preset_path,
            preset_dataset=dataset,
        )

        print(f"=== {dataset} ===")
        print(f"  log_dir: {log_dir}")
        print(f"  ckpt:    {ckpt}")
        print(f"  config:  {config.summary_line()}")

        if dry_run:
            ok.append(dataset)
            continue

        os.chdir(repo_root)
        result = run_moleculeace_gradcam(config)
        if result == 0:
            ok.append(dataset)
        else:
            failed.append((dataset, f"exit code {result}"))
            if stop_on_error:
                break

    print("\n--- Summary ---")
    print(f"OK ({len(ok)}): {', '.join(ok) if ok else '(none)'}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for ds, reason in failed:
            print(f"  {ds}: {reason}")

    if failed and (stop_on_error or len(ok) == 0):
        return 1

    if not dry_run and ok:
        run_aggregate_summary(repo_root, output_base)

    return 0
