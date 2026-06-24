#!/usr/bin/env python3
"""
Aggregate finetuning metrics for ablation-style runs: each group folder (``basic.version``)
contains dataset subfolders with timestamped runs (``config.json`` + ``result.json``).

Computes per-(group, dataset) mean ± std over runs, macro mean across datasets,
and saves CSVs plus a bar chart. Same layout as the legacy
``analyze_pretraining_ablation_study.py``; use ``--group-prefix`` and ``--root`` to
match your experiment tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_group_name(group_dir_name: str, prefix: str) -> str:
    if group_dir_name.startswith(prefix):
        rest = group_dir_name[len(prefix) :]
        return rest.lstrip("_") or group_dir_name
    return group_dir_name


def extract_test_metric(
    result: Dict[str, Any],
    fallback_valid: bool,
) -> Tuple[Optional[str], Optional[float]]:
    """``extensive_finetune`` stores a scalar float; ``02_finetune`` may store a metric dict."""
    avt = result.get("best_valid_on_test")
    if isinstance(avt, (int, float)) and not isinstance(avt, bool):
        return "best_valid_on_test", float(avt)
    if isinstance(avt, dict):
        for key in ("ROCAUC", "MAE", "RMSE", "rocauc", "mae", "rmse"):
            if key in avt and avt[key] is not None:
                name = key.lower()
                return name, float(avt[key])
    if fallback_valid:
        bv = result.get("best_valid")
        if isinstance(bv, (int, float)) and not isinstance(bv, bool):
            return "best_valid", float(bv)
        if isinstance(bv, dict):
            for key in ("ROCAUC", "MAE", "RMSE", "rocauc", "mae", "rmse"):
                if key in bv and bv[key] is not None:
                    return key.lower(), float(bv[key])
    return None, None


def scan_runs(
    root: Path,
    group_prefix: str,
    datasets: List[str],
    fallback_valid: bool,
    only_groups: Optional[set] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    for gdir in sorted(root.iterdir()):
        if not gdir.is_dir() or not gdir.name.startswith(group_prefix):
            continue
        group = gdir.name
        if only_groups is not None:
            short = display_group_name(group, group_prefix)
            if group not in only_groups and short not in only_groups:
                continue
        for ds in datasets:
            dpath = gdir / ds
            if not dpath.is_dir():
                continue
            for run_dir in sorted(dpath.iterdir()):
                if not run_dir.is_dir():
                    continue
                result_path = run_dir / "result.json"
                if not result_path.is_file():
                    continue
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    print(f"Skip invalid JSON: {result_path}", file=sys.stderr)
                    continue
                metric_name, metric_val = extract_test_metric(result, fallback_valid)
                if metric_name is None or metric_val is None:
                    print(
                        f"Skip (no test metric{'/valid' if not fallback_valid else ''}): {result_path}",
                        file=sys.stderr,
                    )
                    continue

                runseed: Optional[int] = None
                cfg_path = run_dir / "config.json"
                if cfg_path.is_file():
                    try:
                        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                        rs = cfg.get("training", {}).get("runseed")
                        if rs is not None:
                            runseed = int(rs)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

                rows.append(
                    {
                        "group": group,
                        "dataset": ds,
                        "run_dir": str(run_dir),
                        "timestamp": run_dir.name,
                        "runseed": runseed,
                        "metric_name": metric_name,
                        "metric_value": metric_val,
                    }
                )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df

    g = (
        df.groupby(["group", "dataset", "metric_name"], as_index=False)["metric_value"]
        .agg(mean="mean", std="std", n="count")
    )
    macro_parts = []
    for grp, sub in g.groupby("group"):
        names = sub["metric_name"].unique()
        if len(names) != 1:
            print(
                f"Warning: group {grp!r} has mixed metric names {list(names)}; check --fallback-valid.",
                file=sys.stderr,
            )
        macro_parts.append(
            {
                "group": grp,
                "macro_mean": sub["mean"].mean(),
                "macro_std_across_datasets": sub["mean"].std(ddof=0),
                "metric_name": names[0] if len(names) == 1 else "mixed",
                "n_datasets": len(sub),
            }
        )
    macro = pd.DataFrame(macro_parts).sort_values("group")
    return g.sort_values(["group", "dataset"]), macro


def plot_grouped_means(per_ds: pd.DataFrame, macro: pd.DataFrame, out_path: Path, title: str) -> None:
    if per_ds.empty:
        return

    mname = per_ds["metric_name"].iloc[0]

    groups = list(per_ds["group"].unique())
    datasets = list(per_ds["dataset"].unique())
    x = np.arange(len(groups))
    n_ds = len(datasets)
    width = min(0.8 / max(n_ds, 1), 0.2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0 = axes[0]
    for i, ds in enumerate(datasets):
        means = []
        errs = []
        for grp in groups:
            row = per_ds[(per_ds["group"] == grp) & (per_ds["dataset"] == ds)]
            if row.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                r = row.iloc[0]
                means.append(r["mean"])
                errs.append(0.0 if pd.isna(r["std"]) else r["std"])
        offset = (i - (n_ds - 1) / 2) * width
        ax0.bar(
            x + offset,
            means,
            width,
            yerr=errs,
            capsize=2,
            label=ds,
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels(groups, rotation=25, ha="right")
    ax0.set_ylabel(f"Test {mname} (mean ± std over runs)")
    ax0.set_title("Per dataset")
    ax0.legend(title="dataset", fontsize=8)
    ax0.grid(axis="y", alpha=0.3)

    ax1 = axes[1]
    if not macro.empty:
        y = macro["macro_mean"].values
        yerr = macro["macro_std_across_datasets"].values
        yerr = np.nan_to_num(yerr, nan=0.0)
        ax1.bar(range(len(macro)), y, yerr=yerr, capsize=3, color="steelblue", alpha=0.85)
        ax1.set_xticks(range(len(macro)))
        ax1.set_xticklabels(macro["group"].tolist(), rotation=25, ha="right")
        ax1.set_ylabel(f"Mean of dataset means ({mname})")
        ax1.set_title("Macro average over datasets")
        ax1.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("results/finetuning"),
        help="Finetuning log root (e.g. results/finetuning/<exp-name>).",
    )
    p.add_argument(
        "--group-prefix",
        default="pretrain_ablation",
        help="Only include group directories whose name starts with this prefix.",
    )
    p.add_argument(
        "--datasets",
        default="bace,bbbp,clintox,tox21",
        help="Comma-separated dataset subfolder names.",
    )
    p.add_argument(
        "--groups",
        default=None,
        help="Optional comma-separated filter by short or full group directory name.",
    )
    p.add_argument(
        "--out-stem",
        default="ablation_study",
        help="Basename for default outputs: <stem>_summary_by_dataset.csv, etc.",
    )
    p.add_argument(
        "--out-csv-detail",
        type=Path,
        default=None,
        help="Override CSV path for per-(group,dataset) summary.",
    )
    p.add_argument(
        "--out-csv-macro",
        type=Path,
        default=None,
        help="Override CSV path for macro summary.",
    )
    p.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="Override PNG path for bar chart.",
    )
    p.add_argument(
        "--fallback-valid",
        action="store_true",
        help="If a run has no best_valid_on_test, use best_valid (validation).",
    )
    args = p.parse_args()

    root = args.root.resolve()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    stem = args.out_stem.strip() or "ablation_study"
    only_groups: Optional[set] = None
    if args.groups:
        only_groups = {x.strip() for x in args.groups.split(",") if x.strip()}

    out_detail = args.out_csv_detail or (root / f"{stem}_summary_by_dataset.csv")
    out_macro = args.out_csv_macro or (root / f"{stem}_summary_macro.csv")
    out_plot = args.out_plot or (root / f"{stem}_summary.png")

    df = scan_runs(root, args.group_prefix, datasets, args.fallback_valid, only_groups)
    if df.empty:
        print(
            f"No scorable runs under {root} (prefix {args.group_prefix!r}). "
            "Directories may exist but each result.json needs a numeric best_valid_on_test "
            "(scalar from extensive_finetune, or dict with ROCAUC/MAE/RMSE from image finetune), "
            "or pass --fallback-valid if only best_valid is set.",
            file=sys.stderr,
        )
        return 1

    per_ds, macro = summarize(df)
    per_ds.to_csv(out_detail, index=False)
    macro.to_csv(out_macro, index=False)
    print(f"Wrote {out_detail}")
    print(f"Wrote {out_macro}")

    print("\n=== Per (group, dataset): mean ± std (n runs) ===")
    for grp in per_ds["group"].unique():
        print(f"\n[{grp}]")
        sub = per_ds[per_ds["group"] == grp]
        for _, r in sub.iterrows():
            std = r["std"]
            std_s = f"{std:.4f}" if pd.notna(std) else "nan"
            print(
                f"  {r['dataset']}: {r['metric_name']} = {r['mean']:.4f} ± {std_s} (n={int(r['n'])})"
            )

    print("\n=== Macro mean over datasets (mean of per-dataset means) ===")
    for _, r in macro.iterrows():
        s = r["macro_std_across_datasets"]
        s_s = f"{s:.4f}" if pd.notna(s) else "nan"
        print(f"  {r['group']}: {r['macro_mean']:.4f} (std across datasets: {s_s}, n_ds={int(r['n_datasets'])})")

    plot_grouped_means(
        per_ds,
        macro,
        out_plot,
        title=f"Finetuning summary ({args.group_prefix}*)",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
