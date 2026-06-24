#!/usr/bin/env python3
"""
Compare finetuning *trajectories* across ablation groups (same layout as
``ablation_study_analyze.py``): ``<root>/<group_prefix>*/<dataset>/<timestamp>/``.

A run is included only if its directory contains ``*_history.csv`` (prefers
``train_val_test_history.csv``). For each dataset, plots train/valid/test curves
with mean ± std over runs per group.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def find_history_csv(run_dir: Path) -> Optional[Path]:
    preferred = run_dir / "train_val_test_history.csv"
    if preferred.is_file():
        return preferred
    matches = sorted(run_dir.glob("*_history.csv"))
    return matches[0] if matches else None


def pick_trajectory_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cols = list(df.columns)
    train_col = "train_step_loss" if "train_step_loss" in cols else None

    def first_metric(prefix: str, loss_fallback: Optional[str]) -> Optional[str]:
        candidates = [c for c in cols if c.startswith(prefix) and "loss" not in c.lower()]
        order = ["rocauc", "rmse", "mae"]
        for key in order:
            for c in candidates:
                if key in c.lower():
                    return c
        if candidates:
            return candidates[0]
        if loss_fallback and loss_fallback in cols:
            return loss_fallback
        return None

    valid_col = first_metric("valid_", "val_loss")
    test_col = first_metric("test_", "test_loss")
    return train_col, valid_col, test_col


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"No 'epoch' column in {path}")
    df = df.sort_values("epoch").reset_index(drop=True)
    return df


def aggregate_epoch_curves(
    dfs: Sequence[pd.DataFrame], value_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from collections import defaultdict

    bucket: Dict[int, List[float]] = defaultdict(list)
    for df in dfs:
        if value_col not in df.columns:
            continue
        for _, row in df.iterrows():
            e = int(row["epoch"])
            v = row[value_col]
            if pd.notna(v):
                bucket[e].append(float(v))
    if not bucket:
        return np.array([]), np.array([]), np.array([])
    epochs = np.array(sorted(bucket.keys()))
    means = np.array([np.mean(bucket[e]) for e in epochs])
    stds = np.array(
        [np.std(bucket[e], ddof=0) if len(bucket[e]) > 1 else 0.0 for e in epochs]
    )
    return epochs, means, stds


def scan_histories(
    root: Path,
    group_prefix: str,
    datasets: List[str],
    only_groups: Optional[set],
) -> List[Dict]:
    rows: List[Dict] = []
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    for gdir in sorted(root.iterdir()):
        if not gdir.is_dir() or not gdir.name.startswith(group_prefix):
            continue
        gname = gdir.name
        short = display_group_name(gname, group_prefix)
        if only_groups is not None and gname not in only_groups and short not in only_groups:
            continue

        for ds in datasets:
            dpath = gdir / ds
            if not dpath.is_dir():
                continue
            for run_dir in sorted(dpath.iterdir()):
                if not run_dir.is_dir():
                    continue
                hist = find_history_csv(run_dir)
                if hist is None:
                    continue
                try:
                    df = load_history(hist)
                except Exception as ex:
                    print(f"Skip (read error) {hist}: {ex}", file=sys.stderr)
                    continue
                train_c, valid_c, test_c = pick_trajectory_columns(df)
                if valid_c is None and train_c is None:
                    print(f"Skip (no plottable columns): {hist}", file=sys.stderr)
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
                        "group": gname,
                        "group_label": short,
                        "dataset": ds,
                        "run_dir": str(run_dir),
                        "timestamp": run_dir.name,
                        "runseed": runseed,
                        "history_path": str(hist),
                        "df": df,
                        "train_col": train_c,
                        "valid_col": valid_c,
                        "test_col": test_c,
                    }
                )
    return rows


def plot_dataset(
    subset: pd.DataFrame,
    dataset: str,
    out_path: Path,
    title_suffix: str,
) -> bool:
    groups = sorted(subset["group_label"].unique())
    if not groups:
        return False

    def _first_unique(series: pd.Series) -> Optional[str]:
        u = series.dropna().unique()
        return str(u[0]) if len(u) else None

    train_col = _first_unique(subset["train_col"])
    valid_col = _first_unique(subset["valid_col"])
    test_col = _first_unique(subset["test_col"])
    if valid_col is None and train_col is None:
        return False

    def _ylabel(col: str) -> str:
        if col.startswith("valid_"):
            return col[len("valid_") :]
        if col.startswith("test_"):
            return col[len("test_") :]
        return col

    panels: List[Tuple[str, Optional[str], str]] = []
    if train_col:
        panels.append(("Training (step loss)", train_col, "loss"))
    if valid_col:
        panels.append(("Validation", valid_col, _ylabel(valid_col)))
    if test_col:
        panels.append(("Test", test_col, _ylabel(test_col)))

    if not panels:
        return False

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3.2 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    color_list = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
    for ax_idx, (title, col, y_hint) in enumerate(panels):
        ax = axes[ax_idx]
        for gi, grp in enumerate(groups):
            gsub = subset[subset["group_label"] == grp]
            dfs = [r["df"] for _, r in gsub.iterrows()]
            dfs = [d for d in dfs if col in d.columns]
            if not dfs:
                continue
            ep, mu, sig = aggregate_epoch_curves(dfs, col)
            if ep.size == 0:
                continue
            color = color_list[gi % len(color_list)]
            ax.plot(ep, mu, label=grp, color=color, linewidth=2)
            if np.any(sig > 0):
                ax.fill_between(ep, mu - sig, mu + sig, color=color, alpha=0.18)
        ax.set_ylabel(y_hint)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("epoch")
    fig.suptitle(f"{dataset} — {title_suffix}", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


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
        help="Only group directories whose name starts with this prefix.",
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
        help="Basename prefix: default --out-dir is <root>/<stem>_trajectories; PNGs are <stem>_trajectory_<dataset>.png.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs and index CSV (default: <root>/<stem>_trajectories).",
    )
    p.add_argument(
        "--title-suffix",
        default="ablation comparison (mean ± std over runs)",
        help="Subtitle for figures.",
    )
    args = p.parse_args()

    root = args.root.resolve()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    only_groups: Optional[set] = None
    if args.groups:
        only_groups = {x.strip() for x in args.groups.split(",") if x.strip()}

    stem = args.out_stem.strip() or "ablation_study"
    out_dir = args.out_dir or (root / f"{stem}_trajectories")

    rows = scan_histories(root, args.group_prefix, datasets, only_groups)
    if not rows:
        print(
            f"No runs with *_history.csv under {root} (prefix={args.group_prefix!r}).",
            file=sys.stderr,
        )
        return 1

    meta = pd.DataFrame(
        [
            {
                "group": r["group"],
                "group_label": r["group_label"],
                "dataset": r["dataset"],
                "run_dir": r["run_dir"],
                "timestamp": r["timestamp"],
                "runseed": r["runseed"],
                "history_path": r["history_path"],
                "train_col": r["train_col"],
                "valid_col": r["valid_col"],
                "test_col": r["test_col"],
            }
            for r in rows
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{stem}_trajectory_run_index.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Wrote run index: {meta_path}")

    df_all = pd.DataFrame(rows)

    n_plots = 0
    for ds in datasets:
        sub = df_all[df_all["dataset"] == ds]
        if sub.empty:
            print(f"No history runs for dataset {ds!r}; skip plot.", file=sys.stderr)
            continue
        out_png = out_dir / f"{stem}_trajectory_{ds}.png"
        if plot_dataset(sub, ds, out_png, args.title_suffix):
            print(f"Wrote {out_png}")
            n_plots += 1

    if n_plots == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
