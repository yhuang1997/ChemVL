from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from sklearn.metrics import davies_bouldin_score


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def apply_representation_paper_style(*, hist: bool = False) -> None:
    """Match legacy ``tsne_main`` / ``tsne_playground`` matplotlib look."""
    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            break
        except OSError:
            continue
    if hist:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 14,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
            }
        )


def _cmap_named(name: str):
    try:
        import matplotlib

        if hasattr(matplotlib, "colormaps"):
            return matplotlib.colormaps[name]
    except (AttributeError, KeyError, TypeError):
        pass
    return cm.get_cmap(name)


def _regression_color_limits(
    y: np.ndarray,
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Tuple[float, float]:
    """Robust vmin/vmax for continuous targets (same for global + AC overlay)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(y)
    if not np.any(finite):
        return 0.0, 1.0
    vmin = float(np.percentile(y[finite], percentile_low))
    vmax = float(np.percentile(y[finite], percentile_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        lo = float(np.nanmin(y[finite]))
        hi = float(np.nanmax(y[finite]))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return lo, hi
        return 0.0, 1.0
    return vmin, vmax


@contextlib.contextmanager
def _nature_regression_style() -> Iterator[None]:
    """Journal-style defaults for regression scatter."""
    apply_representation_paper_style(hist=False)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    ):
        yield


def plot_tsne_regression(
    points: np.ndarray,
    targets: np.ndarray,
    out_path: Path,
    title: str,
    *,
    colorbar_label: Optional[str] = None,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    cmap: str = "cividis",
) -> None:
    _ensure_parent(out_path)
    y = np.asarray(targets, dtype=np.float64).ravel()
    vmin, vmax = _regression_color_limits(y, percentile_low=percentile_low, percentile_high=percentile_high)
    norm = Normalize(vmin=vmin, vmax=vmax)

    with _nature_regression_style():
        # Dedicated colorbar column keeps main axes aspect (no horizontal shrink from divider).
        w_main, w_cbar = 5.5, 0.55
        fig = plt.figure(figsize=(w_main + w_cbar, 4.5), facecolor="white")
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[w_main, w_cbar],
            wspace=0.22,
            left=0.1,
            right=0.98,
            top=0.92,
            bottom=0.11,
        )
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax.set_facecolor("#fafafa")
        sc = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=y,
            cmap=cmap,
            s=28,
            alpha=0.85,
            norm=norm,
            edgecolors="white",
            linewidths=0.35,
            rasterized=True,
        )
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        if colorbar_label:
            cbar.set_label(colorbar_label)
        ax.set_title(title, pad=8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        fig.savefig(out_path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(out_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_tsne_regression_ac_highlight(
    points: np.ndarray,
    targets: np.ndarray,
    cliff_mol: np.ndarray,
    out_path: Path,
    title: str,
    *,
    colorbar_label: Optional[str] = None,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    cmap: str = "cividis",
) -> None:
    """
    Same t-SNE layout as global: gray background for non-AC molecules, colored foreground for ``cliff_mol==1``.
    """
    _ensure_parent(out_path)
    y = np.asarray(targets, dtype=np.float64).ravel()
    cliff = np.asarray(cliff_mol, dtype=np.int32).ravel()
    if cliff.shape[0] != points.shape[0] or y.shape[0] != points.shape[0]:
        raise ValueError("cliff_mol, targets, and points must have the same length")
    ac = cliff == 1
    if not np.any(ac):
        return

    vmin, vmax = _regression_color_limits(y, percentile_low=percentile_low, percentile_high=percentile_high)
    norm = Normalize(vmin=vmin, vmax=vmax)

    with _nature_regression_style():
        w_main, w_cbar = 5.5, 0.55
        fig = plt.figure(figsize=(w_main + w_cbar, 4.5), facecolor="white")
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[w_main, w_cbar],
            wspace=0.22,
            left=0.1,
            right=0.98,
            top=0.92,
            bottom=0.11,
        )
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax.set_facecolor("#fafafa")
        bg = ~ac
        if np.any(bg):
            ax.scatter(
                points[bg, 0],
                points[bg, 1],
                c="#b0b0b0",
                s=12,
                alpha=0.35,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )
        ax.scatter(
            points[ac, 0],
            points[ac, 1],
            c=y[ac],
            cmap=cmap,
            s=36,
            alpha=0.92,
            norm=norm,
            edgecolors="white",
            linewidths=0.4,
            rasterized=True,
            zorder=2,
        )
        sm = ScalarMappable(cmap=_cmap_named(cmap), norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        if colorbar_label:
            cbar.set_label(colorbar_label)
        ax.set_title(title, pad=8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        fig.savefig(out_path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(out_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_tsne_regression_alpha_montage(
    panels: Sequence[Tuple[np.ndarray, float]],
    targets: np.ndarray,
    out_base: Path,
    *,
    colorbar_label: Optional[str] = None,
    max_cols: int = 5,
    cliff_mol: Optional[np.ndarray] = None,
    suptitle: Optional[str] = None,
    cmap: str = "cividis",
) -> List[Path]:
    """
    Multi-panel figure: one subplot per ``(points, alpha)`` in ``panels`` order.
    At most ``max_cols`` subplots per row; extra alphas wrap to further rows.
    If ``cliff_mol`` is set, each panel uses AC-highlight style (gray non-cliff + colored AC).
    Writes ``out_base`` + ``.png`` / ``.svg``. Returns written paths.
    """
    _ensure_parent(out_base)
    if not panels:
        return []
    if max_cols < 1:
        max_cols = 5

    y = np.asarray(targets, dtype=np.float64).ravel()
    vmin, vmax = _regression_color_limits(y)
    norm = Normalize(vmin=vmin, vmax=vmax)
    n = len(panels)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    if cliff_mol is not None:
        cliff = np.asarray(cliff_mol, dtype=np.int32).ravel()
        if cliff.shape[0] != y.shape[0]:
            raise ValueError("cliff_mol length must match targets")
        if not np.any(cliff == 1):
            return []

    written: List[Path] = []
    with _nature_regression_style():
        # Extra figure width for a colorbar column so data panels keep ~2.75" × 2.55" each (no right= squeeze).
        w_panel, h_panel = 2.75, 2.55
        cbar_strip = 0.48
        width_ratios = [1.0] * ncols + [cbar_strip / w_panel]
        fig_w = w_panel * ncols + cbar_strip
        fig_h = h_panel * nrows
        top = 0.86 if suptitle else 0.92
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = fig.add_gridspec(
            nrows,
            ncols + 1,
            width_ratios=width_ratios,
            wspace=0.34,
            hspace=0.46,
            left=0.07,
            right=0.98,
            top=top,
            bottom=0.09,
        )
        axes_arr: List[List[Any]] = [[None] * ncols for _ in range(nrows)]

        for i in range(nrows * ncols):
            r, c = divmod(i, ncols)
            ax = fig.add_subplot(gs[r, c])
            axes_arr[r][c] = ax
            if i >= n:
                ax.set_visible(False)
                continue
            pts, alpha = panels[i][0], float(panels[i][1])
            ax.set_facecolor("#fafafa")
            if cliff_mol is None:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    c=y,
                    cmap=cmap,
                    s=16,
                    alpha=0.88,
                    norm=norm,
                    edgecolors="white",
                    linewidths=0.25,
                    rasterized=True,
                )
            else:
                ac = cliff == 1
                bg = ~ac
                if np.any(bg):
                    ax.scatter(
                        pts[bg, 0],
                        pts[bg, 1],
                        c="#b0b0b0",
                        s=8,
                        alpha=0.32,
                        edgecolors="none",
                        rasterized=True,
                        zorder=1,
                    )
                if np.any(ac):
                    ax.scatter(
                        pts[ac, 0],
                        pts[ac, 1],
                        c=y[ac],
                        cmap=cmap,
                        s=22,
                        alpha=0.9,
                        norm=norm,
                        edgecolors="white",
                        linewidths=0.3,
                        rasterized=True,
                        zorder=2,
                    )
            ax.set_title(f"α = {alpha:.2f}", fontsize=9)
            ax.set_xlabel("t-SNE 1", fontsize=8)
            ax.set_ylabel("t-SNE 2", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
            ax.tick_params(labelsize=7)

        sm = ScalarMappable(cmap=_cmap_named(cmap), norm=norm)
        sm.set_array([])
        if suptitle:
            fig.suptitle(suptitle, fontsize=11, y=0.995)
        cax = fig.add_subplot(gs[:, ncols])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        if colorbar_label:
            cbar.set_label(colorbar_label)
        png_p = out_base.with_suffix(".png")
        svg_p = out_base.with_suffix(".svg")
        fig.savefig(png_p, dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(svg_p, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.extend([png_p, svg_p])

    return written


# Default marker for cluster-trajectory highlight (single-molecule montages).
# ``*`` is a filled star; pair with a contrasting edge (default crimson) for visibility.
CLUSTER_TRACK_MARKER_DEFAULT = "*"


def _cluster_track_marker_edge(marker_edgecolor: Any) -> Any:
    """Default red outline for filled markers; pass ``None`` / empty for ``crimson``."""
    if marker_edgecolor is None:
        return "crimson"
    if isinstance(marker_edgecolor, str) and not marker_edgecolor.strip():
        return "crimson"
    return marker_edgecolor


def plot_tsne_regression_marker_track_montage(
    panels: Sequence[Tuple[np.ndarray, float]],
    targets: np.ndarray,
    chosen_indices: Sequence[int],
    out_base: Path,
    *,
    track_marker: str = CLUSTER_TRACK_MARKER_DEFAULT,
    marker_alpha: float = 1.0,
    marker_size: float = 130.0,
    marker_edgecolor: Any = None,
    colorbar_label: Optional[str] = None,
    max_cols: int = 5,
    cliff_mol: Optional[np.ndarray] = None,
    suptitle: Optional[str] = None,
    cmap: str = "cividis",
    color_by: str = "target",
) -> List[Path]:
    """
    Like ``plot_tsne_regression_alpha_montage`` but overlays tracked molecules with one
    marker (default filled ``*`` star) and a high-contrast edge (default ``crimson``).
    ``marker_alpha`` is the **face** opacity in ``[0, 1]`` (default **1** = fully opaque so
    label colors read clearly). Unfilled ``x`` uses the same opacity on its stroke color.
    """
    _ensure_parent(out_base)
    if not panels:
        return []
    if max_cols < 1:
        max_cols = 5

    y = np.asarray(targets, dtype=np.float64).ravel()
    chosen = [int(x) for x in chosen_indices]
    if not chosen:
        return []

    mk = str(track_marker)

    vmin, vmax = _regression_color_limits(y)
    norm = Normalize(vmin=vmin, vmax=vmax)
    n = len(panels)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    if cliff_mol is not None:
        cliff = np.asarray(cliff_mol, dtype=np.int32).ravel()
        if cliff.shape[0] != y.shape[0]:
            raise ValueError("cliff_mol length must match targets")
        if not np.any(cliff == 1):
            return []

    color_by_l = str(color_by).strip().lower()
    if color_by_l not in ("target", "molecule"):
        color_by_l = "target"
    mol_cmap = _cmap_named("tab10") if color_by_l == "molecule" else None

    written: List[Path] = []
    with _nature_regression_style():
        w_panel, h_panel = 2.75, 2.55
        cbar_strip = 0.48
        width_ratios = [1.0] * ncols + [cbar_strip / w_panel]
        fig_w = w_panel * ncols + cbar_strip
        fig_h = h_panel * nrows
        top = 0.84 if suptitle else 0.92
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = fig.add_gridspec(
            nrows,
            ncols + 1,
            width_ratios=width_ratios,
            wspace=0.34,
            hspace=0.46,
            left=0.07,
            right=0.98,
            top=top,
            bottom=0.14,
        )

        cliff_arr: Optional[np.ndarray] = None
        if cliff_mol is not None:
            cliff_arr = np.asarray(cliff_mol, dtype=np.int32).ravel()

        sm_target = ScalarMappable(cmap=_cmap_named(cmap), norm=norm)
        fa = float(np.clip(marker_alpha, 0.0, 1.0))
        edge_hl = _cluster_track_marker_edge(marker_edgecolor)

        for i in range(nrows * ncols):
            r, c = divmod(i, ncols)
            ax = fig.add_subplot(gs[r, c])
            if i >= n:
                ax.set_visible(False)
                continue
            pts, alpha = panels[i][0], float(panels[i][1])
            ax.set_facecolor("#fafafa")
            if cliff_arr is None:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    c=y,
                    cmap=cmap,
                    s=16,
                    alpha=0.88,
                    norm=norm,
                    edgecolors="white",
                    linewidths=0.25,
                    rasterized=True,
                    zorder=1,
                )
            else:
                ac = cliff_arr == 1
                bg = ~ac
                if np.any(bg):
                    ax.scatter(
                        pts[bg, 0],
                        pts[bg, 1],
                        c="#b0b0b0",
                        s=8,
                        alpha=0.32,
                        edgecolors="none",
                        rasterized=True,
                        zorder=1,
                    )
                if np.any(ac):
                    ax.scatter(
                        pts[ac, 0],
                        pts[ac, 1],
                        c=y[ac],
                        cmap=cmap,
                        s=22,
                        alpha=0.9,
                        norm=norm,
                        edgecolors="white",
                        linewidths=0.3,
                        rasterized=True,
                        zorder=2,
                    )

            for rank, mol_i in enumerate(chosen):
                if color_by_l == "molecule":
                    base = mol_cmap(int(rank) % 10)
                    rgb = (float(base[0]), float(base[1]), float(base[2]))
                else:
                    r0, g0, b0, _ = sm_target.to_rgba(float(y[mol_i]))
                    rgb = (float(r0), float(g0), float(b0))
                if mk.lower() == "x":
                    # Unfilled markers use facecolor for the stroke; edgecolors are ignored (see MPL warning).
                    fc_x = (rgb[0], rgb[1], rgb[2], fa)
                    ax.scatter(
                        [pts[mol_i, 0]],
                        [pts[mol_i, 1]],
                        marker="x",
                        s=marker_size * 2.25,
                        facecolors=[fc_x],
                        linewidths=2.0,
                        zorder=8,
                    )
                else:
                    fc = (rgb[0], rgb[1], rgb[2], fa)
                    sz = float(marker_size) * (1.12 if mk == "*" else 1.0)
                    ax.scatter(
                        [pts[mol_i, 0]],
                        [pts[mol_i, 1]],
                        marker=mk,
                        s=sz,
                        facecolors=[fc],
                        edgecolors=edge_hl,
                        linewidths=1.15,
                        zorder=8,
                    )

            ax.set_title(f"α = {alpha:.2f}", fontsize=9)
            ax.set_xlabel("t-SNE 1", fontsize=8)
            ax.set_ylabel("t-SNE 2", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
            ax.tick_params(labelsize=7)

        sm = ScalarMappable(cmap=_cmap_named(cmap), norm=norm)
        sm.set_array([])
        if suptitle:
            fig.suptitle(suptitle, fontsize=10, y=0.995)
        cax = fig.add_subplot(gs[:, ncols])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        if colorbar_label:
            cbar.set_label(colorbar_label)

        if len(chosen) > 1:
            handles = []
            for rank, mol_i in enumerate(chosen):
                if color_by_l == "molecule":
                    fc0 = mol_cmap(int(rank) % 10)
                    rgb = (float(fc0[0]), float(fc0[1]), float(fc0[2]))
                else:
                    r0, g0, b0, _ = sm_target.to_rgba(float(y[mol_i]))
                    rgb = (float(r0), float(g0), float(b0))
                if mk.lower() == "x":
                    fc_leg = (rgb[0], rgb[1], rgb[2], fa)
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="x",
                            color=fc_leg,
                            markerfacecolor=fc_leg,
                            markeredgecolor=fc_leg,
                            markersize=9,
                            markeredgewidth=1.8,
                            linestyle="None",
                            label=f"r{rank} pt{mol_i}",
                        )
                    )
                else:
                    mfc = (rgb[0], rgb[1], rgb[2], fa)
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker=mk,
                            color="w",
                            markerfacecolor=mfc,
                            markeredgecolor=edge_hl,
                            markeredgewidth=1.1,
                            markersize=9 if mk == "*" else 8,
                            linestyle="None",
                            label=f"r{rank} pt{mol_i}",
                        )
                    )
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(5, len(handles)),
                fontsize=6,
                frameon=False,
                bbox_to_anchor=(0.5, 0.02),
            )

        png_p = out_base.with_suffix(".png")
        svg_p = out_base.with_suffix(".svg")
        fig.savefig(png_p, dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(svg_p, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.extend([png_p, svg_p])

    return written


def plot_downstream_cluster_trajectory(
    *,
    aligned_per_alpha: Sequence[np.ndarray],
    alphas: Sequence[float],
    chosen_indices: np.ndarray,
    smiles: Sequence[str],
    y: np.ndarray,
    out_base: Path,
    fp_cluster_id: int,
    colorbar_label: Optional[str] = None,
    color_by: str = "target",
    context_xy: Optional[np.ndarray] = None,
    cmap: str = "cividis",
) -> List[Path]:
    """
    Static trajectory plot: Procrustes-aligned t-SNE coordinates across ``alphas``.
    Earlier α segments / markers are drawn with lower opacity (``history`` emphasis).
    """
    _ensure_parent(out_base)
    if len(aligned_per_alpha) != len(alphas) or len(alphas) < 1:
        return []
    n_alpha = len(alphas)
    y_all = np.asarray(y, dtype=np.float64).ravel()
    chosen = np.asarray(chosen_indices, dtype=np.int64).ravel()
    n_mol = int(chosen.size)
    if n_mol < 1:
        return []

    if color_by == "molecule":
        mol_cmap = _cmap_named("tab10")
        mol_colors = [mol_cmap(int(r) % 10) for r in range(n_mol)]
    else:
        y_sel = y_all[chosen]
        vmin, vmax = _regression_color_limits(y_sel if y_sel.size else y_all)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm_line = ScalarMappable(cmap=_cmap_named(cmap), norm=norm)

    written: List[Path] = []
    with _nature_regression_style():
        fig, ax = plt.subplots(figsize=(6.0, 4.8), facecolor="white")
        ax.set_facecolor("#fafafa")
        ctx = np.asarray(context_xy if context_xy is not None else aligned_per_alpha[-1], dtype=np.float64)
        ax.scatter(
            ctx[:, 0],
            ctx[:, 1],
            c="#c8c8c8",
            s=10,
            alpha=0.22,
            edgecolors="none",
            rasterized=True,
            zorder=0,
        )

        for mi, mol_i in enumerate(chosen.tolist()):
            if color_by == "molecule":
                base_c = mol_colors[mi]
            else:
                base_c = sm_line.to_rgba(float(y_all[mol_i]))

            xs = [float(aligned_per_alpha[t][mol_i, 0]) for t in range(n_alpha)]
            ys = [float(aligned_per_alpha[t][mol_i, 1]) for t in range(n_alpha)]

            if n_alpha > 1:
                for seg in range(n_alpha - 1):
                    t0, t1 = seg, seg + 1
                    frac = float(t1) / float(n_alpha - 1)
                    seg_alpha = 0.12 + 0.88 * frac
                    ax.plot(
                        [xs[t0], xs[t1]],
                        [ys[t0], ys[t1]],
                        color=base_c,
                        alpha=seg_alpha,
                        linewidth=1.35,
                        solid_capstyle="round",
                        zorder=1 + mi,
                    )

            for t in range(n_alpha):
                frac = float(t) / float(max(n_alpha - 1, 1))
                pt_alpha = 0.18 + 0.82 * frac
                face = (
                    [base_c]
                    if color_by == "molecule"
                    else [sm_line.to_rgba(float(y_all[mol_i]))]
                )
                ax.scatter(
                    [xs[t]],
                    [ys[t]],
                    facecolors=face,
                    s=26 if t == n_alpha - 1 else 18,
                    alpha=pt_alpha,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=10 + mi + t * 0.01,
                )

        note = (
            "Trajectories: t-SNE per α, full-cloud Procrustes aligned to α=0 "
            f"(fp_dataset cluster {fp_cluster_id}, AC mols only; opacity → α)."
        )
        ax.set_title(f"{out_base.name}\n{note}", fontsize=9)
        ax.set_xlabel("t-SNE 1 (aligned)")
        ax.set_ylabel("t-SNE 2 (aligned)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        if color_by != "molecule":
            sm = ScalarMappable(cmap=_cmap_named(cmap), norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
            if colorbar_label:
                cbar.set_label(colorbar_label)
        else:
            for mi, mol_i in enumerate(chosen.tolist()):
                ax.scatter([], [], c=[mol_colors[mi]], s=20, label=str(smiles[mol_i])[:32], edgecolors="white")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, frameon=False)

        fig.tight_layout()
        png_p = out_base.with_suffix(".png")
        svg_p = out_base.with_suffix(".svg")
        fig.savefig(png_p, dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(svg_p, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.extend([png_p, svg_p])

    return written


def try_write_cluster_trajectory_plotly(
    *,
    aligned_per_alpha: Sequence[np.ndarray],
    alphas: Sequence[float],
    chosen_indices: np.ndarray,
    smiles: Sequence[str],
    y: np.ndarray,
    out_path: Path,
    title: str,
    fp_cluster_id: int,
    color_by: str = "target",
) -> bool:
    """Interactive static Plotly figure (context + per-segment lines with α-linked opacity)."""
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        return False

    _ensure_parent(out_path)
    n_alpha = len(alphas)
    if n_alpha < 1:
        return False
    y_all = np.asarray(y, dtype=np.float64).ravel()
    chosen = np.asarray(chosen_indices, dtype=np.int64).ravel()
    ctx = np.asarray(aligned_per_alpha[-1], dtype=np.float64)

    if color_by == "molecule":
        mol_cmap = _cmap_named("tab10")
        mol_colors = [mol_cmap(int(i) % 10) for i in range(len(chosen))]
        pal = [f"rgba({int(r[0]*255)},{int(r[1]*255)},{int(r[2]*255)},{r[3]:.3f})" for r in mol_colors]
    else:
        vmin, vmax = _regression_color_limits(y_all[chosen] if len(chosen) else y_all)
        sm = ScalarMappable(cmap=_cmap_named("cividis"), norm=Normalize(vmin=vmin, vmax=vmax))

    def mol_color(mi: int, mol_i: int) -> str:
        if color_by == "molecule":
            return pal[mi]
        r, g, b, a = sm.to_rgba(float(y_all[mol_i]))
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ctx[:, 0],
            y=ctx[:, 1],
            mode="markers",
            marker=dict(size=4, color="lightgray", opacity=0.22),
            name="context (last α)",
            hoverinfo="skip",
        )
    )

    if n_alpha > 1:
        for mi, mol_i in enumerate(chosen.tolist()):
            col = mol_color(mi, mol_i)
            for seg in range(n_alpha - 1):
                t0, t1 = seg, seg + 1
                frac = float(t1) / float(n_alpha - 1)
                op = 0.12 + 0.88 * frac
                x0 = float(aligned_per_alpha[t0][mol_i, 0])
                y0 = float(aligned_per_alpha[t0][mol_i, 1])
                x1 = float(aligned_per_alpha[t1][mol_i, 0])
                y1 = float(aligned_per_alpha[t1][mol_i, 1])
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(color=col, width=2.5),
                        opacity=op,
                        showlegend=False,
                        hovertemplate=(
                            f"SMILES={str(smiles[mol_i])}<br>α {float(alphas[t0]):.2f}→{float(alphas[t1]):.2f}"
                            f"<br>y={y_all[mol_i]:.4f}<br>fp_cluster={fp_cluster_id}<extra></extra>"
                        ),
                    )
                )

    for mi, mol_i in enumerate(chosen.tolist()):
        col = mol_color(mi, mol_i)
        xs = [float(aligned_per_alpha[t][mol_i, 0]) for t in range(n_alpha)]
        ys = [float(aligned_per_alpha[t][mol_i, 1]) for t in range(n_alpha)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=8, color=col, line=dict(width=0.5, color="white"), opacity=0.9),
                name=str(smiles[mol_i])[:48],
                hovertemplate=f"SMILES={str(smiles[mol_i])}<br>y={y_all[mol_i]:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis_title="t-SNE 1 (aligned)",
        yaxis_title="t-SNE 2 (aligned)",
        template="plotly_white",
        height=520,
        width=680,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return True


def plot_tsne_classification(points: np.ndarray, targets: np.ndarray, out_path: Path, title: str) -> float:
    _ensure_parent(out_path)
    apply_representation_paper_style(hist=False)
    db = float(davies_bouldin_score(points, targets))
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.set_facecolor("#eef2f3")
    cmap = ListedColormap(["green", "red"])
    ax.scatter(points[:, 0], points[:, 1], c=targets, cmap=cmap, s=50, alpha=0.8, edgecolors="white", linewidths=1)
    ax.text(0.95, 0.05, f"DB Index: {db:.3f}", fontsize=12, ha="right", va="bottom", transform=ax.transAxes)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, which="both", color="#d5d8dc", linestyle="-", linewidth=0.5, alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=600)
    plt.savefig(out_path.with_suffix(".svg"), dpi=300)
    plt.close()
    return db


def _extract_closed_regions_with_boundary_handling(
    path: MplPath, x_min: float, x_max: float, y_min: float, y_max: float
) -> List[np.ndarray]:
    """Ported from ``visualization-achived/tsne_playground.py``."""
    regions: List[np.ndarray] = []
    current_region: List[np.ndarray] = []
    for vertex, code in zip(path.vertices, path.codes):
        if x_min <= vertex[0] <= x_max and y_min <= vertex[1] <= y_max:
            current_region.append(vertex)
        if code == MplPath.CLOSEPOLY:
            if current_region:
                regions.append(np.array(current_region))
            current_region = []
    if current_region:
        regions.append(np.array(current_region))
    return regions


def _extract_all_regions_with_boundary_handling(
    contour_collection, x_min: float, x_max: float, y_min: float, y_max: float
) -> List[np.ndarray]:
    all_regions_coords: List[np.ndarray] = []
    for contour in contour_collection.collections:
        for path in contour.get_paths():
            regions = _extract_closed_regions_with_boundary_handling(path, x_min, x_max, y_min, y_max)
            all_regions_coords.extend(regions)
    return all_regions_coords


def _get_sample_indices_in_regions(samples: np.ndarray, regions_coords: Sequence[np.ndarray]) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    for region_coords in regions_coords:
        region_path = MplPath(region_coords)
        inside_indices = np.where(region_path.contains_points(samples))[0]
        groups.append(inside_indices)
    return groups


def plot_feature_shift_density(
    base_points: np.ndarray,
    shifted_a: np.ndarray,
    shifted_b: np.ndarray,
    out_path: Path,
    *,
    label_a: str = "Prompt A",
    label_b: str = "Prompt B",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Ported from ``visualize_feature_shifts_with_density`` (density + contours + region grouping)."""
    _ensure_parent(out_path)
    apply_representation_paper_style(hist=False)

    kde_a = gaussian_kde(shifted_a.T)
    kde_b = gaussian_kde(shifted_b.T)

    buffer = 0.05
    x_min = min(
        base_points[:, 0].min(),
        shifted_a[:, 0].min(),
        shifted_b[:, 0].min(),
    )
    x_max = max(
        base_points[:, 0].max(),
        shifted_a[:, 0].max(),
        shifted_b[:, 0].max(),
    )
    y_min = min(
        base_points[:, 1].min(),
        shifted_a[:, 1].min(),
        shifted_b[:, 1].min(),
    )
    y_max = max(
        base_points[:, 1].max(),
        shifted_a[:, 1].max(),
        shifted_b[:, 1].max(),
    )
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= buffer * x_range
    x_max += buffer * x_range
    y_min -= buffer * y_range
    y_max += buffer * y_range

    x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z_a = kde_a(grid_coords).reshape(x_grid.shape)
    z_b = kde_b(grid_coords).reshape(x_grid.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        base_points[:, 0],
        base_points[:, 1],
        color="grey",
        alpha=0.3,
        s=10,
        label="Base features",
    )
    ax.scatter(
        shifted_a[:, 0],
        shifted_a[:, 1],
        color="green",
        alpha=0.35,
        s=10,
        label=label_a,
    )
    ax.scatter(
        shifted_b[:, 0],
        shifted_b[:, 1],
        color="red",
        alpha=0.35,
        s=10,
        label=label_b,
    )

    contours_a = ax.contour(
        x_grid,
        y_grid,
        z_a,
        levels=1,
        colors="green",
        linestyles="dashed",
        alpha=0.45,
        linewidths=1.2,
    )
    contours_b = ax.contour(
        x_grid,
        y_grid,
        z_b,
        levels=1,
        colors="red",
        linestyles="dashed",
        alpha=0.45,
        linewidths=1.2,
    )

    regions_a = _extract_all_regions_with_boundary_handling(contours_a, x_min, x_max, y_min, y_max)
    regions_b = _extract_all_regions_with_boundary_handling(contours_b, x_min, x_max, y_min, y_max)
    groups_a = _get_sample_indices_in_regions(shifted_a, regions_a)
    groups_b = _get_sample_indices_in_regions(shifted_b, regions_b)

    ax.set_title("Feature shifts under different text prompts (density contours)")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".svg"), dpi=300)
    plt.close()

    return groups_a, groups_b


def export_specific_cluster_hist(
    *,
    target_values: np.ndarray,
    groups: List[np.ndarray],
    cmap_name: str,
    out_path: Path,
    descriptor_title: Optional[str] = None,
    alpha_panel_index: Optional[int] = None,
) -> pd.DataFrame:
    """Ported from ``tsne_main.py`` step3 (stacked hists + line overlay + count labels)."""
    _ensure_parent(out_path)
    apply_representation_paper_style(hist=True)

    data = np.asarray(target_values, dtype=float).copy()
    this_target_min = int(np.min(data))
    this_target_max = int(np.max(data))
    clip_upper = 10
    if this_target_max > clip_upper:
        this_target_max = clip_upper
        data = np.clip(data, None, float(clip_upper))
        group_gts_values = [np.clip(np.asarray(target_values, dtype=float)[idx], None, float(clip_upper)) for idx in groups]
    else:
        group_gts_values = [np.asarray(target_values, dtype=float)[idx] for idx in groups]

    bins = np.arange(this_target_min, this_target_max + 2) - 0.5
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    num_groups = len(group_gts_values)
    cmap = _cmap_named(cmap_name)
    colors = [cmap(j) for j in np.linspace(0.3, 0.8, max(1, num_groups))]

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, alpha=0.5, color="white", edgecolor="black", label="All samples")
    plt.plot(bin_centers, counts, marker="o", linestyle="-.", color=colors[-1], label="Overall histogram", lw=1)

    n, bins_arr, _ = plt.hist(
        group_gts_values,
        bins=bins,
        stacked=True,
        color=colors,
        alpha=0.5,
        edgecolor="black",
        label=[f"{cmap_name} {k + 1}" for k in range(num_groups)],
    )

    for u in range(len(n)):
        row = n[u]
        if isinstance(row, np.ndarray):
            for v in range(len(row)):
                if row[v] > 0:
                    plt.text(bins_arr[v] + 0.5, row[v], str(int(row[v])), ha="center", va="bottom")
        else:
            if row > 0:
                plt.text(bins_arr[u] + 0.5, row, str(int(row)), ha="center", va="bottom")

    if this_target_max == clip_upper and int(np.max(np.asarray(target_values, dtype=float))) > clip_upper:
        xtick_labels = [str(int(x)) for x in bin_centers[:-1]] + [f">={clip_upper}"]
        plt.xticks(bin_centers, xtick_labels)

    title_desc = descriptor_title or "descriptor"
    plt.title(f"Distribution of GTS values for {title_desc}")
    plt.legend()
    plt.xlabel("Values")
    plt.ylabel("Cumulative count")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".svg"), dpi=300)
    plt.close()

    rows: List[Dict[str, Any]] = []
    for gi, arr in enumerate(np.atleast_2d(n)):
        for bi, count in enumerate(arr):
            rows.append(
                {
                    "group_id": gi,
                    "bin_left": float(bins_arr[bi]),
                    "bin_right": float(bins_arr[bi + 1]),
                    "count": float(count),
                }
            )
    df = pd.DataFrame(rows)
    if descriptor_title is not None:
        df["descriptor_title"] = descriptor_title
    if alpha_panel_index is not None:
        df["alpha_panel_index"] = int(alpha_panel_index)
    return df


def plot_tsne_clusters(
    points: np.ndarray,
    cluster_ids: np.ndarray,
    out_path: Path,
    *,
    title: str,
    noise_label: str = "noise",
    colorbar_label: str = "cluster_id",
) -> None:
    """Scatter points colored by discrete cluster ids (e.g. ``cluster_id`` or ``fp_dataset_cluster_id``)."""
    _ensure_parent(out_path)
    apply_representation_paper_style(hist=False)
    cids = np.asarray(cluster_ids, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(10, 8))
    noise = cids < 0
    if np.any(noise):
        ax.scatter(
            points[noise, 0],
            points[noise, 1],
            c="#bbbbbb",
            s=4,
            alpha=0.35,
            label=noise_label,
            rasterized=True,
        )
    rest = ~noise
    if np.any(rest):
        vals = cids[rest]
        sc = ax.scatter(
            points[rest, 0],
            points[rest, 1],
            c=vals,
            cmap="tab20",
            s=6,
            alpha=0.75,
            rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    if np.any(noise):
        ax.legend(loc="best", markerscale=3)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".svg"), dpi=300)
    plt.close()


def try_write_tsne_plotly_html(
    points: np.ndarray,
    cluster_ids: np.ndarray,
    smiles: List[str],
    out_path: Path,
    *,
    title: str,
    color_column: str = "cluster_id",
) -> bool:
    """Write interactive HTML if ``plotly`` is installed; otherwise return False."""
    try:
        import plotly.express as px  # type: ignore
    except ImportError:
        return False
    _ensure_parent(out_path)
    c = np.asarray(cluster_ids, dtype=np.int64)
    df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            color_column: c,
            "smiles": smiles,
        }
    )
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_column,
        hover_data=["smiles", color_column],
        title=title,
        render_mode="webgl",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return True
