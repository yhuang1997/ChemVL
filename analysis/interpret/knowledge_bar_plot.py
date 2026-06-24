"""Lightweight knowledge-attention bar plots for interpret showcase (public)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DCairo


def clean_descriptor(name: str, full_name: bool = False) -> str:
    if full_name:
        return name
    if name not in ("NOCount", "RingCount"):
        name = name.replace("Num", "").replace("Count", "")
        if not name.startswith("fr_"):
            name = name[0].upper() + name[1:]
    return name


def _draw_molecule_image(smiles: str, size: int = 420) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    drawer = MolDraw2DCairo(size, size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def save_figure_multiformat(
    fig: plt.Figure,
    stem: Path,
    *,
    export_formats: Sequence[str] = ("png",),
    dpi: int = 150,
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
    for ext in export_formats:
        if ext == "tiff":
            fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, **save_kwargs)
        else:
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi, **save_kwargs)


def plot_sample_knowledge_attention_bar(
    smiles: str,
    bar_labels: Sequence[str],
    bar_values: Sequence[float],
    *,
    top_k: int,
    save_stem: Path | None = None,
    export_formats: Sequence[str] = ("png",),
    figsize_cm: Tuple[float, float] = (18.0, 6.0),
    dpi: int = 150,
    sample_idx: int | None = None,
    label: int | None = None,
) -> plt.Figure:
    """Structure + top-K relative-importance bar chart for showcase output."""
    labels = list(bar_labels)
    values = list(bar_values)
    if len(labels) != len(values):
        raise ValueError("bar_labels and bar_values must have the same length")
    mol_img = _draw_molecule_image(smiles)

    cm_to_inch = 1 / 2.54
    base_width, base_height = figsize_cm[0] * cm_to_inch, figsize_cm[1] * cm_to_inch
    height = max(base_height, 0.45 * top_k + 1.8)
    fig = plt.figure(figsize=(base_width, height), facecolor="white")
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 1.35],
        wspace=0.35,
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.12,
    )

    ax_mol = fig.add_subplot(gs[0, 0])
    ax_mol.imshow(mol_img)
    ax_mol.set_aspect("equal")
    ax_mol.axis("off")
    title_bits = []
    if sample_idx is not None:
        title_bits.append(f"#{sample_idx}")
    if label is not None:
        title_bits.append(f"label={label}")
    if title_bits:
        ax_mol.set_title(" | ".join(title_bits), fontsize=10, family="Arial", pad=6)

    ax_bar = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(labels))
    bar_color = "#6b84b2"
    edge_color = "#4F81BD"
    bars = ax_bar.barh(
        y_pos,
        values,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.8,
        height=0.72,
        zorder=2,
    )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=9, family="Arial")
    ax_bar.invert_yaxis()
    ax_bar.axvline(0.0, color="#666666", linewidth=0.8, zorder=1)
    ax_bar.set_xlabel("Relative importance vs. average (%)", fontsize=10, family="Arial")
    ax_bar.tick_params(axis="x", labelsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    x_min = min(0.0, min(values) if values else 0.0)
    x_max = max(0.0, max(values) if values else 0.0)
    x_pad = max(3.0, 0.08 * (x_max - x_min + 1.0))
    ax_bar.set_xlim(x_min - x_pad, x_max + x_pad)

    for bar, value in zip(bars, values):
        y = bar.get_y() + bar.get_height() / 2.0
        if value >= 0:
            ax_bar.text(
                value + 0.02 * (x_max - x_min + 1.0),
                y,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=8,
                family="Arial",
                color="#222222",
            )
        else:
            ax_bar.text(
                value - 0.02 * (x_max - x_min + 1.0),
                y,
                f"{value:.1f}",
                va="center",
                ha="right",
                fontsize=8,
                family="Arial",
                color="#222222",
            )

    if save_stem is not None:
        save_figure_multiformat(fig, save_stem, export_formats=export_formats, dpi=dpi)
        plt.close(fig)
    return fig
