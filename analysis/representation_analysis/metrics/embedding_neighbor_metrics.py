"""Neighborhood Preservation@k (Morgan Tanimoto vs embedding) and Label Smoothness@k."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


def _morgan_fps(smiles: List[str], radius: int, n_bits: int) -> tuple[list, np.ndarray]:
    """Return (fps list with None for invalid), valid mask."""
    fps: List[Any] = []
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            fps.append(None)
            continue
        valid[i] = True
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits)))
    return fps, valid


def _topk_excluding_self(scores: np.ndarray, k: int, self_idx: int) -> np.ndarray:
    """Indices of top-k scores among j != self_idx (stable: higher score first, then lower j)."""
    scores = scores.copy()
    scores[self_idx] = -np.inf
    n = scores.size
    if n - 1 < k:
        k = n - 1
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((np.arange(n), -scores))
    return order[:k].astype(np.int64)


def neighborhood_preservation_at_k(
    smiles: List[str],
    emb: np.ndarray,
    ks: Sequence[int],
    *,
    morgan_radius: int = 2,
    morgan_n_bits: int = 1024,
) -> Dict[str, Any]:
    """
    NP@k = mean_i |N_k^chem(i) ∩ N_k^embed(i)| / k.
    Chemical neighbors: Morgan Tanimoto. Embedding neighbors: cosine similarity on L2-normalized emb.
    """
    out: Dict[str, Any] = {}
    n = len(smiles)
    emb = np.asarray(emb, dtype=np.float64)
    if n < 2:
        for k in ks:
            out[f"np_at_{k}"] = float("nan")
        out["np_n_molecules"] = 0
        return out

    fps, valid = _morgan_fps(smiles, morgan_radius, morgan_n_bits)
    idx_ok = np.where(valid)[0]
    if len(idx_ok) < 2:
        for k in ks:
            out[f"np_at_{k}"] = float("nan")
        out["np_n_molecules"] = int(len(idx_ok))
        return out

    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim_emb = emb_n @ emb_n.T

    k_max = max(ks)
    overlaps = {int(k): [] for k in ks}

    for i in idx_ok:
        t_row = np.zeros(n, dtype=np.float64)
        for j in range(n):
            if i == j or not valid[j]:
                t_row[j] = -1.0
                continue
            fi, fj = fps[i], fps[j]
            if fi is None or fj is None:
                t_row[j] = -1.0
                continue
            t_row[j] = float(DataStructs.TanimotoSimilarity(fi, fj))

        chem_nn = _topk_excluding_self(t_row, k_max, i)
        embed_nn = _topk_excluding_self(sim_emb[i], k_max, i)

        for k in ks:
            chem_k = set(chem_nn[:k].tolist()) if len(chem_nn) >= k else set(chem_nn.tolist())
            emb_k = set(embed_nn[:k].tolist()) if len(embed_nn) >= k else set(embed_nn.tolist())
            if k == 0:
                overlaps[int(k)].append(float("nan"))
            else:
                overlaps[int(k)].append(len(chem_k & emb_k) / float(k))

    out["np_n_molecules"] = int(len(idx_ok))
    for k in ks:
        vals = overlaps[int(k)]
        out[f"np_at_{k}"] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def label_smoothness_at_k(
    emb: np.ndarray,
    y: np.ndarray,
    ks: Sequence[int],
) -> Dict[str, Any]:
    """
    LS@k = (1/N) sum_i (1/k) sum_{j in N_k(i)} |y_i - y_j|,
    N_k(i) = top-k embedding neighbors by cosine similarity (excluding i).
    """
    out: Dict[str, Any] = {}
    n = emb.shape[0]
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if n < 2 or y.shape[0] != n:
        for k in ks:
            out[f"ls_at_{k}"] = float("nan")
        out["ls_n_molecules"] = 0
        return out

    finite = np.isfinite(y)
    if finite.sum() < 2:
        for k in ks:
            out[f"ls_at_{k}"] = float("nan")
        out["ls_n_molecules"] = int(finite.sum())
        return out

    emb_n = emb.astype(np.float64) / (np.linalg.norm(emb.astype(np.float64), axis=1, keepdims=True) + 1e-12)
    sim_emb = emb_n @ emb_n.T

    k_max = max(ks)
    idx_ok = np.where(finite)[0]
    per_k_vals = {int(k): [] for k in ks}

    for i in idx_ok:
        nn = _topk_excluding_self(sim_emb[i], k_max, i)
        yi = float(y[i])
        for k in ks:
            if k <= 0:
                per_k_vals[int(k)].append(float("nan"))
                continue
            take = nn[:k]
            if len(take) < k:
                per_k_vals[int(k)].append(float("nan"))
                continue
            acc = 0.0
            ok = True
            for j in take:
                if not finite[j]:
                    ok = False
                    break
                acc += abs(yi - float(y[j]))
            if not ok:
                per_k_vals[int(k)].append(float("nan"))
            else:
                per_k_vals[int(k)].append(acc / float(k))

    out["ls_n_molecules"] = int(finite.sum())
    for k in ks:
        vals = per_k_vals[int(k)]
        out[f"ls_at_{k}"] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def neighbor_metrics_block_from_preset(preset: Dict[str, Any]) -> Dict[str, Any]:
    return preset.get("embedding_metrics") or {}
