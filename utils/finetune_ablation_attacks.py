"""Finetuning-only ablation hooks (misplacement / structure mask). Kept out of model code paths."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from utils.graph_utils import MASK_ATOM_FEAT_INDEX


def _chemvl_ablation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return (cfg.get("dataset") or {}).get("chemvl_ablation") or {}


def misplacement_enabled(cfg: Dict[str, Any]) -> bool:
    spec = _chemvl_ablation(cfg).get("misplacement") or {}
    return bool(spec.get("enabled"))


def structure_mask_enabled(cfg: Dict[str, Any]) -> bool:
    spec = _chemvl_ablation(cfg).get("structure_mask") or {}
    return bool(spec.get("enabled"))


def resolve_misplacement_k(
    k: Optional[int],
    n: int,
    seed: int,
    runseed: int,
    dataset_name: str,
) -> Optional[int]:
    """Return shift k in 1..n-1, or None if misplacement cannot be applied (n<=1)."""
    if n <= 1:
        return None
    if k is not None:
        kk = int(k) % n
        if kk == 0:
            return resolve_misplacement_k(None, n, seed, runseed, dataset_name)
        return kk
    h = hashlib.sha256(f"{int(seed)}|{int(runseed)}|{dataset_name}".encode()).hexdigest()
    h_int = int(h[:12], 16)
    return 1 + (h_int % (n - 1))


def apply_circular_shift_misplacement(arr: np.ndarray, k: int) -> np.ndarray:
    """struct_i <- struct_{(i+k) mod n}; labels stay aligned with original index i."""
    n = len(arr)
    if n == 0 or k % n == 0:
        return arr
    idx = (np.arange(n, dtype=np.int64) + k) % n
    return arr[idx]


def apply_train_misplacement(
    cfg: Dict[str, Any],
    name_train: np.ndarray,
    train_smiles: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    spec = _chemvl_ablation(cfg).get("misplacement") or {}
    if not spec.get("enabled"):
        return name_train, train_smiles
    split = (spec.get("split") or "train").lower()
    if split != "train":
        return name_train, train_smiles
    n = len(name_train)
    ds_name = str((cfg.get("dataset") or {}).get("dataset", ""))
    seed = int((cfg.get("training") or {}).get("seed", 0))
    runseed = int((cfg.get("training") or {}).get("runseed", 0))
    k = resolve_misplacement_k(spec.get("k"), n, seed, runseed, ds_name)
    if k is None:
        print(f"[chemvl_ablation.misplacement] skip: train size n={n} (need n>=2).")
        return name_train, train_smiles
    mode = (spec.get("mode") or "circular_shift").lower()
    if mode != "circular_shift":
        raise ValueError(f"Unknown chemvl_ablation.misplacement.mode: {mode!r}")
    new_names = apply_circular_shift_misplacement(np.asarray(name_train), k)
    new_smiles = None
    if train_smiles is not None:
        new_smiles = apply_circular_shift_misplacement(np.asarray(train_smiles), k)
    print(f"[chemvl_ablation.misplacement] train circular_shift k={k} (n={n}, dataset={ds_name}).")
    return new_names, new_smiles


def _parse_structure_mask_train(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    spec = _chemvl_ablation(cfg).get("structure_mask") or {}
    if not spec.get("enabled"):
        return None
    if (spec.get("split") or "train").lower() != "train":
        return None
    return spec


def append_image_structure_mask_transform(
    train_compose: Any,
    image_spec: Dict[str, Any],
) -> Any:
    """Append RandomErasing after existing train transforms (expects tensor input)."""
    from torchvision import transforms

    mode = (image_spec.get("mode") or "random_erasing").lower()
    if mode != "random_erasing":
        raise ValueError(f"Unknown structure_mask.image.mode: {mode!r}")
    area = float(image_spec.get("area_ratio", 0.25))
    area = min(max(area, 1e-4), 1.0)
    lo = max(area - 0.02, 0.01)
    hi = min(area + 0.02, 1.0)
    if hi <= lo:
        hi = min(lo + 0.05, 1.0)
    inner = list(train_compose.transforms)
    inner.append(
        transforms.RandomErasing(
            p=1.0,
            scale=(lo, hi),
            ratio=(0.3, 3.3),
            value=0.0,
        )
    )
    return transforms.Compose(inner)


def make_dataloader_worker_init_fn(base_seed: int):
    def _fn(worker_id: int) -> None:
        s = int(base_seed) + int(worker_id) * 9973
        np.random.seed(s)
        torch.manual_seed(s)

    return _fn


def apply_graph_structure_mask(
    data: Any,
    node_mask_fraction: float,
    edge_drop_fraction: float,
    rng: np.random.Generator,
) -> Any:
    """Clone graph; mask random node embeddings (MolCLR mask token); drop random undirected edge pairs."""
    data = data.clone()
    n = int(data.num_nodes)
    if n <= 0:
        return data

    nm = min(max(float(node_mask_fraction), 0.0), 1.0)
    if nm > 0:
        num_mask = int(round(nm * n))
        num_mask = min(max(num_mask, 0), n)
        if num_mask > 0:
            masked_idx = rng.choice(n, size=num_mask, replace=False)
            data.x = data.x.clone()
            data.x[masked_idx, 0] = MASK_ATOM_FEAT_INDEX
            data.x[masked_idx, 1] = 0

    ed = min(max(float(edge_drop_fraction), 0.0), 1.0)
    ei = data.edge_index
    ea = getattr(data, "edge_attr", None)
    if ed > 0 and ei.numel() > 0 and ei.shape[1] % 2 == 0:
        num_pairs = ei.shape[1] // 2
        num_keep = int(round((1.0 - ed) * num_pairs))
        num_keep = min(max(num_keep, 0), num_pairs)
        if num_keep < num_pairs:
            pair_idx = np.arange(num_pairs, dtype=np.int64)
            keep_pairs = rng.choice(pair_idx, size=num_keep, replace=False)
            keep_pairs = np.sort(keep_pairs)
            col_idx = np.concatenate([2 * keep_pairs, 2 * keep_pairs + 1])
            data.edge_index = ei[:, torch.as_tensor(col_idx, dtype=torch.long, device=ei.device)]
            if ea is not None:
                data.edge_attr = ea[torch.as_tensor(col_idx, dtype=torch.long, device=ea.device)]
    return data


def graph_mask_rng_seed(cfg: Dict[str, Any], idx: int) -> int:
    seed = int((cfg.get("training") or {}).get("seed", 0))
    runseed = int((cfg.get("training") or {}).get("runseed", 0))
    ds = str((cfg.get("dataset") or {}).get("dataset", ""))
    h = int(hashlib.sha256(f"graph_mask|{seed}|{runseed}|{ds}|{idx}".encode()).hexdigest()[:12], 16)
    return h % (2**31 - 1)


def maybe_append_image_structure_mask_train(
    cfg: Dict[str, Any], train_transforms: Any,
) -> Tuple[Any, bool]:
    spec = _parse_structure_mask_train(cfg)
    if spec is None:
        return train_transforms, False
    if (cfg.get("dataset") or {}).get("representation", "image") != "image":
        return train_transforms, False
    img = spec.get("image") or {}
    return append_image_structure_mask_transform(train_transforms, img), True


def image_mask_dataloader_seed(cfg: Dict[str, Any]) -> int:
    seed = int((cfg.get("training") or {}).get("seed", 0))
    runseed = int((cfg.get("training") or {}).get("runseed", 0))
    ds = str((cfg.get("dataset") or {}).get("dataset", ""))
    h = int(hashlib.sha256(f"image_mask|{seed}|{runseed}|{ds}".encode()).hexdigest()[:12], 16)
    return h % (2**31 - 1)


def graph_structure_mask_train_args(cfg: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """If graph train structure mask is active, return (graph_subspec, cfg); else (None, None)."""
    spec = _parse_structure_mask_train(cfg)
    if spec is None:
        return None, None
    if (cfg.get("dataset") or {}).get("representation", "image") != "graph":
        return None, None
    g = spec.get("graph") or {}
    nf = float(g.get("node_mask_fraction", 0.0) or 0.0)
    ef = float(g.get("edge_drop_fraction", 0.0) or 0.0)
    if nf <= 0.0 and ef <= 0.0:
        return None, None
    return g, cfg
