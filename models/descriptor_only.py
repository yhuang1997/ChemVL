"""Descriptor-only baselines: numeric vector + MLP, or CLIP text on prior sentences + pool + MLP."""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.clip_model_utils import load_pretrained_weights
from utils.path_utils import get_descriptor_only_text_cache_dir
from utils.prior_knowledge import PriorKnowledgeLib


def _mlp_tower(in_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(p=dropout)])
        d = h
    return nn.Sequential(*layers), d


class DescriptorOnlyFeatureModel(nn.Module):
    """Standardized RDKit descriptor vector -> shared MLP -> per-task heads (ChemVL logit shape)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        num_classes: int,
        num_tasks: int,
        task_type: str,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.backbone, feat_dim = _mlp_tower(in_dim, hidden_dims, dropout)
        head_out = num_classes if task_type == "classification" else 1
        self.heads = nn.ModuleList([nn.Linear(feat_dim, head_out) for _ in range(num_tasks)])

    def forward(self, x: Optional[torch.Tensor] = None, smiles: Optional[Any] = None) -> torch.Tensor:
        del smiles
        if x is None:
            raise ValueError("DescriptorOnlyFeatureModel requires input tensor x.")
        h = self.backbone(x)
        logits = [head(h) for head in self.heads]
        return torch.stack(logits, dim=-1)


class DescriptorOnlyTextModel(nn.Module):
    """
    Per-molecule: same string list as ``PriorKnowledgeLib.load_prior_knowledge_features``,
    CLIP ``encode_text`` + L2 normalize + mean pool -> MLP heads.
    Pooled vectors optionally cached under ``get_descriptor_only_text_cache_dir()``.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        prior_version: str,
        hidden_dims: Sequence[int],
        dropout: float,
        num_classes: int,
        num_tasks: int,
        task_type: str,
        cache_enabled: bool,
        ckpt_fingerprint: str,
        freeze_clip_text: bool = True,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.lib = PriorKnowledgeLib(version=prior_version)
        self.prior_version = prior_version
        self.cache_enabled = cache_enabled
        self.ckpt_fingerprint = ckpt_fingerprint
        self._text_dim = int(clip_model.text_projection.shape[1])
        if freeze_clip_text:
            for p in self.clip_model.parameters():
                p.requires_grad = False
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.backbone, feat_dim = _mlp_tower(self._text_dim, hidden_dims, dropout)
        head_out = num_classes if task_type == "classification" else 1
        self.heads = nn.ModuleList([nn.Linear(feat_dim, head_out) for _ in range(num_tasks)])

    def _cache_dir(self) -> str:
        base = get_descriptor_only_text_cache_dir()
        sub = os.path.join(str(base), f"pooled_{self.prior_version}_{self.ckpt_fingerprint}")
        os.makedirs(sub, exist_ok=True)
        return sub

    def _cache_path(self, smiles: str) -> str:
        h = hashlib.sha256(f"{smiles}|{self.prior_version}|{self.ckpt_fingerprint}".encode("utf-8")).hexdigest()
        return os.path.join(self._cache_dir(), f"{h}.pt")

    def _encode_one_smiles(self, smiles: str, device: torch.device) -> torch.Tensor:
        if self.cache_enabled:
            path = self._cache_path(smiles)
            if os.path.isfile(path):
                try:
                    t = torch.load(path, map_location=device, weights_only=True)
                except TypeError:
                    t = torch.load(path, map_location=device)
                if isinstance(t, dict) and "pooled" in t:
                    return t["pooled"].to(device=device, dtype=torch.float32)
        pk = self.lib.load_prior_knowledge_features([smiles])[0]
        import clip

        tpk = clip.tokenize(pk).to(device)
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.clip_model.parameters())):
            text_features = self.clip_model.encode_text(tpk)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        pooled = text_features.mean(dim=0)
        if self.cache_enabled:
            path = self._cache_path(smiles)
            torch.save({"pooled": pooled.detach().cpu()}, path)
        return pooled.to(dtype=torch.float32)

    def forward(self, x: Optional[torch.Tensor] = None, smiles: Optional[Sequence[str]] = None) -> torch.Tensor:
        if smiles is None:
            raise ValueError("DescriptorOnlyTextModel requires smiles= list of SMILES per batch.")
        device = next(self.clip_model.parameters()).device
        feats = []
        for s in smiles:
            feats.append(self._encode_one_smiles(str(s), device))
        h_in = torch.stack(feats, dim=0)
        h = self.backbone(h_in)
        logits = [head(h) for head in self.heads]
        return torch.stack(logits, dim=-1)


def build_descriptor_only_model(cfg: Dict[str, Any]) -> nn.Module:
    ds = cfg.get("dataset") or {}
    mode = (ds.get("descriptor_only_mode") or "").strip().lower()
    spec = (cfg.get("model") or {}).get("descriptor_only") or {}
    task_type = ds["task_type"]
    num_tasks = int(ds["num_tasks"])
    if task_type == "classification":
        num_classes = len(ds["class_names"])
    else:
        num_classes = 1
    hidden_dims = tuple(spec.get("hidden_dims") or (512, 256))
    dropout = float(spec.get("dropout", 0.1))
    prior_ver = (ds.get("prior_descriptor_version") or "all").strip()

    if mode == "feature":
        in_dim = len(PriorKnowledgeLib(version=prior_ver).prior_keys)
        return DescriptorOnlyFeatureModel(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=num_classes,
            num_tasks=num_tasks,
            task_type=task_type,
        )

    if mode == "text":
        import clip

        arch = (cfg.get("model") or {}).get("vision_architecture", "RN50")
        clip_model, _ = clip.load(arch, device="cpu", jit=False)
        ckpt = (cfg.get("model") or {}).get("resume")
        fp = "no_ckpt"
        if ckpt:
            ckpt = str(ckpt)
            fp = hashlib.sha256(os.path.abspath(ckpt).encode()).hexdigest()[:16]
            try:
                load_pretrained_weights(clip_model, ckpt, verbose=False)
            except Exception as e:
                warnings.warn(f"Descriptor-only text: partial ChemVL load failed ({e}); keeping OpenAI CLIP weights.")
        freeze = bool(spec.get("freeze_clip_text", True))
        cache = bool(spec.get("cache_text_embeddings", True))
        return DescriptorOnlyTextModel(
            clip_model=clip_model,
            prior_version=prior_ver,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=num_classes,
            num_tasks=num_tasks,
            task_type=task_type,
            cache_enabled=cache,
            ckpt_fingerprint=fp,
            freeze_clip_text=freeze,
        )

    raise ValueError(f"Unknown dataset.descriptor_only_mode: {mode!r}; expected 'feature' or 'text'.")
