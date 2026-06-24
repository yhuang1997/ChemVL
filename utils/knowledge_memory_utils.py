"""Helpers for prior knowledge_memory pkl paths and training config sync."""

from __future__ import annotations

import json
import os
from pathlib import Path

from utils.path_utils import get_project_root


def unwrap_model(model):
    """Unwrap nn.DataParallel / similar wrappers."""
    return model.module if hasattr(model, "module") else model


def sync_knowledge_memory_path_to_config(cfg, log_dir, model):
    """
    If knowledge_memory_path was unset, write resolved default path (relative to repo root)
    to log_dir/config.json. Only meaningful for prior_guided_tuning.
    """
    if cfg.get("training", {}).get("finetune_strategy") != "prior_guided_tuning":
        return
    m = cfg.get("model") or {}
    v = m.get("knowledge_memory_path", None)
    if v is not None and isinstance(v, str) and v.strip() != "":
        return
    pf = unwrap_model(model).prior_fusion_block
    abspath = pf.resolved_knowledge_memory_path
    root = get_project_root()
    try:
        rel = str(Path(abspath).relative_to(root))
    except ValueError:
        rel = abspath
    cfg.setdefault("model", {})["knowledge_memory_path"] = rel
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
