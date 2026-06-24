# SPDX-License-Identifier: MIT
"""Load ImageMol pretrained backbone (first 120 state_dict keys)."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from utils.external.molmcl.molmcl_external_config import chemvl_repo_root


def resolve_imagemol_checkpoint(path: str | None) -> str | None:
    if not path:
        return None
    p = os.path.expanduser(str(path))
    if os.path.isfile(p):
        return os.path.abspath(p)
    repo = chemvl_repo_root()
    cand = (repo / p).resolve()
    if cand.is_file():
        return str(cand)
    env_root = os.environ.get("CHEMVL_DATA_ROOT", "")
    if env_root:
        cand2 = os.path.join(env_root, p.lstrip("/"))
        if os.path.isfile(cand2):
            return os.path.abspath(cand2)
    return p


def load_imagemol_pretrained_backbone(
    model: nn.Module, checkpoint_path: str, *, resume_key: str = "state_dict"
) -> None:
    """Copy first 120 keys from ImageMol checkpoint into ``model.net`` (ResNet18)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt[resume_key] if isinstance(ckpt, dict) and resume_key in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected ImageMol checkpoint format: {type(state)}")

    model_sd = model.net.state_dict()
    ckp_keys = list(state.keys())[:120]
    cur_keys = list(model_sd.keys())[:120]
    for ckp_key, cur_key in zip(ckp_keys, cur_keys):
        model_sd[cur_key] = state[ckp_key]
    model.net.load_state_dict(model_sd)
    print(f"Loaded ImageMol backbone (120 keys) from {checkpoint_path}")
