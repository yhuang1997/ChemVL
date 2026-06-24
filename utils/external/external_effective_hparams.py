# SPDX-License-Identifier: MIT
"""Snapshot merged external-backend hyperparameters for run config audit."""

from __future__ import annotations

from typing import Any, Dict


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def snapshot_external_effective_hparams(cfg: Dict[str, Any], backend_name: str) -> Dict[str, Any]:
    """
    Return audit-friendly effective hparams for ``finetune_external`` backends.

    Written to ``cfg['external_effective_hparams']`` before ``config.json`` is saved.
    """
    name = str(backend_name or "").lower()
    out: Dict[str, Any] = {"backend": name}

    if name == "imagemol_moleculenet":
        from utils.external.imagemol.imagemol_external_config import load_merged_imagemol_config

        merged = load_merged_imagemol_config(cfg)
        im = (cfg.get("model") or {}).get("imagemol") or {}
        out["source_yaml"] = im.get("external_hparams_yaml") or "configs/external/imagemol/params_imagemol.yaml"
        for key in ("lr", "batch_size", "epochs", "momentum", "weight_decay", "optimizer"):
            if key in merged:
                out[key] = _json_safe(merged[key])
        return out

    if name == "molclr_moleculenet":
        from utils.external.molclr.molclr_external_config import load_merged_molclr_config

        merged = load_merged_molclr_config(cfg)
        mc = (cfg.get("model") or {}).get("molclr") or {}
        out["source_yaml"] = mc.get("external_hparams_yaml") or "scripts/external/molclr_under_chemvl/params_molclr.yaml"
        out["external_config"] = mc.get("external_config") or "external/MolCLR/config_finetune.yaml"
        for key in ("init_lr", "init_base_lr", "batch_size", "epochs", "model_type"):
            if key in merged:
                out[key] = _json_safe(merged[key])
        return out

    if name in ("molmcl_moleculenet", "molmcl_moleculeace"):
        from utils.external.molmcl.molmcl_external_config import load_merged_molmcl_config

        merged = load_merged_molmcl_config(cfg)
        mc = (cfg.get("model") or {}).get("molmcl") or {}
        out["external_config"] = mc.get("external_config")
        out["external_config_extra"] = mc.get("external_config_extra")
        out["yaml_config"] = mc.get("yaml_config")
        for key in ("epochs", "batch_size"):
            if key in merged:
                out[key] = _json_safe(merged[key])
        optim = merged.get("optim")
        if isinstance(optim, dict):
            out["optim"] = _json_safe(
                {k: optim[k] for k in ("finetune_lr", "prompt_lr", "pretrain_lr", "weight_decay") if k in optim}
            )
        return out

    return out
