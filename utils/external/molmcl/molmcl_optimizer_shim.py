# SPDX-License-Identifier: MIT
"""Copied from ``external/MolMCL/scripts/finetune.py`` (avoids importing ``prompt_optim`` / ``rogi``)."""

from __future__ import annotations

import torch


def get_optimizer(model: torch.nn.Module, lr_params: dict) -> torch.optim.Optimizer:
    assert isinstance(lr_params, dict)

    pretrain_name, prompt_name, finetune_name = [], [], []
    for name, _param in model.named_parameters():
        if "gnn" in name or "aggr" in name:
            pretrain_name.append(name)
        elif "graph_pred_linear" in name:
            finetune_name.append(name)
        else:
            prompt_name.append(name)

    pretrain_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in pretrain_name, model.named_parameters())))
    )
    finetune_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in finetune_name, model.named_parameters())))
    )
    prompt_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in prompt_name, model.named_parameters())))
    )

    return torch.optim.Adam(
        [
            {"params": finetune_params},
            {"params": pretrain_params, "lr": float(lr_params["pretrain_lr"])},
            {"params": prompt_params, "lr": float(lr_params["prompt_lr"])},
        ],
        lr=float(lr_params["finetune_lr"]),
        weight_decay=float(lr_params["decay"]),
    )
