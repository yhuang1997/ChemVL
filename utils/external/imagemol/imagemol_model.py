# SPDX-License-Identifier: MIT
"""Official ImageMol ResNet18 + single ``fc(num_tasks)`` head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def build_imagemol_resnet(num_outputs: int) -> nn.Module:
    """Match ``temp_imagemol_finetune.py`` / official ``finetune.py`` (one fc for all tasks)."""
    model = torchvision.models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_outputs)
    return model


class ImageMolFinetuneModel(nn.Module):
    def __init__(self, num_tasks: int, task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type
        self.num_tasks = num_tasks
        out_dim = num_tasks if task_type == "classification" else 1
        self.net = build_imagemol_resnet(out_dim)

    def forward(self, images, smiles=None, **kwargs):
        del smiles, kwargs
        return self.net(images)
