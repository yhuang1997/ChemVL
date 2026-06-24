# SPDX-License-Identifier: MIT
"""Train-set label normalizer (from ``external/MolCLR/finetune.py``)."""

from __future__ import annotations

import torch


class MolCLRLabelNormalizer:
    def __init__(self, tensor: torch.Tensor):
        self.mean = torch.mean(tensor.float())
        self.std = torch.std(tensor.float())
        if self.std == 0:
            self.std = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)

    def denorm(self, normed: torch.Tensor) -> torch.Tensor:
        return normed * self.std.to(normed.device) + self.mean.to(normed.device)
