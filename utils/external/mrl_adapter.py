# SPDX-License-Identifier: MIT
"""Shared MRL-adapter infrastructure (Task2): registry hooks, protocol shims, etc.

Keep external-project-specific logic in subpackages (e.g. ``utils.external.molmcl``).
"""

from utils.external.chemvl_external_backend import (
    FinetuneBackend,
    build_finetune_backend,
)

__all__ = ["FinetuneBackend", "build_finetune_backend"]
