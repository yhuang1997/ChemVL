# SPDX-License-Identifier: MIT
"""Load ``external/MolCLR/models/*`` without shadowing ChemVL ``models`` package."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from typing import Any, Type


def _load_module_from_path(module_name: str, file_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_molclr_ginet_class(molclr_root: str) -> Type[Any]:
    path = os.path.join(molclr_root, "models", "ginet_finetune.py")
    mod = _load_module_from_path("molclr_ginet_finetune", path)
    return mod.GINet


def get_molclr_gcn_class(molclr_root: str) -> Type[Any]:
    path = os.path.join(molclr_root, "models", "gcn_finetune.py")
    mod = _load_module_from_path("molclr_gcn_finetune", path)
    return mod.GCN
