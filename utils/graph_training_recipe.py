"""Graph-only training recipe helpers (legacy vs optimized_v1)."""
from __future__ import annotations

from typing import Any, Dict

RECIPE_LEGACY = "legacy"
RECIPE_OPTIMIZED_V1 = "optimized_v1"
VALID_RECIPES = frozenset({RECIPE_LEGACY, RECIPE_OPTIMIZED_V1})


def get_graph_training_recipe(cfg: Dict[str, Any]) -> str:
    """Return graph training recipe; default legacy preserves pre-optimization behavior."""
    model = cfg.get("model") or {}
    recipe = str(model.get("graph_training_recipe", RECIPE_LEGACY)).strip().lower()
    if recipe not in VALID_RECIPES:
        raise ValueError(
            f"Unknown model.graph_training_recipe={recipe!r}; expected one of {sorted(VALID_RECIPES)}"
        )
    return recipe


def is_graph_representation(cfg: Dict[str, Any]) -> bool:
    return (cfg.get("dataset") or {}).get("representation", "image") == "graph"


def graph_add_hs_enabled(cfg: Dict[str, Any]) -> bool:
    """Whether to AddHs when building PyG graphs (graph representation only)."""
    if not is_graph_representation(cfg):
        return False
    dataset = cfg.get("dataset") or {}
    if "graph_add_hs" in dataset:
        return bool(dataset["graph_add_hs"])
    recipe = get_graph_training_recipe(cfg)
    if recipe == RECIPE_OPTIMIZED_V1:
        return True
    return False


def graph_per_target_classification_enabled(cfg: Dict[str, Any]) -> bool:
    """Multi-label per-target independent training (optimized_v1 opt-in)."""
    if not is_graph_representation(cfg):
        return False
    training = cfg.get("training") or {}
    if "graph_per_target_classification" in training:
        return bool(training["graph_per_target_classification"])
    return get_graph_training_recipe(cfg) == RECIPE_OPTIMIZED_V1 and bool(
        training.get("graph_per_target_classification_default", False)
    )
