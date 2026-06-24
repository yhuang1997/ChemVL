"""PromptLearner init seed resolution (Tier 1 finetune; shared with fs_zs HPO)."""
from __future__ import annotations

import random
from typing import Any, Dict

PL_INIT_FINETUNE_STRATEGIES = frozenset(
    {
        "text_prompt_tuning",
        "text_prompt_tuning_prompt_only",
        "prior_guided_tuning",
    }
)


def resolve_pl_init_seed(training: Dict[str, Any], pl_init_index: int) -> int:
    """Seed for PromptLearner random init; runseed still drives few-shot sampling."""
    explicit = training.get("pl_init_seed")
    runseed = int(training["runseed"])
    if explicit is not None and str(explicit) != "":
        return int(explicit)
    if pl_init_index <= 0:
        return runseed
    stride = int(training.get("pl_init_stride", 1000))
    return runseed + pl_init_index * stride


def ensure_training_seeds(cfg: Dict[str, Any]) -> None:
    """Assign ``runseed`` / ``pl_init_seed`` when missing (public finetune default)."""
    training = cfg.setdefault("training", {})
    assigned: list[str] = []
    if training.get("runseed") is None:
        training["runseed"] = random.randint(0, 100)
        assigned.append(f"runseed={training['runseed']}")
    strategy = training.get("finetune_strategy")
    if strategy in PL_INIT_FINETUNE_STRATEGIES:
        if training.get("pl_init_seed") is None and training.get("pl_init_index") is None:
            training["pl_init_seed"] = random.randint(0, 100)
            assigned.append(f"pl_init={training['pl_init_seed']}")
    if assigned:
        print(f"[seed] {' '.join(assigned)}")
