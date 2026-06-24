import os
import random
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

# Must match the number of deterministic views in ``models.clip_model_utils.tta_transforms``.
_MULTI_VIEW_TRAIN_NUM_VIEWS = 4


def apply_multi_view_train_batch_size_override(cfg: Dict[str, Any]) -> None:
    """
    When ``data_augmentation.multi_view`` is true, set ``training.batch_size`` to
    ``max(1, batch_size // _MULTI_VIEW_TRAIN_NUM_VIEWS)``.

    Configs keep one nominal batch size (e.g. 64 from uniform hparams); only the train
    loader batch shrinks to reduce peak memory across per-view forwards.
    """
    aug = cfg.get("data_augmentation") or {}
    if not aug.get("multi_view"):
        return
    tr = cfg.setdefault("training", {})
    if "batch_size" not in tr:
        return
    orig = int(tr["batch_size"])
    new_bs = max(1, orig // _MULTI_VIEW_TRAIN_NUM_VIEWS)
    if new_bs != orig:
        tr["batch_size"] = new_bs
        print(
            f"multi_view: training.batch_size {orig} -> {new_bs} "
            f"(÷{_MULTI_VIEW_TRAIN_NUM_VIEWS} train-time views)"
        )


def fix_train_random_seed(seed=2021):
    # fix random seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


from utils.pl_init_utils import PL_INIT_FINETUNE_STRATEGIES, resolve_pl_init_seed


def apply_prompt_learner_init_seed(cfg: Dict[str, Any]) -> Optional[int]:
    """
    Set RNG for CoOp PromptLearner init before ``load_model``.

    Only when ``training.pl_init_seed`` or ``training.pl_init_index`` is set and
    strategy is PT/KGPT. Returns the seed used, or None if skipped.

    ``runseed`` should be fixed separately for split / training (see ``extensive_finetune``).
    """
    training = cfg.get("training") or {}
    strategy = training.get("finetune_strategy")
    if strategy not in PL_INIT_FINETUNE_STRATEGIES:
        return None
    if training.get("pl_init_seed") is None and training.get("pl_init_index") is None:
        return None
    pl_init_index = int(training.get("pl_init_index") or 0)
    pl_seed = resolve_pl_init_seed(training, pl_init_index)
    fix_train_random_seed(pl_seed)
    return pl_seed


def load_smiles(txt_file):
    '''
    :param txt_file: should be {dataset}_processed_ac.csv
    :return:
    '''
    df = pd.read_csv(txt_file)
    smiles = df["smiles"].values.flatten().tolist()
    return smiles

