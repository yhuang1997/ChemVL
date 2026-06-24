"""Shared config parsing, data loading, and training helpers for ChemVL fine-tuning entry points."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from dataloader.image_dataloader import ImageDataset, get_datasets
from utils.moleculeace_molmcl import moleculeace_split
from utils.splitter import (
    random_scaffold_split_train_val_test,
    scaffold_split_balanced_train_val_test,
    scaffold_split_train_val_test,
    split_train_val_test_idx,
    split_train_val_test_idx_stratified_v2,
)
from utils.finetune_ablation_attacks import (
    apply_train_misplacement,
    graph_structure_mask_train_args,
    image_mask_dataloader_seed,
    make_dataloader_worker_init_fn,
    maybe_append_image_structure_mask_train,
)
from utils.transform_utils import get_default_transforms, get_transforms as get_augmentation_transforms


# --- Logging and timestamps ---


def prefix_add_formatted_time(prefix=None):
    current_time = time.time()
    current_datetime = datetime.fromtimestamp(current_time)
    formatted_time = current_datetime.strftime("%Y_%m_%d_%H_%M")
    if prefix is not None:
        prefix = prefix + "_" + formatted_time
    return prefix, formatted_time


def prepare_log_dir(cfg, mkdir=True):
    """Create run log directory from ``cfg`` paths; does not write ``config.json``."""
    log_dir = os.path.join(
        cfg["basic"]["log_dir_base"],
        cfg["basic"]["version"],
        cfg["dataset"]["dataset"],
        cfg["basic"]["timestamp"],
    )
    if mkdir:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_run_config(cfg, log_dir):
    """Persist full run ``cfg`` (including ``external_effective_hparams`` when set)."""
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)


def get_logdir_and_save_config(cfg, mkdir=True):
    log_dir = prepare_log_dir(cfg, mkdir=mkdir)
    save_run_config(cfg, log_dir)
    return log_dir


# --- Early stopping (training.use_patience + training.patience) ---


def training_use_patience(cfg: Dict[str, Any]) -> bool:
    return bool((cfg.get("training") or {}).get("use_patience", False))


def training_patience_epochs(cfg: Dict[str, Any]) -> int:
    return int((cfg.get("training") or {}).get("patience", 30))


class ValidEarlyStopping:
    """Stop when valid metric fails to improve for ``patience`` consecutive epochs."""

    def __init__(self, patience: int) -> None:
        self.patience = max(0, int(patience))
        self.stale_epochs = 0

    def step(self, improved: bool) -> bool:
        if improved:
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience


def init_valid_early_stopping(cfg: Dict[str, Any], results: Dict[str, Any]) -> ValidEarlyStopping | None:
    if not training_use_patience(cfg):
        return None
    patience = training_patience_epochs(cfg)
    results["use_patience"] = True
    results["patience"] = patience
    return ValidEarlyStopping(patience)


def check_valid_early_stop(
    early_stopper: ValidEarlyStopping | None,
    *,
    valid_improved: bool,
    epoch: int,
    results: Dict[str, Any],
) -> bool:
    if early_stopper is None:
        return False
    if early_stopper.step(valid_improved):
        results["stopped_early"] = True
        results["epochs_run"] = epoch + 1
        print(
            {
                "early_stop": True,
                "patience": results.get("patience"),
                "epoch": epoch,
                "epochs_run": epoch + 1,
            }
        )
        return True
    return False


# --- Metrics and loss ---


def get_metric(cfg, labels_train=None):
    task_type = cfg["dataset"]["task_type"]

    if task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
        if cfg["training"]["weighted_CE"]:
            labels_train_list = labels_train[labels_train != -1].flatten().tolist()
            count_labels_train = Counter(labels_train_list)
            imbalance_weight = {
                key: 1 - count_labels_train[key] / len(labels_train_list) for key in count_labels_train.keys()
            }
            weights = np.array(sorted(imbalance_weight.items(), key=lambda x: x[0]), dtype="float")[:, 1]
        else:
            weights = None
        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

    elif task_type == "regression":
        if cfg["dataset"]["dataset"] in ["qm7", "qm8", "qm9"]:
            eval_metric = "mae"
        else:
            eval_metric = "rmse"
        valid_select = "min"
        min_value = np.inf
        weights = None
        criterion = nn.MSELoss()
    else:
        raise Exception("{} is not supported".format(task_type))

    return eval_metric, valid_select, min_value, criterion, weights


def get_metric_moleculeace(cfg, labels_train=None):
    """Same as ``get_metric``; regression may use R2 as the primary metric."""
    task_type = cfg["dataset"]["task_type"]
    if task_type == "regression":
        spec = (cfg.get("training") or {}).get("eval_metric") or (cfg.get("training") or {}).get(
            "moleculeace_eval_metric"
        )
        if spec and str(spec).lower() == "r2":
            return "r2", "max", -np.inf, nn.MSELoss(), None
    return get_metric(cfg, labels_train)


# --- Data paths ---


def get_datafile(cfg):
    depiction = cfg["dataset"].get("depiction")
    image_folder, txt_file = get_datasets(
        cfg["dataset"]["dataset"],
        cfg["dataset"]["dataroot"],
        data_type="processed",
        depiction=depiction,
    )
    return image_folder, txt_file


# --- Split indices ---


def get_split(cfg, names, labels, smiles):
    split = cfg["dataset"]["split"]
    seed = cfg["training"]["seed"]
    chirality = cfg["dataset"]["chirality"]

    if split == "random":
        train_idx, val_idx, test_idx = split_train_val_test_idx(
            list(range(0, len(names))),
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=seed,
        )
    elif split == "stratified":
        train_idx, val_idx, test_idx = split_train_val_test_idx_stratified_v2(
            list(range(0, len(names))),
            labels,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=seed,
        )
    elif split == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split_train_val_test(
            list(range(0, len(names))),
            smiles,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            include_chirality=chirality,
        )
    elif split == "random_scaffold":
        train_idx, val_idx, test_idx = random_scaffold_split_train_val_test(
            list(range(0, len(names))),
            smiles,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=seed,
            include_chirality=chirality,
        )
    elif split == "scaffold_balanced":
        train_idx, val_idx, test_idx = scaffold_split_balanced_train_val_test(
            list(range(0, len(names))),
            smiles,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=seed,
            balanced=True,
            include_chirality=chirality,
        )
    else:
        raise Exception("split {} is not supported.".format(split))

    return train_idx, val_idx, test_idx


def get_split_moleculeace(
    cfg: Dict[str, Any],
    names: np.ndarray,
    labels: np.ndarray,
    smiles: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    benchmark = cfg["dataset"].get("benchmark", "moleculeace")
    protocol = (cfg["dataset"].get("protocol") or "MolMCL").strip()
    if benchmark != "moleculeace":
        raise ValueError("get_split_moleculeace expects dataset.benchmark == 'moleculeace'.")

    if protocol == "default":
        raise NotImplementedError(
            'dataset.protocol "default" (MoleculeACE paper / official pipeline) is not '
            'implemented yet. Use protocol "MolMCL".'
        )
    if protocol != "MolMCL":
        raise ValueError(f"Unknown dataset.protocol: {protocol!r}; expected MolMCL or default.")

    split_cfg = cfg["dataset"].get("moleculeace_split") or {}
    in_log10 = cfg["dataset"].get("moleculeace_in_log10", True)

    y = np.asarray(labels, dtype=float).reshape(-1)
    if y.shape[0] != len(smiles):
        raise ValueError("labels and smiles length mismatch for moleculeace split.")

    train_idx, val_idx, test_idx = moleculeace_split(
        list(smiles),
        y.tolist(),
        in_log10=in_log10,
        n_clusters=int(split_cfg.get("n_clusters", 5)),
        val_size=float(split_cfg.get("val_size", 0.1)),
        test_size=float(split_cfg.get("test_size", 0.1)),
        similarity=float(split_cfg.get("similarity", 0.9)),
        potency_fold=int(split_cfg.get("potency_fold", 10)),
        remove_stereo=bool(split_cfg.get("remove_stereo", False)),
    )
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# --- DataLoader / augmentation ---


def get_transforms(cfg):
    if cfg["data_augmentation"]["image_aug"]:
        train_transforms = get_augmentation_transforms(cfg["data_augmentation"])
    else:
        train_transforms = get_default_transforms()
    val_transforms = get_default_transforms()
    test_transforms = get_default_transforms()
    return train_transforms, val_transforms, test_transforms


def get_dataloader(cfg, names, labels, smiles, train_idx, val_idx, test_idx):
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["basic"]["num_workers"]
    finetune_strategy = cfg["training"]["finetune_strategy"]
    representation = cfg["dataset"].get("representation", "image")

    smiles = np.array(smiles)
    if finetune_strategy == "prior_guided_tuning" or representation == "graph":
        train_smiles, val_smiles, test_smiles = smiles[train_idx], smiles[val_idx], smiles[test_idx]
    else:
        train_smiles, val_smiles, test_smiles = None, None, None

    name_train, name_val, name_test = names[train_idx], names[val_idx], names[test_idx]
    labels_train, labels_val, labels_test = labels[train_idx], labels[val_idx], labels[test_idx]

    name_train, train_smiles = apply_train_misplacement(cfg, name_train, train_smiles)

    if representation == "graph":
        from dataloader.graph_dataloader import GraphDataset, build_graph_dataloader
        from utils.graph_training_recipe import graph_add_hs_enabled

        graph_add_hs = graph_add_hs_enabled(cfg)
        g_mask_spec, g_mask_cfg = graph_structure_mask_train_args(cfg)
        train_dataset = GraphDataset(
            name_train,
            labels_train,
            smiles=train_smiles,
            structure_mask_graph_spec=g_mask_spec,
            cfg_for_graph_mask=g_mask_cfg,
            add_hs=graph_add_hs,
        )
        val_dataset = GraphDataset(name_val, labels_val, smiles=val_smiles, add_hs=graph_add_hs)
        test_dataset = GraphDataset(name_test, labels_test, smiles=test_smiles, add_hs=graph_add_hs)

        train_dataloader = build_graph_dataloader(train_dataset, batch_size, True, num_workers)
        val_dataloader = build_graph_dataloader(val_dataset, batch_size, False, num_workers)
        test_dataloader = build_graph_dataloader(test_dataset, batch_size, False, num_workers)
    else:
        train_transforms, val_transforms, test_transforms = get_transforms(cfg)
        train_transforms, use_image_mask_workers = maybe_append_image_structure_mask_train(cfg, train_transforms)
        train_dataset = ImageDataset(
            name_train, labels_train, img_transformer=train_transforms, normalize=None, smiles=train_smiles
        )
        val_dataset = ImageDataset(name_val, labels_val, img_transformer=val_transforms, normalize=None, smiles=val_smiles)
        test_dataset = ImageDataset(
            name_test, labels_test, img_transformer=test_transforms, normalize=None, smiles=test_smiles
        )

        train_kw: Dict[str, Any] = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        if use_image_mask_workers and num_workers > 0:
            train_kw["worker_init_fn"] = make_dataloader_worker_init_fn(image_mask_dataloader_seed(cfg))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kw)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    return train_dataloader, val_dataloader, test_dataloader


# --- Optimizer / scheduler / train step ---


def _split_encoder_head_params(model, finetune_strategy):
    """Split trainable params into encoder vs head for MolCLR-style dual LR."""
    if finetune_strategy not in ("fully_tuning", "from_scratch"):
        return None

    encoder_prefixes = ("feature_extractor.", "image_encoder.", "backbone.")
    head_prefixes = ("pred_bottleneck.", "heads.", "fcs.")

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith("prior_fusion_block"):
            continue
        if any(name.startswith(prefix) for prefix in encoder_prefixes):
            encoder_params.append(param)
        elif any(name.startswith(prefix) for prefix in head_prefixes):
            head_params.append(param)
        else:
            head_params.append(param)

    if encoder_params and head_params:
        return encoder_params, head_params
    return None


def _build_optimizer_param_groups(cfg, params):
    training = cfg["training"]
    optimizer_name = training["optimizer"]
    if optimizer_name == "SGD":
        return [
            {
                "params": params,
                "lr": training["lr"],
                "momentum": training["momentum"],
                "weight_decay": training["weight_decay"],
            }
        ]
    if optimizer_name == "Adam":
        return [
            {
                "params": params,
                "lr": training["lr"],
                "weight_decay": training["weight_decay"],
            }
        ]
    if optimizer_name == "AdamW":
        return [
            {
                "params": params,
                "lr": training["lr"],
                "weight_decay": training["weight_decay"],
            }
        ]
    raise Exception("Optimizer {} is not supported.".format(optimizer_name))


def get_optimizer(cfg, model):
    finetune_strategy = cfg["training"]["finetune_strategy"]
    training = cfg["training"]
    encoder_lr = training.get("encoder_lr")

    split = _split_encoder_head_params(model, finetune_strategy) if encoder_lr is not None else None
    if split is not None:
        encoder_params, head_params = split
        param_groups = [
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": training["weight_decay"],
            },
            {
                "params": head_params,
                "lr": training["lr"],
                "weight_decay": training["weight_decay"],
            },
        ]
        if training["optimizer"] == "SGD":
            for group in param_groups:
                group["momentum"] = training["momentum"]
        elif training["optimizer"] not in ("Adam", "AdamW"):
            raise Exception("Optimizer {} is not supported.".format(training["optimizer"]))
    else:
        bottleneck_params = [
            param
            for name, param in model.named_parameters()
            if not name.startswith("prior_fusion_block") and param.requires_grad
        ]
        param_groups = _build_optimizer_param_groups(cfg, bottleneck_params)

    if training["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(param_groups)
    elif training["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(param_groups)
    elif training["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise Exception("Optimizer {} is not supported.".format(training["optimizer"]))

    if finetune_strategy == "prior_guided_tuning":
        prior_fusion_block_params = [
            param for name, param in model.prior_fusion_block.named_parameters() if not name.startswith("clip_model")
        ]
        optimizer.add_param_group(
            {
                "params": prior_fusion_block_params,
                "lr": cfg["training"]["lr"] * cfg["training"]["prior_lr_factor"],
            }
        )

    return optimizer


def get_scheduler(cfg, optimizer):
    from models.scheduler import LinearWarmupScheduler
    scheduler_name = cfg["training"]["scheduler"]
    num_epochs = cfg["training"]["epochs"]

    if scheduler_name is None:
        return None
    if scheduler_name == "LinearWarmupScheduler":
        return LinearWarmupScheduler(optimizer, warmup_steps=30, total_steps=num_epochs)
    elif scheduler_name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported")


# --- Multitask per-task test metrics (result.json hotfix) ---

_TOX21_TASK_NAMES = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


def multitask_column_names(cfg: Dict[str, Any], num_tasks: int) -> List[str]:
    """MoleculeNet multitask column names for per-task metrics (never uses ``class_names``).

    ``dataset.class_names`` is CoOp binary label text (e.g. class1/class2), not subtask names.
    """
    ds = cfg.get("dataset") or {}
    dataset = str(ds.get("dataset", "")).lower()
    if dataset == "tox21" and num_tasks == 12:
        return list(_TOX21_TASK_NAMES)
    if dataset == "clintox" and num_tasks == 2:
        return ["FDA_APPROVED", "CT_TOX"]
    if dataset in ("lipophilicity", "lipo") and num_tasks == 1:
        return ["experimental"]
    if dataset == "esol" and num_tasks == 1:
        return ["logS"]
    return [f"task_{i}" for i in range(num_tasks)]


def extract_per_task_test_metrics(
    test_results: Dict[str, Any],
    task_names: List[str],
    metric_key: str,
) -> Dict[str, float]:
    """Read per-task test metrics from evaluate_on_multitask multitask output."""
    task_list = test_results.get("result_list_dict_each_task") or []
    out: Dict[str, float] = {}
    key = str(metric_key).upper()
    for i, name in enumerate(task_names):
        if i >= len(task_list) or not task_list[i]:
            continue
        val = task_list[i].get(key)
        if val is None:
            val = task_list[i].get(key.lower())
        if val is not None:
            out[name] = float(val)
    return out


def get_train_fn(finetune_strategy):
    from models.clip_model_utils import train_one_epoch_multitask, train_one_epoch_multitask_separately

    if finetune_strategy in ["text_prompt_tuning", "text_prompt_tuning_prompt_only", "image_adapter_tuning", "prior_guided_tuning"]:
        train_fn = train_one_epoch_multitask_separately
    else:
        train_fn = train_one_epoch_multitask
    return train_fn
