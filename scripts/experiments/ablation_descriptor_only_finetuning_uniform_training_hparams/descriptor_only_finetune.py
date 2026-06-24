"""
Descriptor-only MPP finetuning (numeric RDKit descriptors + MLP, or CLIP text on prior strings + MLP).

Run from repo root (same as ``finetune_moleculenet.py``), e.g.::

    python scripts/experiments/ablation_descriptor_only_finetuning_uniform_training_hparams/descriptor_only_finetune.py \\
      --config ... --config ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root: .../ChemVL-master (this file lives under scripts/experiments/<dir>/)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from dataloader.descriptor_only_dataloader import build_descriptor_only_dataloaders
from dataloader.image_dataloader import load_filenames_and_labels_multitask
from models.descriptor_only import build_descriptor_only_model
from utils.argparser import parse_args
from utils.finetune_utils import (
    get_datafile,
    get_logdir_and_save_config,
    get_metric,
    get_scheduler,
    get_split,
    prefix_add_formatted_time,
)
from utils.descriptor_only_session import (
    build_descriptor_only_optimizer,
    descriptor_training_main_loop,
    run_descriptor_only_session,
)
from utils.public_utils import setup_device
from utils.train_utils import fix_train_random_seed, load_smiles


def main(cfg):
    mode = (cfg["dataset"].get("descriptor_only_mode") or "").strip().lower()
    if mode not in ("feature", "text"):
        raise ValueError("dataset.descriptor_only_mode must be 'feature' or 'text' for this script.")

    task_type = cfg["dataset"]["task_type"]
    image_folder, txt_file = get_datafile(cfg)
    names, labels = load_filenames_and_labels_multitask(image_folder, txt_file, task_type=task_type)
    names, labels = np.array(names), np.array(labels)
    smiles = np.array(load_smiles(txt_file))
    train_idx, val_idx, test_idx = get_split(cfg, names, labels, smiles)

    if mode == "feature":
        run_descriptor_only_session(cfg, smiles, labels, train_idx, val_idx, test_idx)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["basic"]["gpu"]
    device, device_ids = setup_device(cfg["basic"]["ngpu"])
    fix_train_random_seed(cfg["training"]["runseed"])

    cfg["basic"]["prefix"], cfg["basic"]["timestamp"] = prefix_add_formatted_time()
    log_dir = get_logdir_and_save_config(cfg, mkdir=True)

    # --- text mode (not exposed via few-shot utils; kept here) ---
    train_loader, val_loader, test_loader = build_descriptor_only_dataloaders(
        cfg, mode, None, smiles, labels, train_idx, val_idx, test_idx
    )

    eval_metric, valid_select, min_value, criterion, weights = get_metric(cfg, labels[train_idx])

    model = build_descriptor_only_model(cfg)
    model = model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = build_descriptor_only_optimizer(cfg, model.module if isinstance(model, nn.DataParallel) else model)
    scheduler = get_scheduler(cfg, optimizer)

    descriptor_training_main_loop(
        cfg,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        weights,
        device,
        device_ids,
        task_type,
        mode,
        log_dir,
        eval_metric,
        valid_select,
        min_value,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
