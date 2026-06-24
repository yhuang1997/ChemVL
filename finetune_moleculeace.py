"""
MoleculeACE fine-tuning — **public release** entry (MolMCL split + ChemVL training).

Split: ``dataset.benchmark == "moleculeace"`` + ``dataset.protocol == "MolMCL"`` →
``moleculeace_split`` (same partition as MolMCL finetune).

Training semantics (differs from internal ``moleculeace_finetune.py``):

- Each epoch: train + validation only. **No test split inside the training loop.**
- ``train_best.pth`` / ``valid_best.pth`` are always saved (ignores ``save_finetune_ckpt``).
- No ``train_val_test_history.csv`` or ``*_on_test`` fields during training.
- After training, both checkpoints are loaded and evaluated on test once; see
  ``final_test_train_best`` / ``final_test_valid_best`` in ``result.json``.
- **Seeds**: if ``training.runseed`` / ``training.pl_init_seed`` are omitted from config,
  each run draws ``random.randint(0, 100)`` before training (PT/KGPT only for ``pl_init_seed``).
  Formal ``configs/moleculeace/`` configs may set them explicitly.

Example::

    python finetune_moleculeace.py --config configs/moleculeace/molmcl_chembl_template.json
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.parallel

from dataloader.image_dataloader import load_filenames_and_labels_multitask
from models.clip_model_utils import evaluate_on_multitask, load_model, save_finetune_ckpt
from utils.argparser import parse_args
from utils.finetune_public_utils import final_eval_saved_ckpts_on_test
from utils.finetune_utils import (
    check_valid_early_stop,
    get_dataloader,
    get_datafile,
    get_logdir_and_save_config,
    get_metric_moleculeace as get_metric,
    get_optimizer,
    get_scheduler,
    get_split_moleculeace,
    get_train_fn,
    init_valid_early_stopping,
    prefix_add_formatted_time,
)
from utils.plot_utils import plot_loss_rocauc
from utils.public_utils import is_left_better_right, setup_device
from utils.pl_init_utils import ensure_training_seeds
from utils.train_utils import (
    apply_multi_view_train_batch_size_override,
    apply_prompt_learner_init_seed,
    fix_train_random_seed,
    load_smiles,
)


def main(cfg: Dict[str, Any]):
    ensure_training_seeds(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["basic"]["gpu"]
    device, device_ids = setup_device(cfg["basic"]["ngpu"])
    fix_train_random_seed(cfg["training"]["runseed"])
    apply_multi_view_train_batch_size_override(cfg)

    cfg["basic"]["prefix"], cfg["basic"]["timestamp"] = prefix_add_formatted_time()
    log_dir = get_logdir_and_save_config(cfg, mkdir=True)

    protocol = (cfg["dataset"].get("protocol") or "MolMCL").strip()
    if protocol != "MolMCL":
        raise NotImplementedError(f"protocol={protocol!r}: use protocol MolMCL for finetune_moleculeace.")

    task_type = cfg["dataset"]["task_type"]
    finetune_strategy = cfg["training"]["finetune_strategy"]
    ckpt_path = cfg["model"]["resume"]

    image_folder, txt_file = get_datafile(cfg)

    cfg.setdefault("regression_scheduler", {})["labels_csv_path"] = txt_file

    names, labels = load_filenames_and_labels_multitask(image_folder, txt_file, task_type=task_type)
    names, labels = np.array(names), np.array(labels)
    smiles = load_smiles(txt_file)

    train_idx, val_idx, test_idx = get_split_moleculeace(cfg, names, labels, smiles)
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        cfg, names, labels, smiles, train_idx, val_idx, test_idx
    )

    eval_metric, valid_select, min_value, criterion, weights = get_metric(cfg, labels[train_idx])
    representation = cfg["dataset"].get("representation", "image")
    eval_with_tta = representation == "image"

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("No checkpoint found at '{}'".format(ckpt_path))
        print("Using vanilla CLIP model checkpoints.")
    else:
        print("Loading pre-trained checkpoint '{}'".format(ckpt_path))

    pl_init_seed = apply_prompt_learner_init_seed(cfg)
    model = load_model(cfg)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    train_fn = get_train_fn(finetune_strategy)

    loss_list = []
    valid_metric_list = []
    metric_key = eval_metric.upper()

    results = {
        "best_valid": min_value,
        "best_valid_epoch": 0,
        "best_train_loss": np.inf,
        "best_train_epoch": 0,
        "runseed": cfg["training"]["runseed"],
    }
    if pl_init_seed is not None:
        results["pl_init_seed"] = pl_init_seed

    early_stopper = init_valid_early_stopping(cfg, results)

    for epoch in range(cfg["training"]["start_epoch"], cfg["training"]["epochs"]):
        train_step_loss = train_fn(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_dataloader,
            criterion=criterion,
            weights=weights,
            device=device,
            epoch=epoch,
            task_type=task_type,
            cfg=cfg,
        )

        val_results, _ = evaluate_on_multitask(
            model=model,
            data_loader=val_dataloader,
            device=device,
            task_type=task_type,
            return_data_dict=True,
            tta=eval_with_tta,
        )

        loss_list.append(train_step_loss)
        valid_result = float(val_results[metric_key])
        valid_metric_list.append(valid_result)

        print(
            {
                "Epoch": epoch,
                "train_step_loss": train_step_loss,
                f"valid_{eval_metric}": valid_result,
            }
        )

        if is_left_better_right(train_step_loss, results["best_train_loss"], standard="min"):
            results["best_train_loss"] = train_step_loss
            results["best_train_epoch"] = epoch
            save_finetune_ckpt(
                model,
                optimizer,
                epoch,
                log_dir,
                "train_best",
                lr_scheduler=scheduler,
                result_dict=results,
            )

        valid_improved = is_left_better_right(valid_result, results["best_valid"], standard=valid_select)
        if valid_improved:
            results["best_valid"] = valid_result
            results["best_valid_epoch"] = epoch
            save_finetune_ckpt(
                model,
                optimizer,
                epoch,
                log_dir,
                "valid_best",
                lr_scheduler=scheduler,
                result_dict=results,
            )

        if check_valid_early_stop(
            early_stopper, valid_improved=valid_improved, epoch=epoch, results=results
        ):
            break

        if epoch == 0 and finetune_strategy == "prior_guided_tuning":
            model.prior_fusion_block.save_knowledge_memory(dataset=cfg["dataset"]["dataset"])

        reg_plot = eval_metric.lower() if task_type == "regression" else "rmse"
        plot_loss_rocauc(loss_list, valid_metric_list, task_type, log_dir=log_dir, regression_metric=reg_plot)

    final_eval_saved_ckpts_on_test(
        log_dir=log_dir,
        test_dataloader=test_dataloader,
        device=device,
        task_type=task_type,
        eval_with_tta=eval_with_tta,
        eval_metric=eval_metric,
        results=results,
    )

    with open(os.path.join(log_dir, "result.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
