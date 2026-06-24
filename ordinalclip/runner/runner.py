import json
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from ordinalclip.models import MODELS
from ordinalclip.models.ordinalclip import OrdinalCLIP
from ordinalclip.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups
from .utils import freeze_param, load_pretrained_weights

import psutil
import gc

logger = get_logger(__name__)


class Runner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        loss_weights=dict(
            ce_loss=1.0,
            kl_loss=1.0,
        ),
        ckpt_path="",
        **kwargs,
    ) -> None:
        super().__init__()
        self.module = MODELS.build(model_cfg)

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = nn.KLDivLoss(reduction="sum")
        self.loss_weights = loss_weights
        self.num_ranks = self.module.num_ranks
        self.register_buffer("rank_output_value_array", torch.arange(0, self.num_ranks).float(), persistent=False)
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)

        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path

    # Model Forward
    def forward(self, images):
        return self.module(images)

    def forward_text_only(self):
        return self.forward_text_only()

    # Running Steps
    def run_step(self, batch, batch_idx):
        x, y = batch
        logits, *_ = self.module(x)

        losses = self.compute_losses(logits, y)
        loss = sum([weight * losses[k] for k, weight in self.loss_weights.items()])

        metrics_exp = self.compute_per_example_metrics(logits, y, "exp")
        metrics_max = self.compute_per_example_metrics(logits, y, "max")
        return {"loss": loss, **losses, **metrics_exp, **metrics_max}

    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)
        batch_size = self._infer_batch_size(batch)

        self.logging(outputs, "train", on_step=True, on_epoch=True, batch_size=batch_size)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        return outputs

    # Epoch Eval
    def eval_epoch_end(self, outputs, run_type):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        stats = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)
        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

    def validation_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "test")

    def on_train_epoch_start(self) -> None:
        param_group_lrs = {pg["name"]: (pg["lr"], len(list(pg["params"]))) for pg in self.optimizers().param_groups}
        logger.info(f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: {param_group_lrs}")

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    # Logging Utils
    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str):
        for suffix in self.loggings_suffix:
            if key.endswith(suffix):
                return True
        else:
            return False

    def logging(self, outputs: dict, run_type: str, on_step=True, on_epoch=True, batch_size=None):
        for k, v in outputs.items():
            if self._valid_key(k):
                self.log(
                    f"{run_type}_{k}",
                    v.mean(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                )

    # Loss & Metrics
    def compute_losses(self, logits, y):
        losses = {}
        losses["ce_loss"] = self.ce_loss_func(logits, y)
        losses["kl_loss"] = self.compute_kl_loss(logits, y)

        return losses

    def compute_kl_loss(self, logits, y):
        y_t = F.one_hot(y, self.num_ranks).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        y_t_reduction = (y_t * 10.0).softmax(-1)
        y_t_reduction[y_t_row_ind <= 0] = 0

        logits_t = logits.t()
        kl_loss = self.kl_loss_func(F.log_softmax(logits_t, dim=-1), y_t_reduction) / num_slots
        return kl_loss

    def compute_per_example_metrics(self, logits, y, gather_type="exp"):
        dtype = logits.dtype
        probs = F.softmax(logits, -1)

        if gather_type == "exp":
            rank_output_value_array = self.rank_output_value_array.type(dtype)
            predict_y = torch.sum(probs * rank_output_value_array, dim=-1)
        elif gather_type == "max":
            predict_y = torch.argmax(probs, dim=-1).type(dtype)
        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        y = y.type(dtype)
        mae = torch.abs(predict_y - y)
        acc = (torch.round(predict_y) == y).type(logits.dtype)

        return {f"mae_{gather_type}_metric": mae, f"acc_{gather_type}_metric": acc, "predict_y": predict_y}

    # Optimizer & Scheduler
    def configure_optimizers(self):
        return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    def _infer_batch_size(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) == 0:
            return None
        inputs = batch[0]
        if hasattr(inputs, "num_graphs"):
            return getattr(inputs, "num_graphs")
        if isinstance(inputs, torch.Tensor):
            return inputs.shape[0]
        target = batch[1] if len(batch) > 1 else None
        return self._infer_size_from_target(target)

    @staticmethod
    def _infer_size_from_target(target):
        if isinstance(target, torch.Tensor):
            return target.shape[0]
        if isinstance(target, (list, tuple)) and target:
            first = target[0]
            if isinstance(first, torch.Tensor):
                return first.shape[0]
        return None

    def _infer_batch_size(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) == 0:
            return None
        inputs = batch[0]
        if hasattr(inputs, "num_graphs"):
            return getattr(inputs, "num_graphs")
        if isinstance(inputs, torch.Tensor):
            return inputs.shape[0]
        target = batch[1] if len(batch) > 1 else None
        return self._infer_size_from_target(target)

    @staticmethod
    def _infer_size_from_target(target):
        if isinstance(target, torch.Tensor):
            return target.shape[0]
        if isinstance(target, (list, tuple)) and target:
            first = target[0]
            if isinstance(first, torch.Tensor):
                return first.shape[0]
        return None

    def build_optmizer_and_scheduler(
        self,
        param_dict_cfg=None,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ):
        param_dict_ls = self.build_param_dict(**param_dict_cfg)

        optim = build_optimizer(
            model=param_dict_ls,
            **optimizer_cfg,
        )
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    # Model IO
    def load_weights(
        self,
        init_model_weights=None,
        init_prompt_learner_weights=None,
        init_image_encoder_weights=None,
        init_text_encoder_weights=None,
    ):
        if init_model_weights is not None:
            self._custom_logger.info("init_model_weights")
            load_pretrained_weights(self.module, init_model_weights)
            return

        if init_prompt_learner_weights is not None and getattr(self.module, "prompt_learner", None) is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.prompt_learner, init_prompt_learner_weights)
        if init_image_encoder_weights is not None and getattr(self.module, "image_encoder", None) is not None:
            self._custom_logger.info("init_image_encoder_weights")
            load_pretrained_weights(self.module.image_encoder, init_image_encoder_weights)
        if init_text_encoder_weights is not None and getattr(self.module, "text_encoder", None) is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.text_encoder, init_text_encoder_weights)
        return

    def build_param_dict(
        self,
        lr_prompt_learner_context,
        lr_prompt_learner_ranks,
        lr_image_encoder,
        lr_graph_encoder,
        lr_text_encoder,
        lr_logit_scale,
        staged_lr_image_encoder,
    ):
        param_dict_ls = []
        if lr_prompt_learner_context > 0 and self.module.prompt_learner is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.context_embeds,
                    "lr": lr_prompt_learner_context,
                    "init_lr": lr_prompt_learner_context,
                    "name": "lr_prompt_learner_context",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.models.prompt_learner.context_embeds)")
            try:
                freeze_param(self.module.prompt_learner.context_embeds)
            except AttributeError:
                pass

        if lr_prompt_learner_ranks > 0 and self.module.prompt_learner is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.rank_embeds,
                    "lr": lr_prompt_learner_ranks,
                    "init_lr": lr_prompt_learner_ranks,
                    "name": "lr_prompt_learner_ranks",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.models.prompt_learner.rank_embeds)")
            try:
                freeze_param(self.module.prompt_learner.rank_embeds)
            except AttributeError:
                pass

        if getattr(self.module, "image_encoder", None) is not None:
            if lr_image_encoder > 0:
                if staged_lr_image_encoder is not None:
                    self._custom_logger.info("staged_lr_image_encoder activated")
                    image_encoder_param_groups = build_staged_lr_param_groups(
                        model=self.module.image_encoder,
                        lr=lr_image_encoder,
                        **staged_lr_image_encoder,
                    )
                    param_dict_ls.extend(image_encoder_param_groups)
                else:
                    param_dict_ls.append(
                        {
                            "params": self.module.image_encoder.parameters(),
                            "lr": lr_image_encoder,
                            "init_lr": lr_image_encoder,
                            "name": "image_encoder",
                        }
                    )
            else:
                self._custom_logger.info("freeze_param(self.models.image_encoder)")
                freeze_param(self.module.image_encoder)

        if lr_graph_encoder > 0 and getattr(self.module, "graph_encoder", None) is not None:
            param_dict_ls.append(
                {
                    "params": self.module.graph_encoder.parameters(),
                    "lr": lr_graph_encoder,
                    "init_lr": lr_graph_encoder,
                    "name": "graph_encoder",
                }
            )
        elif getattr(self.module, "graph_encoder", None) is not None:
            self._custom_logger.info("freeze_param(self.models.graph_encoder)")
            freeze_param(self.module.graph_encoder)

        if lr_text_encoder > 0 and self.module.text_encoder is not None:
            param_dict_ls.append(
                {
                    "params": self.module.text_encoder.parameters(),
                    "lr": lr_text_encoder,
                    "init_lr": lr_text_encoder,
                    "name": "text_encoder",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.models.text_encoder)")
            freeze_param(self.module.text_encoder)

        if lr_logit_scale > 0 and self.module.logit_scale is not None:
            param_dict_ls.append(
                {
                    "params": self.module.logit_scale,
                    "lr": lr_logit_scale,
                    "init_lr": lr_logit_scale,
                    "name": "logit_scale",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.models.logit_scale)")
            freeze_param(self.module.logit_scale)
        return param_dict_ls

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # Perform inference on a single batch
        x, y = batch
        logits, *_ = self.module(x)
        return x, logits, y


class MultiTaskRunner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        loss_weights=dict(
            ce_loss=1.0,
            kl_loss=1.0,
        ),
        ckpt_path="",
        multi_task_scheme="random",
        save_memory_on_validation: bool = True,
    ) -> None:
        super().__init__()
        self.module = MODELS.build(model_cfg)

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = nn.KLDivLoss(reduction="sum")
        self.loss_weights = loss_weights
        self.num_ranks_list = self.module.num_ranks_list
        for i, num_ranks in enumerate(self.num_ranks_list):
            self.register_buffer(f"rank_output_value_array_{i}", torch.arange(0, num_ranks).float(), persistent=False)
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)

        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.multitask_scheme = multi_task_scheme
        self.save_memory_on_validation = save_memory_on_validation

    @staticmethod
    def _select_y_for_tasks(y, task_ids):
        """Batch labels indexed by task (list of length num_tasks or tensor [B, num_tasks])."""
        if isinstance(y, torch.Tensor):
            if y.dim() == 2:
                return [y[:, i].long() for i in task_ids]
            raise ValueError(f"Unexpected label tensor shape {y.shape} for multitask batch.")
        return [y[i] for i in task_ids]

    # Model Forward
    def forward(self, images):
        return self.module(images)

    def forward_text_only(self):
        return self.forward_text_only()

    # Running Steps
    def run_step(self, batch, batch_idx, mode):
        x, y = batch
        if mode == "train":
            if self.multitask_scheme == "random":
                task_id_for_batch = np.random.randint(0, len(self.num_ranks_list), size=1).tolist()
            elif self.multitask_scheme == "round_robin":
                task_id_for_batch = [batch_idx % len(self.num_ranks_list)]
            elif self.multitask_scheme == "all":
                task_id_for_batch = list(range(len(self.num_ranks_list)))
            else:
                raise ValueError(f"Invalid multitask_scheme: {self.multitask_scheme}")
        else:
            task_id_for_batch = list(range(len(self.num_ranks_list)))
        y_sel = self._select_y_for_tasks(y, task_id_for_batch)
        logits, *_ = self.module(x, task_id_for_batch)
        losses = self.compute_losses(logits, y_sel, task_id_for_batch)
        ce_loss_weight = self.loss_weights["ce_loss"]
        kl_loss_weight = self.loss_weights["kl_loss"]

        loss = 0.0
        for loss_name, loss_value in losses.items():
            if loss_name.startswith("ce_loss"):
                loss += ce_loss_weight * loss_value
            elif loss_name.startswith("kl_loss"):
                loss += kl_loss_weight * loss_value
            else:
                raise ValueError(f"Invalid loss_name: {loss_name}")
        metrics_exp = self.compute_per_example_metrics(logits, y_sel, task_id_for_batch, "exp")
        metrics_max = self.compute_per_example_metrics(logits, y_sel, task_id_for_batch, "max")
        metrics_exp["mae_exp_metric"] = sum(
            metrics_exp[f"mae_exp_metric_{task_id}"] / self.num_ranks_list[task_id] for task_id in task_id_for_batch
        )
        metrics_max["mae_max_metric"] = sum(
            metrics_max[f"mae_max_metric_{task_id}"] / self.num_ranks_list[task_id] for task_id in task_id_for_batch
        )
        return {"loss": loss, **losses, **metrics_exp, **metrics_max}

    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, "train")
        batch_size = self._infer_batch_size(batch)

        self.logging(outputs, "train", on_step=True, on_epoch=True, batch_size=batch_size)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, "val")
        if self.save_memory_on_validation:
            return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in outputs.items()}
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, "test")
        if self.save_memory_on_validation:
            return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in outputs.items()}
        return outputs

    # Epoch Eval
    def eval_epoch_end(self, outputs, run_type):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        stats = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)
        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

    def validation_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "test")

    def on_train_epoch_start(self) -> None:
        param_group_lrs = {pg["name"]: (pg["lr"], len(list(pg["params"]))) for pg in self.optimizers().param_groups}
        logger.info(f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: {param_group_lrs}")

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    # Logging Utils
    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str):
        for suffix in self.loggings_suffix:
            if key.endswith(suffix):
                return True
        else:
            return False

    def logging(self, outputs: dict, run_type: str, on_step=True, on_epoch=True, batch_size=None):
        for k, v in outputs.items():
            if self._valid_key(k):
                self.log(
                    f"{run_type}_{k}",
                    v.mean(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                )

    # Loss & Metrics
    def compute_losses(self, logits, y, task_id):
        losses = {}
        inner_index = 0
        for i, num_ranks in enumerate(self.num_ranks_list):
            if i in task_id:
                losses[f"ce_loss_{i}"] = self.ce_loss_func(logits[inner_index], y[inner_index])
                losses[f"kl_loss_{i}"] = self.compute_kl_loss(logits[inner_index], y[inner_index], num_ranks)
                inner_index += 1
        return losses

    def compute_kl_loss(self, logits, y, num_ranks):
        y_t = F.one_hot(y, num_ranks).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        y_t_reduction = (y_t * 10.0).softmax(-1)
        y_t_reduction[y_t_row_ind <= 0] = 0

        logits_t = logits.t()
        kl_loss = self.kl_loss_func(F.log_softmax(logits_t, dim=-1), y_t_reduction) / num_slots
        return kl_loss

    def compute_per_example_metrics(self, logits_list, y_list, task_id, gather_type="exp"):
        info = {}
        inner_index = 0
        for i in range(len(self.num_ranks_list)):
            if i in task_id:
                logits = logits_list[inner_index]
                y = y_list[inner_index]
                dtype = logits.dtype
                probs = F.softmax(logits, -1)

                if gather_type == "exp":
                    rank_output_value_array = getattr(self, f"rank_output_value_array_{i}").type(dtype)
                    predict_y = torch.sum(probs * rank_output_value_array, dim=-1)
                elif gather_type == "max":
                    predict_y = torch.argmax(probs, dim=-1).type(dtype)
                else:
                    raise ValueError(f"Invalid gather_type: {gather_type}")

                y = y.type(dtype)
                mae = torch.abs(predict_y - y)
                acc = (torch.round(predict_y) == y).type(logits.dtype)

                info[f"mae_{gather_type}_metric_{i}"] = mae
                info[f"acc_{gather_type}_metric_{i}"] = acc
                info[f"predict_y_{gather_type}_{i}"] = predict_y

                inner_index += 1

        return info

    # Optimizer & Scheduler
    def configure_optimizers(self):
        return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    def _infer_batch_size(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) == 0:
            return None
        inputs = batch[0]
        if hasattr(inputs, "num_graphs"):
            return getattr(inputs, "num_graphs")
        if isinstance(inputs, torch.Tensor):
            return inputs.shape[0]
        target = batch[1] if len(batch) > 1 else None
        return self._infer_size_from_target(target)

    @staticmethod
    def _infer_size_from_target(target):
        if isinstance(target, torch.Tensor):
            return target.shape[0]
        if isinstance(target, (list, tuple)) and target:
            first = target[0]
            if isinstance(first, torch.Tensor):
                return first.shape[0]
        return None

    def build_optmizer_and_scheduler(
        self,
        param_dict_cfg=None,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ):
        param_dict_ls = self.build_param_dict(**param_dict_cfg)

        optim = build_optimizer(
            model=param_dict_ls,
            **optimizer_cfg,
        )
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    def build_param_dict(
        self,
        lr_prompt_learner_context,
        lr_prompt_learner_ranks,
        lr_image_encoder,
        lr_graph_encoder,
        lr_text_encoder,
        lr_logit_scale,
        staged_lr_image_encoder,
    ):
        param_dict_ls = []
        if lr_prompt_learner_context > 0 and self.module.prompt_learners is not None:
            for i, prompt_learner in enumerate(self.module.prompt_learners):
                param_dict_ls.append(
                    {
                        "params": prompt_learner.context_embeds,
                        "lr": lr_prompt_learner_context,
                        "init_lr": lr_prompt_learner_context,
                        "name": f"lr_prompt_learner_context_{i}",
                    }
                )
        else:
            self._custom_logger.info("freeze_param(self.models.prompt_learner.context_embeds)")
            try:
                for prompt_learner in self.module.prompt_learners:
                    freeze_param(prompt_learner.context_embeds)
            except AttributeError:
                pass

        if lr_prompt_learner_ranks > 0 and self.module.prompt_learners is not None:
            for i, prompt_learner in enumerate(self.module.prompt_learners):
                param_dict_ls.append(
                    {
                        "params": prompt_learner.rank_embeds,
                        "lr": lr_prompt_learner_ranks,
                        "init_lr": lr_prompt_learner_ranks,
                        "name": f"lr_prompt_learner_ranks_{i}",
                    }
                )
        else:
            self._custom_logger.info("freeze_param(self.models.prompt_learner.rank_embeds)")
            try:
                for prompt_learner in self.module.prompt_learners:
                    freeze_param(prompt_learner.rank_embeds)
            except AttributeError:
                pass

        if getattr(self.module, "image_encoder", None) is not None:
            if lr_image_encoder > 0:
                if staged_lr_image_encoder is not None:
                    self._custom_logger.info("staged_lr_image_encoder activated")
                    image_encoder_param_groups = build_staged_lr_param_groups(
                        model=self.module.image_encoder,
                        lr=lr_image_encoder,
                        **staged_lr_image_encoder,
                    )
                    param_dict_ls.extend(image_encoder_param_groups)
                else:
                    param_dict_ls.append(
                        {
                            "params": self.module.image_encoder.parameters(),
                            "lr": lr_image_encoder,
                            "init_lr": lr_image_encoder,
                            "name": "image_encoder",
                        }
                    )
            else:
                self._custom_logger.info("freeze_param(self.models.image_encoder)")
                freeze_param(self.module.image_encoder)

        if lr_graph_encoder > 0 and getattr(self.module, "graph_encoder", None) is not None:
            param_dict_ls.append(
                {
                    "params": self.module.graph_encoder.parameters(),
                    "lr": lr_graph_encoder,
                    "init_lr": lr_graph_encoder,
                    "name": "graph_encoder",
                }
            )
        elif getattr(self.module, "graph_encoder", None) is not None:
            self._custom_logger.info("freeze_param(self.models.graph_encoder)")
            freeze_param(self.module.graph_encoder)

        if lr_text_encoder > 0 and self.module.text_encoder is not None:
            param_dict_ls.append(
                {
                    "params": self.module.text_encoder.parameters(),
                    "lr": lr_text_encoder,
                    "init_lr": lr_text_encoder,
                    "name": "text_encoder",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.models.text_encoder)")
            freeze_param(self.module.text_encoder)

        if lr_logit_scale > 0 and self.module.logit_scale is not None:
            param_dict_ls.append(
                {
                    "params": self.module.logit_scale,
                    "lr": lr_logit_scale,
                    "init_lr": lr_logit_scale,
                    "name": "logit_scale",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.models.logit_scale)")
            freeze_param(self.module.logit_scale)
        return param_dict_ls

    # Model IO
    def load_weights(
        self,
        init_model_weights=None,
        init_prompt_learner_weights=None,
        init_image_encoder_weights=None,
        init_text_encoder_weights=None,
    ):
        if init_model_weights is not None:
            self._custom_logger.info("init_model_weights")
            load_pretrained_weights(self.module, init_model_weights)
            return

        if init_prompt_learner_weights is not None and getattr(self.module, "prompt_learners", None) is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            for prompt_learner in self.module.prompt_learners:
                load_pretrained_weights(prompt_learner, init_prompt_learner_weights)
        if init_image_encoder_weights is not None and getattr(self.module, "image_encoder", None) is not None:
            self._custom_logger.info("init_image_encoder_weights")
            load_pretrained_weights(self.module.image_encoder, init_image_encoder_weights)
        if init_text_encoder_weights is not None and getattr(self.module, "text_encoder", None) is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.text_encoder, init_text_encoder_weights)
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=None, task_id=None):
        if task_id is None:
            task_id = list(range(len(self.num_ranks_list)))
        # Perform inference on a single batch
        x, y = batch
        logits, *_ = self.module(x, task_id)
        return x, logits, y
