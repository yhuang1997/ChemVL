import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import collections
import pickle
from functools import partial
from collections import OrderedDict
import warnings
import sys
import os
import random
import os.path as osp
from .cocoop import CocoopCLIP
from .coop import CoopCLIP
import clip

from models.evaluate import metric as utils_evaluate_metric
from models.evaluate import metric_multitask as utils_evaluate_metric_multitask
from models.evaluate import metric_reg as utils_evaluate_metric_reg
from models.evaluate import metric_reg_multitask as utils_evaluate_metric_reg_multitask
from utils.prior_knowledge import PriorKnowledgeLib
from utils.regression_utils import RegressionDataScheduler
from utils.knowledge_memory_store import load_knowledge_memory_store, save_knowledge_memory_store
from utils.path_utils import get_project_root, get_knowledge_cache_dir, resolve_optional_dir_under_project
from ordinalclip.models.graph_encoders import GRAPH_ENCODERS
from typing import Any, Dict, List, Optional, Sequence

tta_transforms = [
    transforms.Compose([]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)])
]


def _multi_view_train_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    if cfg is None:
        return False
    return bool(cfg.get("data_augmentation", {}).get("multi_view", False))


def _mean_prediction_tta_views(
    model: torch.nn.Module,
    images: torch.Tensor,
    smiles: Optional[torch.Tensor],
    selected_task_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Same 4 deterministic views as eval TTA; average model logits."""
    logits: List[torch.Tensor] = []
    for tta_transform in tta_transforms:
        view = torch.stack([tta_transform(image) for image in images])
        if selected_task_ids is not None:
            logits.append(model(view, smiles=smiles, selected_task_ids=selected_task_ids))
        else:
            logits.append(model(view, smiles=smiles))
    return torch.mean(torch.stack(logits, dim=0), dim=0)


def get_input_batch_size(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.shape[0]
    if hasattr(inputs, "num_graphs"):
        return inputs.num_graphs
    return 0


class GraphCLIPVisual(nn.Module):
    def __init__(self, graph_encoder_cfg, embed_dim):
        super().__init__()
        graph_encoder_cfg = graph_encoder_cfg.copy()
        graph_type = graph_encoder_cfg.pop("type")
        self.graph_encoder = GRAPH_ENCODERS.build(dict(type=graph_type, **graph_encoder_cfg))
        graph_feat_dim = getattr(self.graph_encoder, "feat_dim", embed_dim)
        if graph_feat_dim != embed_dim:
            self.proj = nn.Linear(graph_feat_dim, embed_dim)
        else:
            self.proj = nn.Identity()
        self.input_representation = "graph"

    def forward(self, graph_batch):
        outputs = self.graph_encoder(graph_batch)
        if isinstance(outputs, tuple):
            graph_features = outputs[0]
        else:
            graph_features = outputs
        graph_features = self.proj(graph_features)
        return graph_features


class GraphFeatureExtractor(nn.Module):
    def __init__(self, graph_encoder_cfg):
        super().__init__()
        graph_encoder_cfg = graph_encoder_cfg.copy()
        graph_type = graph_encoder_cfg.pop("type")
        self.graph_encoder = GRAPH_ENCODERS.build(dict(type=graph_type, **graph_encoder_cfg))
        self.feat_dim = self.graph_encoder.feat_dim


    def forward(self, graph_batch):
        outputs = self.graph_encoder(graph_batch)
        if isinstance(outputs, tuple):
            features = outputs[0]
        else:
            features = outputs
        return features


class AdaptedCLIP(nn.Module):
    """
    Wrap the OrdinalCLIP models for fine-tuning regression tasks
    """

    def __init__(self,
                 clip_model,
                 num_classes,
                 finetune_strategy,
                 task_type,
                 dataset,
                 max_grad_norm=None,
                 fixed_params=None,
                 args=None):
        super(AdaptedCLIP, self).__init__()
        assert finetune_strategy in [
            "text_prompt_tuning",
            "text_prompt_tuning_prompt_only",
            "image_adapter_tuning",
            "prior_guided_tuning",
        ]
        # shared components in CoopCLIP and OrdinalCLIP
        self.clip_model = clip_model
        self.image_encoder = clip_model.image_encoder
        self.text_encoder = clip_model.text_encoder
        self.prompt_learners = clip_model.prompt_learners
        self.input_representation = getattr(clip_model, "input_representation", "image")

        self.logit_scale = clip_model.logit_scale
        # self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.embed_dims = clip_model.text_encoder.text_projection.shape[1]

        # specific components in OrdinalCLIP
        if task_type == "classification":
            self.rds = None
            if isinstance(self.clip_model, CoopCLIP):
                self.forward = self.forward_classification_coop
            elif isinstance(self.clip_model, CocoopCLIP):
                self.forward = self.forward_classification_coop
            else:
                raise NotImplementedError(f"clip_model: {type(clip_model)} is not implemented yet.")
        elif task_type == "regression":
            self.rds = clip_model.rds  # regression_data_scheduler
            if isinstance(self.clip_model, CoopCLIP):
                self.forward = self.forward_regression_coop
            else:
                raise NotImplementedError(f"clip_model: {type(clip_model)} is not implemented yet.")
        else:
            raise NotImplementedError(f"task_type: {task_type} is not implemented yet.")

        self.num_classes = num_classes
        self.finetune_strategy = finetune_strategy
        self.task_type = task_type
        self.dataset = dataset
        self.num_tasks = clip_model.num_tasks

        if finetune_strategy == "image_adapter_tuning":
            self.ratio = 0.2
            self.adapter = Adapter(self.embed_dims, reduction=4)
        else:
            self.ratio = 0.0
            self.adapter = nn.Identity()

        # fixed_params
        if fixed_params is not None:
            for param in fixed_params:
                freeze_param(param)

        if self.finetune_strategy == "prior_guided_tuning":
            cache_dir = resolve_optional_dir_under_project(args.get("knowledge_cache_dir"))
            self.prior_fusion_block = PriorFusionBlock(
                self.clip_model,
                embed_dim=1024,
                prior_version=args["prior_version"],
                dataset=dataset,
                mode=args["attention_mode"],
                attention_temperature=args["attention_temperature"],
                dropout=args["dropout"],
                reduction_ratio=args["reduction_ratio"],
                knowledge_memory_path=args.get("knowledge_memory_path"),
                cache_dir=cache_dir,
            )
        if max_grad_norm is not None:
            # Setup hooks and gradient storage
            self.max_grad_norm = max_grad_norm
            self.gradients = collections.defaultdict(list)
            self.hooks = []
            self.register_hooks()
        else:
            self.max_grad_norm = None

        # convert to clip_model dtype
        self.dtype = clip_model.dtype
        self.to(self.dtype)

        # Only for evaluation stage, Text feature memory for prompt learner of specific tasks.
        self.text_feature_memory = {}

    def encode_inputs(self, inputs):
        if self.input_representation == "graph":
            return self.image_encoder(inputs)
        return self.image_encoder(inputs.type(self.dtype))

    def register_hooks(self):
        """
        Register hooks to capture gradients only for modules that require gradients.
        """

        for name, module in self.named_modules():
            if any(param.requires_grad for param in module.parameters()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    self.hooks.append(module.register_full_backward_hook(self.save_gradient(name)))

    def save_gradient(self, module_name):
        """
        Hook to save the gradient.
        """

        def hook(module, grad_input, grad_output):
            grad_norm = grad_output[0].norm().item()
            self.gradients[module_name].append(grad_norm)

        return hook

    def clear_gradients(self):
        """
        Clear the stored gradients.
        """
        self.gradients = collections.defaultdict(list)

    def clip_gradients(self):
        """
        Clip gradients to avoid explosion.
        """
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

    def plot_gradients(self, title=None):
        """
        Plot the gradients for each layer.
        """
        plt.figure(figsize=(10, 6))

        colors = plt.cm.get_cmap('tab20', len(self.gradients))
        for idx, (module_name, grads) in enumerate(self.gradients.items()):
            name_split = module_name.split('.')
            if len(name_split) > 1:
                module_name = "_".join(name_split[1:])
            actual_grads = grads
            clipped_grads = [min(grad, self.max_grad_norm) for grad in grads]

            plt.plot(range(len(actual_grads)), actual_grads, linestyle='dashed', color=colors(idx),
                     label=f'{module_name} (actual)')
            plt.plot(range(len(clipped_grads)), clipped_grads, linestyle='solid', color=colors(idx),
                     label=f'{module_name} (clipped)')

        plt.xlabel('Batch')
        plt.ylabel('Gradient norm')
        plt.yscale('log')  # Use log scale for better visibility
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Position legend in the lower right corner
        plt.legend(loc='lower right', fontsize='small', ncol=1)

        if title is None:
            title = ''
        title = title + f'_max_grad_norm_{self.max_grad_norm}'

        plt.title(title)
        plt.tight_layout()
        plt.show()

    def forward_for_multitask(self, image_features, prior_knowledge_features=None, selected_task_ids=None):
        logit_scale = self.logit_scale.exp()
        logits_for_all_tasks = []

        for task_id in selected_task_ids:
            if self.text_feature_memory.get(task_id) is None or self.training:
                tokenized_prompts = self.prompt_learners[task_id].tokenized_prompts
                if isinstance(self.clip_model, CoopCLIP):
                    prompts = self.prompt_learners[task_id]()
                    text_features = self.text_encoder(prompts, tokenized_prompts)
                    self.text_feature_memory[task_id] = text_features
                else:
                    prompts = self.prompt_learners[task_id](image_features)
                    text_features = []
                    for prompt in prompts:
                        text_feature = self.text_encoder(prompt, tokenized_prompts)
                        text_features.append(text_feature)
                    text_features = torch.stack(text_features)
            else:
                text_features = self.text_feature_memory[task_id]

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if self.finetune_strategy == "prior_guided_tuning":
                logits = self.prior_fusion_block(image_features, text_features, prior_knowledge_features)
            else:
                logits = logit_scale * image_features @ text_features.t()
            logits_for_all_tasks.append(logits)

        logits_for_all_tasks = torch.stack(logits_for_all_tasks, dim=-1)  # (batch_size, num_classes, num_tasks)
        return logits_for_all_tasks

    def forward_classification_coop(self, images, smiles=None, selected_task_ids=None):
        image_features = self.encode_inputs(images)

        image_features = self.adapter(image_features) * self.ratio + image_features * (1 - self.ratio)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.finetune_strategy == "prior_guided_tuning":
            prior_knowledge_features = self.prior_fusion_block.forward_priors(smiles)
        else:
            prior_knowledge_features = None
        if selected_task_ids is None:
            selected_task_ids = range(self.num_tasks)
        return self.forward_for_multitask(image_features, prior_knowledge_features, selected_task_ids)

    def forward_regression_coop(self, images, smiles=None, selected_task_ids=None):
        logit_scale = self.logit_scale.exp()

        image_features = self.encode_inputs(images)

        image_features = self.adapter(image_features) * self.ratio + image_features * (1 - self.ratio)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        regression_values_for_all_tasks = []
        if selected_task_ids is None:
            selected_task_ids = range(self.num_tasks)

        if self.finetune_strategy == "prior_guided_tuning":
            prior_knowledge_features = self.prior_fusion_block.forward_priors(smiles)
        for task_id in selected_task_ids:
            tokenized_prompts = self.prompt_learners[task_id].tokenized_prompts
            prompts = self.prompt_learners[task_id]()
            text_features = self.text_encoder(prompts, tokenized_prompts)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if self.finetune_strategy == "prior_guided_tuning":
                logits = self.prior_fusion_block(image_features, text_features, prior_knowledge_features)
            else:
                logits = logit_scale * image_features @ text_features.t()

            # convert to regression values
            probs = F.softmax(logits, dim=-1)
            device, dtype = logits.device, logits.dtype

            # Generate regression values for each bin
            regression_bins = torch.tensor([self.rds.get_regression_value(task_id, b) for b in range(logits.size(1))],
                                           dtype=dtype, device=device)

            # Calculate the expected regression value using probabilities
            weighted_regression_values = torch.sum(probs * regression_bins, dim=-1)
            regression_values_for_all_tasks.append(weighted_regression_values)

        # (batch_size, num_tasks)
        regression_values_for_all_tasks = torch.stack(regression_values_for_all_tasks, dim=-1)
        return regression_values_for_all_tasks

    def forward_classification_cocoop(self, images, smiles=None, selected_task_ids=None):
        logit_scale = self.logit_scale.exp()

        image_features = self.encode_inputs(images)
        image_features = self.adapter(image_features) * self.ratio + image_features * (1 - self.ratio)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_for_all_tasks = []
        if selected_task_ids is None:
            selected_task_ids = range(self.num_tasks)
        for task_id in selected_task_ids:
            tokenized_prompts = self.prompt_learners[task_id].tokenized_prompts
            prompts = self.prompt_learners[task_id](image_features)
            logits = []
            for pts_i, imf_i in zip(prompts, image_features):
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)
            logits_for_all_tasks.append(logits)

        logits_for_all_tasks = torch.stack(logits_for_all_tasks, dim=-1)  # (batch_size, num_classes, num_tasks)

        return logits_for_all_tasks

    def calculate_text_knowledge_attention(self, smiles=None, selected_task_ids=None):
        if self.prior_fusion_block.mode in ["attention", "static_attention"]:
            calculate_attention_function = self.prior_fusion_block.calculate_attention
        elif self.prior_fusion_block.mode in ["attention_v2", "static_attention_v2"]:
            calculate_attention_function = self.prior_fusion_block.calculate_attention_v2
        else:
            raise NotImplementedError(f"mode: {self.prior_fusion_block.mode} is not implemented yet.")
        assert self.finetune_strategy == "prior_guided_tuning", "Only support prior_guided_tuning strategy."
        attentions = []
        if selected_task_ids is None:
            selected_task_ids = range(self.num_tasks)

        prior_knowledge_features = self.prior_fusion_block.forward_priors(smiles)
        for task_id in selected_task_ids:
            tokenized_prompts = self.prompt_learners[task_id].tokenized_prompts
            prompts = self.prompt_learners[task_id]()
            text_features = self.text_encoder(prompts, tokenized_prompts)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            attention, _ = calculate_attention_function(text_features, prior_knowledge_features)
            attentions.append(attention)

        attentions = torch.stack(attentions, dim=-1)  # (batch_size, num_classes, num_tasks)
        return attentions

    def encode_text_by_clip(self, text):
        x = clip.tokenize(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.text_encoder.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward_text_only(self, task_id=0):
        tokenized_prompts = self.prompt_learners[task_id].tokenized_prompts
        prompts = self.prompt_learners[task_id]()
        text_features = self.text_encoder(prompts, tokenized_prompts)

        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class PriorFusionBlock(nn.Module):
    def __init__(self, clip_model, prior_version, dataset, embed_dim=1024, mode="test",
                 attention_temperature=100.0, reduction_ratio=4,
                 dropout=0.0, cache_dir=None, knowledge_memory_path=None):
        super(PriorFusionBlock, self).__init__()
        # Do not register clip_model as a submodule: self.apply(init_weights) below must not
        # recurse into CoopCLIP (it would re-init text_encoder after load_pretrained_weights).
        object.__setattr__(self, "_clip_model", clip_model)
        self.embed_dim = embed_dim
        self.mode = mode
        self.attention_temperature = None

        self.prior_version = prior_version
        self.dataset = dataset
        self.cache_dir = cache_dir if cache_dir is not None else get_knowledge_cache_dir()

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        kmp = knowledge_memory_path
        self._knowledge_memory_explicit = (
            kmp is not None and isinstance(kmp, str) and kmp.strip() != ""
        )
        cache_dir_abs = os.path.abspath(os.fspath(self.cache_dir))
        if self._knowledge_memory_explicit:
            raw = kmp.strip()
            if os.path.isabs(raw):
                self.resolved_knowledge_memory_path = os.path.abspath(raw)
            else:
                self.resolved_knowledge_memory_path = os.path.abspath(
                    str(get_project_root() / raw)
                )
        else:
            self.resolved_knowledge_memory_path = os.path.join(
                cache_dir_abs,
                "%s_knowledge_memory_%s.pkl" % (dataset, self.prior_version),
            )
        
        if prior_version == "v2_107":
            self.num_prior_knowledge = 107
        elif prior_version == "v2_105":
            self.num_prior_knowledge = 105
        elif prior_version == "v1_3":
            self.num_prior_knowledge = 3
        elif prior_version == "all":
            self.num_prior_knowledge = 208
        else:
            raise NotImplementedError(f"prior_version: {prior_version} is not implemented yet.")

        self.lib = PriorKnowledgeLib(version=prior_version)
        if mode == "linear":
            self.prior_knowledge_linear = nn.Sequential(
                nn.Linear(self.num_prior_knowledge * embed_dim, embed_dim),
                nn.ReLU(),
            )
        elif mode in ["attention", "static_attention"]:
            self.attention_temperature = attention_temperature
            self.reduction_ratio = reduction_ratio

            self.image_features_linear = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // self.reduction_ratio),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )

            self.w_q = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // self.reduction_ratio),
                nn.Dropout(p=dropout)
            )

            self.w_k = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // self.reduction_ratio),
                nn.Dropout(p=dropout)
            )

            self.w_v = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // self.reduction_ratio),
                nn.Dropout(p=dropout)
            )

            self.layer_norm = nn.LayerNorm(embed_dim // self.reduction_ratio)

        elif mode in ["attention_v2", "static_attention_v2"]:
            self.image_alpha = nn.parameter.Parameter(torch.tensor(0.3))
            self.attention_temperature = attention_temperature
            self.reduction_ratio = reduction_ratio

            self.image_features_linear1 = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // self.reduction_ratio),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )

            self.image_features_linear2 = nn.Sequential(
                nn.Linear(embed_dim // self.reduction_ratio, embed_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )

            self.w_q = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // self.reduction_ratio),
                nn.Dropout(p=dropout)
            )

            self.w_k = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // self.reduction_ratio),
                nn.Dropout(p=dropout)
            )

            self.w_v = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // self.reduction_ratio),
                nn.Dropout(p=dropout)
            )

            self.w_o = nn.Sequential(
                nn.Linear(embed_dim // self.reduction_ratio, embed_dim),
                # nn.ReLU(),
                nn.Dropout(p=dropout)
            )

            self.layer_norm = nn.LayerNorm(embed_dim // self.reduction_ratio)
        elif mode == "image_prior_fusion":
            self.w_q = nn.Linear(embed_dim, embed_dim // 4)
            self.w_k = nn.Linear(embed_dim, embed_dim // 4)
            self.w_v = nn.Linear(embed_dim, embed_dim // 4)

            self.image_features_linear = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
            )

        else:
            hidden_dim = 256
            self.image_project = nn.Linear(embed_dim, hidden_dim)
            self.text_project = nn.Linear(embed_dim, hidden_dim * 2)
            self.prior_knowledge_project = nn.Linear(embed_dim, hidden_dim)
            self.fusion_layer = nn.Linear(hidden_dim * self.num_prior_knowledge, hidden_dim)
            self.attention_layer = nn.Linear(hidden_dim, 1)
            self.output_layer = nn.Linear(hidden_dim, 2)

        self.apply(self.init_weights)

        self.knowledge_memory = self.load_knowledge_memory(dataset=dataset)

        self.prior_descriptions = self.lib.load_prior_descriptions()
        self.prior_description_features = None

    @property
    def clip_model(self):
        return self._clip_model

    @property
    def text_encoder(self):
        return self._clip_model.text_encoder

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)

    def forward_priors(self, smiles_list):
        prior_knowledge_embeddings = []
        dev = next(self.text_encoder.parameters()).device
        for smiles in smiles_list:
            if smiles not in self.knowledge_memory.keys():
                pk = self.lib.load_prior_knowledge_features([smiles])[0]
                tpk = clip.tokenize(pk).to(dev)
                text_features = self.text_encoder.encode_text(tpk)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                self.knowledge_memory[smiles] = text_features.cpu()
            else:
                text_features = self.knowledge_memory[smiles].to(dev)
            prior_knowledge_embeddings.append(text_features)
        prior_knowledge_embeddings = torch.stack(prior_knowledge_embeddings)

        return prior_knowledge_embeddings  # batch, num_priors, embed_dim

    def forward_prior_descriptions(self, prior_descriptions):
        description_embeddings = []
        dev = next(self.text_encoder.parameters()).device
        for description in prior_descriptions:
            tpk = clip.tokenize(description).to(dev)
            text_features = self.text_encoder.encode_text(tpk)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            description_embeddings.append(text_features)
        description_embeddings = torch.concatenate(description_embeddings)
        return description_embeddings

    def forward(self, image_features, text_features, prior_knowledge_features):
        # Linear
        if self.mode == "linear":
            prior_knowledge_features = prior_knowledge_features.view(prior_knowledge_features.shape[0], -1)
            prior_knowledge_aggregation = self.prior_knowledge_linear(prior_knowledge_features)
            prior_knowledge_aggregation = prior_knowledge_aggregation / prior_knowledge_aggregation.norm(dim=-1,
                                                                                                         keepdim=True)
            priors_logits = self.clip_model.logit_scale * prior_knowledge_aggregation @ text_features.t()
        # Attention
        elif self.mode in ["attention", "static_attention"]:
            weights, priors_guided_sentences = self.calculate_attention(text_features, prior_knowledge_features)

            image_features = self.image_features_linear(image_features)

            # batch_size, 1, num_classes
            priors_logits = self.clip_model.logit_scale.exp() * image_features.unsqueeze(
                1) @ priors_guided_sentences.transpose(1, 2)
            priors_logits = priors_logits.squeeze(1)

        elif self.mode in ["attention_v2", "static_attention_v2"]:
            weights, priors_guided_sentences = self.calculate_attention_v2(text_features, prior_knowledge_features)

            out = self.image_features_linear1(image_features)
            out = self.image_features_linear2(out)
            image_features = self.image_alpha * out + (1 - self.image_alpha) * image_features

            # batch_size, 1, num_classes
            priors_logits = self.clip_model.logit_scale.exp() * image_features.unsqueeze(
                1) @ priors_guided_sentences.transpose(1, 2)
            priors_logits = priors_logits.squeeze(1)

        elif self.mode == "image_prior_fusion":
            Q = text_features
            K = prior_knowledge_features
            V = prior_knowledge_features
            attention = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
            weights = torch.softmax(attention, dim=-1)
            priors_guided_sentences = torch.matmul(weights, V)
        else:
            image_output = F.relu(self.image_project(image_features))
            text_output = F.relu(self.text_project(text_features))
            prior_knowledge_output = F.relu(self.prior_knowledge_project(prior_knowledge_features))
            attention_weights = F.softmax(self.attention_layer(prior_knowledge_output), dim=1)

            weighted_prior_knowledge_output = torch.mul(prior_knowledge_output, attention_weights)
            weighted_prior_knowledge_output = torch.sum(weighted_prior_knowledge_output, dim=1)

            fused_features = torch.cat((image_output, weighted_prior_knowledge_output), dim=1)

            priors_logits = self.clip_model.logit_scale * fused_features @ text_output.t()
        return priors_logits

    def calculate_attention(self, text_features, prior_knowledge_features):
        if self.mode == "attention":
            Q = self.w_q(text_features)  # n_classes, embeds
            K = self.w_k(prior_knowledge_features)  # batch, n_priors, embeds
            V = self.w_v(prior_knowledge_features)  # batch, n_priors, embeds
            attention = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
            attention = attention * self.attention_temperature
            weights = torch.softmax(attention, dim=-1)
            priors_guided_sentences = torch.matmul(weights, V)
            priors_guided_sentences = self.layer_norm(priors_guided_sentences)
        elif self.mode == "static_attention":
            # num_priors, embed_dim
            self.prior_description_features = self.forward_prior_descriptions(self.prior_descriptions)
            Q = self.w_q(text_features)  # n_classes, embeds
            K = self.w_k(self.prior_description_features)  # n_priors, embeds
            V = self.w_v(prior_knowledge_features)  # batch, n_priors, embeds
            attention = torch.matmul(Q, K.transpose(1, 0)) / torch.sqrt(
                torch.tensor(Q.size(-1), dtype=torch.float32))  # n_classes, n_priors
            # attention = torch.einsum('ij,klj->kil', Q, K) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
            attention = attention * self.attention_temperature
            weights = torch.softmax(attention, dim=-1)
            priors_guided_sentences = torch.matmul(weights, V)
            priors_guided_sentences = self.layer_norm(priors_guided_sentences)
        else:
            raise NotImplementedError(f"mode: {self.mode} is not implemented yet.")
        return weights, priors_guided_sentences

    def calculate_attention_v2(self, text_features, prior_knowledge_features):
        if self.mode == "attention_v2":
            Q = self.w_q(text_features)
            K = self.w_k(prior_knowledge_features)
            V = self.w_v(prior_knowledge_features)
            attention = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
            attention = attention * self.attention_temperature
            weights = torch.softmax(attention, dim=-1)
            priors_guided_sentences = torch.matmul(weights, V)
            priors_guided_sentences = self.layer_norm(priors_guided_sentences)
            priors_guided_sentences = self.w_o(priors_guided_sentences)
        elif self.mode == "static_attention_v2":
            # num_priors, embed_dim
            self.prior_description_features = self.forward_prior_descriptions(self.prior_descriptions)
            Q = self.w_q(text_features)
            K = self.w_k(self.prior_description_features)
            V = self.w_v(prior_knowledge_features)
            attention = torch.matmul(Q, K.transpose(1, 0)) / torch.sqrt(
                torch.tensor(Q.size(-1), dtype=torch.float32))
            attention = attention * self.attention_temperature
            weights = torch.softmax(attention, dim=-1)
            priors_guided_sentences = torch.matmul(weights, V)
            priors_guided_sentences = self.layer_norm(priors_guided_sentences)
            priors_guided_sentences = self.w_o(priors_guided_sentences)
        else:
            raise NotImplementedError(f"mode: {self.mode} is not implemented yet.")
        return weights, priors_guided_sentences

    def save_knowledge_memory(self, dataset):
        save_knowledge_memory_store(self.resolved_knowledge_memory_path, self.knowledge_memory)

    def load_knowledge_memory(self, dataset):
        filename = self.resolved_knowledge_memory_path
        if self._knowledge_memory_explicit and not os.path.exists(filename):
            from utils.knowledge_memory_store import km_shard_dir

            if not os.path.isdir(km_shard_dir(filename)):
                raise FileNotFoundError(
                    "knowledge_memory_path is set but file does not exist: %s" % filename
                )
        knowledge_memory = load_knowledge_memory_store(filename)
        if not knowledge_memory and not os.path.exists(filename):
            from utils.knowledge_memory_store import km_shard_dir

            if not os.path.isdir(km_shard_dir(filename)) and not self._knowledge_memory_explicit:
                warnings.warn(
                    "Knowledge memory cache not found at %s; prior embeddings will be computed "
                    "on the fly and accumulated in RAM until save_knowledge_memory() is called."
                    % filename,
                    stacklevel=2,
                )
        return knowledge_memory


class ExtendedCLIPVisual(nn.Module):
    def __init__(self, image_encoder, num_classes, num_tasks, fix_pretrain_weights=False):
        super(ExtendedCLIPVisual, self).__init__()
        self.image_encoder = image_encoder
        self.num_classes = num_classes
        self.num_tasks = num_tasks

        self.fcs = nn.ModuleList([nn.Linear(image_encoder.output_dim, num_classes) for _ in range(num_tasks)])

        if fix_pretrain_weights:
            freeze_param(self.image_encoder)
        self.to(dtype=torch.float32)

    def forward(self, images, smiles=None):
        x = self.image_encoder(images)
        logits = [fc(x) for fc in self.fcs]
        logits = torch.stack(logits, dim=-1)
        return logits


class ImageMolResNetAdapter(nn.Module):
    """
    Wraps ImageMol-style ResNet for ChemVL multitask loops: accepts ``smiles`` (ignored) and returns
    logits shaped like ``ExtendedCLIPVisual`` — classification ``(B, num_classes, num_tasks)``,
    regression ``(B, 1, num_tasks)`` after the same ``unsqueeze`` convention as ``ExtendedGraphModel``.
    Splits the single pretrained ``fc`` into ``num_tasks`` heads (each initialized from that ``fc``).
    """

    def __init__(self, backbone, num_classes, num_tasks, task_type="classification"):
        super().__init__()
        if not isinstance(backbone.fc, nn.Linear):
            raise TypeError("ImageMolResNetAdapter expects backbone.fc to be nn.Linear.")
        in_features = backbone.fc.in_features
        orig_fc = backbone.fc
        head_out_dim = num_classes if task_type == "classification" else 1
        if orig_fc.out_features != head_out_dim:
            raise ValueError(
                f"backbone.fc.out_features ({orig_fc.out_features}) must match head_out_dim ({head_out_dim})"
            )
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.heads = nn.ModuleList(
            [nn.Linear(in_features, head_out_dim) for _ in range(num_tasks)]
        )
        with torch.no_grad():
            sd = orig_fc.state_dict()
            self.heads[0].load_state_dict(sd)
            for i in range(1, num_tasks):
                self.heads[i].load_state_dict(sd)

    def forward(self, images, smiles=None, **kwargs):
        x = self.backbone(images)
        logits = [head(x) for head in self.heads]
        logits = torch.stack(logits, dim=-1)
        if self.task_type == "regression":
            logits = logits.unsqueeze(1)
        return logits


class ExtendedGraphModel(nn.Module):
    def __init__(self, feature_extractor, num_classes, num_tasks, task_type, fix_backbone=False):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_tasks = num_tasks
        self.task_type = task_type
        head_out_dim = num_classes if task_type == "classification" else 1
        self.pred_bottleneck = nn.Sequential(
            nn.Linear(feature_extractor.feat_dim, feature_extractor.feat_dim // 2),
            nn.Softplus(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(feature_extractor.feat_dim // 2, head_out_dim) for _ in range(num_tasks)]
        )
        if fix_backbone:
            freeze_param(self.feature_extractor)

    def forward(self, graphs, smiles=None):
        features = self.feature_extractor(graphs)
        features = self.pred_bottleneck(features)
        logits = [head(features) for head in self.heads]
        logits = torch.stack(logits, dim=-1)
        if self.task_type == "regression":
            logits = logits.unsqueeze(1)
        return logits


def get_support_model_names():
    return ["RN18*", "RN18", "RN50"]


def load_model(args):
    vision_architecture = args['model']['vision_architecture']
    checkpoint = args['model']['resume']
    prompt_learner = args['model'].get('prompt_learner')
    finetune_strategy = args['training']['finetune_strategy']
    task_type = args['dataset']['task_type']
    num_tasks = args['dataset']['num_tasks']
    representation = args['dataset'].get('representation', 'image')
    if task_type == 'classification':
        class_names = args['dataset']['class_names']
        num_classes = len(class_names)
    else:
        num_classes = 1

    if representation == "image":
        assert vision_architecture in get_support_model_names()
    else:
        assert args['model'].get('graph_encoder') is not None
    assert finetune_strategy in ["from_scratch",
                                 "linear_probing",
                                 "fully_tuning",
                                 "text_prompt_tuning",
                                 "text_prompt_tuning_prompt_only",
                                 "image_adapter_tuning",
                                 "prior_guided_tuning"]

    if checkpoint is not None:
        cp = str(checkpoint).strip()
        checkpoint = cp if osp.isabs(cp) else str(get_project_root() / cp)
    else:
        checkpoint = None
    if finetune_strategy in ["text_prompt_tuning", "text_prompt_tuning_prompt_only", "image_adapter_tuning"]:
        assert prompt_learner in ["coop", "cocoop"]

    # traditional fine-tuning
    if finetune_strategy in ["from_scratch", "linear_probing", "fully_tuning"]:
        if representation == "graph":
            graph_cfg = args['model'].get('graph_encoder')
            feature_extractor = GraphFeatureExtractor(graph_cfg)
            if checkpoint is not None:
                load_pretrained_weights(feature_extractor.graph_encoder, checkpoint)
            visual_model = ExtendedGraphModel(
                feature_extractor,
                num_classes,
                num_tasks,
                task_type,
                fix_backbone=(finetune_strategy == "linear_probing"),
            )
            return visual_model
        # RN18* is the visual_model used in ImageMol, it is a resnet18 built from torchvision
        if vision_architecture == "RN18*":
            visual_model = torchvision.models.resnet18(pretrained=False)
            visual_model.fc = nn.Linear(visual_model.fc.in_features, num_classes)
            if checkpoint is not None:
                # Source code from https://github.com/HongxinXiang/ImageMol/blob/master/finetune.py
                checkpoint = torch.load(checkpoint, map_location="cpu")
                ckp_keys = list(checkpoint["state_dict"])
                cur_keys = list(visual_model.state_dict())
                model_sd = visual_model.state_dict()
                ckp_keys = ckp_keys[:120]
                cur_keys = cur_keys[:120]

                for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                    model_sd[cur_key] = checkpoint["state_dict"][ckp_key]
                visual_model.load_state_dict(model_sd)
            visual_model = ImageMolResNetAdapter(
                visual_model, num_classes=num_classes, num_tasks=num_tasks, task_type=task_type
            )
            if finetune_strategy == "linear_probing":
                freeze_param(visual_model.backbone)
                unfreeze_param(visual_model.heads)
            else:
                unfreeze_param(visual_model)
        # Other backbone is the visual_model used in CLIP, it is a modified ResNet version.
        else:
            if vision_architecture == "RN50":
                clip_model, _ = clip.load(vision_architecture)
                visual_model = clip_model.visual
            else:
                raise NotImplementedError(f"vision_architecture: {vision_architecture} is not implemented yet.")
            # add fc layer and set backbone to be frozen
            if checkpoint is not None:
                load_pretrained_weights(visual_model, checkpoint)
            if finetune_strategy == "linear_probing":
                visual_model = ExtendedCLIPVisual(visual_model, num_classes, num_tasks, fix_pretrain_weights=True)
            else:
                visual_model = ExtendedCLIPVisual(visual_model, num_classes, num_tasks, fix_pretrain_weights=False)
        return visual_model

    # language-guided fine-tuning
    elif finetune_strategy in ["text_prompt_tuning", "text_prompt_tuning_prompt_only", "image_adapter_tuning", "prior_guided_tuning"]:
        if task_type == "regression":
            num_classes = args["regression_scheduler"]["num_classes"]
            bin_mode = args["regression_scheduler"]["bin_mode"]
            try:
                min_max_percentile = args["regression_scheduler"]["min_max_percentile"]
            except KeyError:
                min_max_percentile = None
            labels_csv_path = args["regression_scheduler"].get("labels_csv_path")
            rds = RegressionDataScheduler(task_name=args["dataset"]["dataset"],
                                          num_cls=num_classes,
                                          bin_mode=bin_mode,
                                          min_max_percentile=min_max_percentile,
                                          labels_csv_path=labels_csv_path)
            class_names = rds.generate_class_names()
        else:
            # num_classes -> [num_tasks, num_classes]
            class_names = np.array([class_names] * num_tasks)
            rds = None
        coop_over = args.get("model", {}).get("coop") or {}
        cfg = {
            "task_type": task_type,
            "classnames": class_names,
            "N_CTX": int(coop_over.get("N_CTX", 16)),
            "CSC": False,
            "CTX_INIT": args['dataset']['context_initialization'],
            "PREC": "fp32",
            "CLASS_TOKEN_POSITION": str(coop_over.get("CLASS_TOKEN_POSITION", "end")),
            "N_TASKS": num_tasks,
        }
        if prompt_learner == "coop":
            clip_model = CoopCLIP(text_encoder_name="RN50",
                                  image_encoder_name=vision_architecture,
                                  prompt_learner_cfg=cfg,
                                  regression_data_scheduler=rds)
        else:
            clip_model = CocoopCLIP(text_encoder_name="RN50",
                                    image_encoder_name=vision_architecture,
                                    prompt_learner_cfg=cfg)
        if checkpoint is not None:
            discarded_weights = load_pretrained_weights(clip_model, checkpoint)

        if representation == "graph":
            graph_cfg = args['model'].get('graph_encoder', None)
            if graph_cfg is None:
                raise ValueError("Graph-based fine-tuning requires `model.graph_encoder` configuration.")
            embed_dim = clip_model.text_encoder.text_projection.shape[1]
            clip_model.image_encoder = GraphCLIPVisual(graph_cfg, embed_dim)
            clip_model.input_representation = "graph"

        clip_model.logit_scale = nn.Parameter(torch.tensor(args['model']['logit_scale'])) if args['model'][
                                                                                                 'logit_scale'] is not None else clip_model.logit_scale

        if args["model"].get("freeze_prompt_learner", False):
            for pl in clip_model.prompt_learners:
                freeze_param(pl)

        all_image_encoder_params = list(clip_model.image_encoder.parameters())
        freeze_structure_encoder = (
            args["model"].get("freeze_structure_encoder", False)
            or finetune_strategy == "text_prompt_tuning_prompt_only"
        )

        # For image-based models, only the last layer4 of the structure encoder is finetuned
        # (except text_prompt_tuning_prompt_only / freeze_structure_encoder: freeze full encoder).
        # For graph-based models, all layers of the structure encoder are finetuned
        # (except text_prompt_tuning_prompt_only / freeze_structure_encoder: freeze full encoder).
        if representation == "image":
            if freeze_structure_encoder:
                trainable_image_encoder_params = []
                fixed_image_encoder_params = set(all_image_encoder_params)
            elif hasattr(clip_model.image_encoder, "layer4"):
                trainable_image_encoder_params = list(clip_model.image_encoder.layer4.parameters())
                fixed_image_encoder_params = set(all_image_encoder_params) - set(trainable_image_encoder_params)
            else:
                trainable_image_encoder_params = all_image_encoder_params
                fixed_image_encoder_params = set(all_image_encoder_params) - set(trainable_image_encoder_params)

        elif representation == "graph":
            if freeze_structure_encoder:
                trainable_image_encoder_params = []
                fixed_image_encoder_params = set(all_image_encoder_params)
            else:
                trainable_image_encoder_params = all_image_encoder_params
                fixed_image_encoder_params = set(all_image_encoder_params) - set(trainable_image_encoder_params)

        fixed_params = list(fixed_image_encoder_params) + list(clip_model.text_encoder.parameters())
        if finetune_strategy == "text_prompt_tuning_prompt_only":
            freeze_param(clip_model.logit_scale)

        # wrap the models
        wrapped_model = AdaptedCLIP(clip_model,
                                    num_classes,
                                    finetune_strategy,
                                    task_type,
                                    args['dataset']['dataset'],
                                    max_grad_norm=args['training']['max_grad_norm'],
                                    fixed_params=fixed_params,
                                    args=args['model'])

        return wrapped_model

    else:
        raise NotImplementedError(f"finetune_strategy: {finetune_strategy} is not implemented yet.")


def load_checkpoint(fpath):
    r"""Load checkpoint.
    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.
    Args:
        fpath (str): path to checkpoint.
    Returns:
        dict
    Examples::
        >>> fpath = 'log/my_model/models.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError(f'File is not found at "{fpath}"')

    # Always load tensors to CPU: ckpts may reference cuda:3 (or any id) while this
    # machine has fewer GPUs; model.to(device) runs after weights are applied.
    map_location = torch.device("cpu")

    try:
        checkpoint = torch.load(fpath, map_location=map_location, weights_only=False)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location, weights_only=False
        )

    except Exception:
        print(f'Unable to load checkpoint from "{fpath}"')
        raise

    return checkpoint


def load_pretrained_weights(model: nn.Module, weight_path, prompt_learnrer_idx=None, verbose=False):
    r"""Load pretrianed weights to models.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network models.
        weight_path (str): path to pretrained weights.
    Examples::
        >>> weight_path = 'log/my_model/models-best.pth.tar'
        >>> load_pretrained_weights(models, weight_path)
    """
    if model is None:
        print("models is not instantialized.")
        return

    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if checkpoint.get("epoch") and verbose:
        print(f"load from epoch: {checkpoint.get('epoch')}!")

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        # add support for models built from torchvision but ckpt pretrained from CLIP
        # check image_encoder.xxxx -> xxxx
        elif k[14:] in model_dict and model_dict[k[14:]].size() == v.size():
            new_state_dict[k[14:]] = v
            matched_layers.append(k[14:])
        else:
            if prompt_learnrer_idx is not None:
                if k.startswith("prompt_learners.{}".format(prompt_learnrer_idx)):
                    k = k.replace("prompt_learners.{}.".format(prompt_learnrer_idx), "prompt_learner.")
                    if k in model_dict and model_dict[k].size() == v.size():
                        new_state_dict[k] = v
                        matched_layers.append(k)
                    else:
                        discarded_layers.append(k)
                else:
                    discarded_layers.append(k)
            else:
                discarded_layers.append(k)

    # AdaptedCLIP may register the same text_encoder under both `text_encoder.*` and
    # `clip_model.text_encoder.*`. Pruned checkpoints may only store one prefix; fill aliases.
    for mk in model_dict.keys():
        if mk.startswith("text_encoder.") and mk not in new_state_dict:
            alt = "clip_model." + mk
            if alt in new_state_dict:
                new_state_dict[mk] = new_state_dict[alt]
                matched_layers.append(mk)
        elif mk.startswith("prior_fusion_block.text_encoder.") and mk not in new_state_dict:
            alt = "clip_model.text_encoder." + mk[len("prior_fusion_block.text_encoder.") :]
            if alt in new_state_dict:
                new_state_dict[mk] = new_state_dict[alt]
                matched_layers.append(mk)
        elif mk.startswith("clip_model.text_encoder.") and mk not in new_state_dict:
            alt = mk[len("clip_model.") :]
            if alt in new_state_dict:
                new_state_dict[mk] = new_state_dict[alt]
                matched_layers.append(mk)

    if verbose:
        print(f"len of matched ckpt state_dict: {len(new_state_dict)}")
        print(f"len of models state_dict: {len(model_dict)}")
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)".format(weight_path)
        )
        raise NameError(f"No matched layers for checkpoint: {weight_path}.")
    else:
        if verbose:
            print(f'Successfully loaded pretrained weights from "{weight_path}"')
            if len(discarded_layers) > 0:
                print(
                    "** The following layers are discarded "
                    "due to unmatched keys or layer size: {}".format(discarded_layers)
                )
    return discarded_layers


def set_requires_grad(module, requires_grad):
    if isinstance(module, nn.Module):
        for param in module.parameters():
            param.requires_grad = requires_grad
    elif isinstance(module, nn.parameter.Parameter):
        module.requires_grad = requires_grad
    else:
        raise TypeError(f"The type of the module is wrong: {type(module)}")

    return None


def freeze_param(module):
    if module is None:
        print("models is not instantialized.")
        return
    set_requires_grad(module, False)


def unfreeze_param(module):
    if module is None:
        print("models is not instantialized.")
        return
    set_requires_grad(module, True)


def train_one_epoch_multitask_separately(model, optimizer, data_loader, criterion, weights, device, epoch, task_type,
                                         cfg=None, scheduler=None):
    assert task_type in ["classification", "regression"]

    model.train()
    model.clear_gradients()
    optimizer.zero_grad()
    accu_loss = torch.zeros(1).to(device)
    k = min(5, cfg['dataset']['num_tasks'])

    sample_num = 0
    # data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        if len(data) == 2:
            images, labels = data
            smiles = None
        else:
            images, labels, smiles = data
        selected_task_ids = random.sample(range(labels.shape[1]), k)
        labels = labels[:, selected_task_ids]
        images, labels = images.to(device), labels.to(device)
        batch_size = get_input_batch_size(images)
        sample_num += batch_size if batch_size is not None else 0

        if _multi_view_train_enabled(cfg) and torch.is_tensor(images):
            pred = _mean_prediction_tta_views(
                model, images, smiles, selected_task_ids=selected_task_ids
            )
        else:
            pred = model(images, smiles=smiles,
                         selected_task_ids=selected_task_ids)  # (batch_size, num_classes, selected_num_tasks)
        if task_type == "classification":
            labels = labels.to(torch.int64)  # (batch_size, selected_num_tasks)
            is_valid = labels != -1
            # loss_mat = criterion(pred.double(), labels)
            loss_mat = criterion(pred, labels)
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            if weights is None:
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            else:
                cls_weights = labels.clone()
                cls_weights_mask = []
                for i, weight in enumerate(weights):
                    cls_weights_mask.append(cls_weights == i)
                for i, cls_weight_mask in enumerate(cls_weights_mask):
                    cls_weights[cls_weight_mask] = weights[i]
                loss = torch.sum(loss_mat * cls_weights) / torch.sum(is_valid)
        elif task_type == "regression":
            labels = labels.to(torch.float32)  # (batch_size, selected_num_tasks)
            loss = criterion(pred, labels)
            # loss = criterion(pred.double(), labels)

        print(f"step: {step}, loss: {loss.item()}")
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        model.clip_gradients()
        optimizer.step()
        optimizer.zero_grad()

    if scheduler is not None:
        print(f"lr: {optimizer.param_groups[0]['lr']}")
        scheduler.step()
    # Plot gradients after training
    if cfg['training']['max_grad_norm']:
        model.plot_gradients(title=f"Epoch {epoch}")
    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_on_multitask(model, data_loader, device, task_type="classification",
                          return_data_dict=False, tta=True):
    assert task_type in ["classification", "regression"]

    model.eval()

    y_scores, y_true, y_pred, y_prob = [], [], [], []
    sample_num = 0
    print("Calculating probs...")
    for step, data in enumerate(data_loader):
        if len(data) == 2:
            images, labels = data
            smiles = None
        else:
            images, labels, smiles = data
        images, labels = images.to(device), labels.to(device)
        batch_size = get_input_batch_size(images)
        sample_num += batch_size if batch_size is not None else 0

        with torch.no_grad():
            supports_tta = torch.is_tensor(images)
            if tta and supports_tta:
                tta_logits = []
                for tta_transform in tta_transforms:
                    # Apply TTA transforms to the image
                    tta_images = torch.stack([tta_transform(image) for image in images])
                    tta_pred = model(tta_images, smiles=smiles)

                    tta_logits.append(tta_pred)
                pred = torch.mean(torch.stack(tta_logits), dim=0)
            else:
                pred = model(images, smiles=smiles)
            if task_type == "classification":
                labels = labels.to(torch.int64)  # (batch_size, num_tasks)

        y_true.append(labels)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()  # (N, num_tasks)
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()  # (N, C, num_tasks)

    y_pro = torch.softmax(torch.Tensor(y_scores), dim=1)  # (N, C, num_tasks)
    y_pred = torch.argmax(y_pro, dim=1).numpy()  # (N, C, num_tasks)

    if task_type == "regression":
        if y_scores.shape[1] == 1 and len(y_scores.shape) == 3:
            y_scores = y_scores.squeeze(1)

    print("Calculating metrics...")
    if y_true.shape[1] == 1:
        if task_type == "classification":
            # match sigmoid version to calculate metrics
            y_pro = y_pro[:, 1, :].numpy()  # the probability of positive class
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1), data_dict
            else:
                return utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_scores": y_scores}
                return utils_evaluate_metric_reg(y_true, y_scores), data_dict
            else:
                return utils_evaluate_metric_reg(y_true, y_scores)
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            # match sigmoid version to calculate metrics
            y_pro = y_pro[:, 1, :].numpy()  # the probability of positive class
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1],
                                                       empty=-1), data_dict
            else:
                return utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_scores": y_scores}
                return utils_evaluate_metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1]), data_dict
            else:
                return utils_evaluate_metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1])
    else:
        raise Exception("error in the number of task.")


def train_one_epoch_multitask(model, optimizer, data_loader, criterion, weights, device, epoch, task_type,
                              cfg=None, scheduler=None):
    '''
    :param model:
    :param optimizer:
    :param data_loader:
    :param criterion:
    :param device:
    :param epoch:
    :param criterion_lambda:
    :return:
    '''
    assert task_type in ["classification", "regression"]

    model.train()
    optimizer.zero_grad()
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    # data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        if len(data) == 2:
            images, labels = data
            smiles = None
        else:
            images, labels, smiles = data
        images, labels = images.to(device), labels.to(device)
        batch_size = get_input_batch_size(images)
        sample_num += batch_size if batch_size is not None else 0

        if _multi_view_train_enabled(cfg) and torch.is_tensor(images):
            pred = _mean_prediction_tta_views(model, images, smiles)
        else:
            pred = model(images, smiles=smiles)  # (batch_size, num_classes, num_tasks)
        if task_type == "classification":
            labels = labels.to(torch.int64)  # (batch_size, num_tasks)
            is_valid = labels != -1
            loss_mat = criterion(pred, labels)
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            if weights is None:
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            else:
                cls_weights = labels.clone()
                cls_weights_mask = []
                for i, weight in enumerate(weights):
                    cls_weights_mask.append(cls_weights == i)
                for i, cls_weight_mask in enumerate(cls_weights_mask):
                    cls_weights[cls_weight_mask] = weights[i]
                loss = torch.sum(loss_mat * cls_weights) / torch.sum(is_valid)
        elif task_type == "regression":
            labels = labels.to(torch.float32)
            # reduce labels to (batch_size, num_tasks)
            pred = pred.squeeze(1)
            loss = criterion(pred, labels)

        print(f"step: {step}, loss: {loss.item()}")
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    if scheduler is not None:
        print(f"lr: {optimizer.param_groups[0]['lr']}")
        scheduler.step()
    # Plot gradients after training
    if cfg['training']['max_grad_norm']:
        model.plot_gradients(title=f"Epoch {epoch}")
    return accu_loss.item() / (step + 1)

def save_finetune_ckpt(model, optimizer, epoch, save_path, filename_pre, lr_scheduler=None, result_dict=None,
                       logger=None):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
        'epoch': epoch,
        'model_state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler,
        'result_dict': result_dict
    }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory {} is created.".format(save_path))

    filename = '{}/{}.pth'.format(save_path, filename_pre)
    torch.save(state, filename)
    print('models has been saved as {}'.format(filename))
