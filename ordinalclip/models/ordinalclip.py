import os.path as osp

import torch
import torch.nn as nn
import torchvision.models as models
from clip import clip

from ordinalclip.utils import get_logger

from . import image_encoders
from .builder import MODELS
from .graph_encoders import GRAPH_ENCODERS
from .prompt_leaners import PROMPT_LEARNERS
from .prompt_leaners.plain_prompt_learner import PlainPromptLearner

logger = get_logger(__name__)


@MODELS.register_module()
class OrdinalCLIP(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        # convert to float32
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 models, call `clip.models.convert_weights`")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learner: PlainPromptLearner = PROMPT_LEARNERS.build(prompt_learner_cfg)
        self.psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens
        self.logit_scale = clip_model.logit_scale

        self.embed_dims = clip_model.text_projection.shape[1]
        self.num_ranks = self.prompt_learner.num_ranks

    def forward(self, images):
        sentence_embeds = self.prompt_learner()
        numpy_sentence_embeds = sentence_embeds.detach().cpu().numpy()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features

    def forward_text_only(self):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)

        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)


@MODELS.register_module()
class OrdinalCLIPForMultiTask(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        # convert to float32
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 models, call `clip.models.convert_weights`")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learners = self._build_prompt_learners(prompt_learner_cfg)
        self.psudo_sentence_tokens_list = [pl.psudo_sentence_tokens for pl in self.prompt_learners]
        self.logit_scale = clip_model.logit_scale

        self.embed_dims = clip_model.text_projection.shape[1]
        self.num_ranks_list = [pl.num_ranks for pl in self.prompt_learners]

    def _build_prompt_learners(self, prompt_learner_cfg):
        assert prompt_learner_cfg["num_tasks"] == len(prompt_learner_cfg["num_ranks"])
        prompt_learners = nn.ModuleList()
        for i in range(prompt_learner_cfg["num_tasks"]):
            prompt_learner_cfg_per_rank = prompt_learner_cfg.copy()
            prompt_learner_cfg_per_rank["num_ranks"] = prompt_learner_cfg["num_ranks"][i]
            prompt_learners.append(PROMPT_LEARNERS.build(prompt_learner_cfg_per_rank))
        return prompt_learners

    def forward(self, images, task_id=None):
        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if task_id is None:
            task_id = list(range(len(self.prompt_learners)))
        sentence_embeds_by_task = {tid: self.prompt_learners[tid]() for tid in task_id}
        psudo_sentence_tokens_list = self.psudo_sentence_tokens_list
        logit_scale = self.logit_scale.exp()  # shared logit scale for all tasks.

        logits_list = []
        text_features_list = []
        for tid in task_id:
            text_features = self.text_encoder(sentence_embeds_by_task[tid], psudo_sentence_tokens_list[tid])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features.t()
            logits_list.append(logits)
            text_features_list.append(text_features)
        return logits_list, image_features, text_features_list


@MODELS.register_module()
class GraphOrdinalCLIP(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        prompt_learner_cfg,
        graph_encoder_cfg,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            text_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 models, call `clip.models.convert_weights`")

        self.graph_encoder = GRAPH_ENCODERS.build(graph_encoder_cfg)
        self.text_encoder = TextEncoder(clip_model)
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learner: PlainPromptLearner = PROMPT_LEARNERS.build(prompt_learner_cfg)
        self.psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens
        self.logit_scale = clip_model.logit_scale

        self.embed_dims = clip_model.text_projection.shape[1]
        self.graph_feat_dim = getattr(self.graph_encoder, "feat_dim", None)
        if self.graph_feat_dim is None:
            raise ValueError("Graph encoder must expose `feat_dim` attribute.")
        if self.graph_feat_dim != self.embed_dims:
            self.graph_proj = nn.Linear(self.graph_feat_dim, self.embed_dims)
        else:
            self.graph_proj = nn.Identity()
        self.num_ranks = self.prompt_learner.num_ranks

    def encode_graph(self, data):
        graph_features, _ = self.graph_encoder(data)
        graph_features = self.graph_proj(graph_features)
        return graph_features

    def forward(self, data):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        graph_features = self.encode_graph(data)
        graph_features = graph_features / graph_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * graph_features @ text_features.t()

        return logits, graph_features, text_features

    def forward_text_only(self):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)

        return text_features


@MODELS.register_module()
class GraphOrdinalCLIPForMultiTask(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        prompt_learner_cfg,
        graph_encoder_cfg,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            text_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 models, call `clip.models.convert_weights`")

        self.graph_encoder = GRAPH_ENCODERS.build(graph_encoder_cfg)
        self.text_encoder = TextEncoder(clip_model)
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learners = self._build_prompt_learners(prompt_learner_cfg)
        self.psudo_sentence_tokens_list = [pl.psudo_sentence_tokens for pl in self.prompt_learners]
        self.logit_scale = clip_model.logit_scale

        self.embed_dims = clip_model.text_projection.shape[1]
        self.graph_feat_dim = getattr(self.graph_encoder, "feat_dim", None)
        if self.graph_feat_dim is None:
            raise ValueError("Graph encoder must expose `feat_dim` attribute.")
        if self.graph_feat_dim != self.embed_dims:
            self.graph_proj = nn.Linear(self.graph_feat_dim, self.embed_dims)
        else:
            self.graph_proj = nn.Identity()

        self.num_ranks_list = [pl.num_ranks for pl in self.prompt_learners]

    def _build_prompt_learners(self, prompt_learner_cfg):
        assert prompt_learner_cfg["num_tasks"] == len(prompt_learner_cfg["num_ranks"])
        prompt_learners = nn.ModuleList()
        for i in range(prompt_learner_cfg["num_tasks"]):
            prompt_learner_cfg_per_rank = prompt_learner_cfg.copy()
            prompt_learner_cfg_per_rank["num_ranks"] = prompt_learner_cfg["num_ranks"][i]
            prompt_learners.append(PROMPT_LEARNERS.build(prompt_learner_cfg_per_rank))
        return prompt_learners

    def encode_graph(self, data):
        graph_features, _ = self.graph_encoder(data)
        graph_features = self.graph_proj(graph_features)
        graph_features = graph_features / graph_features.norm(dim=-1, keepdim=True)
        return graph_features

    def forward(self, data, task_id=None):
        graph_features = self.encode_graph(data)

        if task_id is None:
            task_id = list(range(len(self.prompt_learners)))
        sentence_embeds_by_task = {tid: self.prompt_learners[tid]() for tid in task_id}
        psudo_sentence_tokens_list = self.psudo_sentence_tokens_list
        logit_scale = self.logit_scale.exp()

        logits_list = []
        text_features_list = []
        for tid in task_id:
            text_features = self.text_encoder(sentence_embeds_by_task[tid], psudo_sentence_tokens_list[tid])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * graph_features @ text_features.t()
            logits_list.append(logits)
            text_features_list.append(text_features)
        return logits_list, graph_features, text_features_list


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype


def load_clip_to_cpu(
    text_encoder_name,
    image_encoder_name,
    root=osp.join(osp.expanduser("~/.cache/clip")),
):
    # text backbone
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print

    print_func("Building CLIP models...")
    text_backbone_name = text_encoder_name
    print_func(f"Text backbone : {text_backbone_name}'s counterpart.")
    url = clip._MODELS[text_backbone_name]
    model_path = clip._download(url, root=root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    # image backbone
    embed_dim = model.text_projection.shape[1]
    input_resolution = model.visual.input_resolution
    image_backbone_name = image_encoder_name
    print_func(f"Image backbone: {image_backbone_name}")

    if image_backbone_name != text_backbone_name:
        # remove the stochastic back-prop in vgg and alexnet
        MODEL = getattr(image_encoders, image_backbone_name, None)
        if MODEL is None:
            MODEL = getattr(models, image_backbone_name, None)
            logger.warning(f"Try PyTorch Official image models: {image_backbone_name}")
        else:
            logger.info(f"Try Custom image models: {image_backbone_name}")
        if MODEL is None:
            raise ValueError(f"Invalid torchvison models name: {image_backbone_name}")
        model.visual = MODEL(num_classes=embed_dim)
        model.visual.input_resolution = input_resolution
    else:
        print_func(f"CLIP Image encoder: {image_backbone_name}!")

    return model
