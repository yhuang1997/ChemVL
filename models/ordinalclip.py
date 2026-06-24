import torch
import torch.nn as nn
import os.path as osp
from models.prompt_leaners.plain_prompt_learner import PlainPromptLearner
from models.build_clip_utils import load_clip_to_cpu


# codes from: https://github.com/xk-huang/OrdinalCLIP
class OrdinalCLIP(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
    ) -> None:
        super().__init__()

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        # convert to float32
        if prompt_learner_cfg["PREC"] == "fp16":
            clip_model.half()
        elif prompt_learner_cfg["PREC"] == "fp32":
            clip_model.float()
        else:
            print(f"Unknown precision {prompt_learner_cfg.PREC}")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.clip_encode_text = clip_model.encode_text
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learner = PlainPromptLearner(**prompt_learner_cfg)
        self.psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens
        self.logit_scale = clip_model.logit_scale

        self.embed_dims = clip_model.text_projection.shape[1]
        self.num_ranks = self.prompt_learner.num_ranks

    def forward(self, images):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features

    def forward_text_only(self,**kwargs):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)

        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)


class OrdinalCLIPForMultiTask(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
    ) -> None:
        super().__init__()

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        # convert to float32
        if prompt_learner_cfg["PREC"] == "fp16":
            clip_model.half()
        elif prompt_learner_cfg["PREC"] == "fp32":
            clip_model.float()
        else:
            print(f"Unknown precision {prompt_learner_cfg.PREC}")

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
            prompt_learners.append(PlainPromptLearner(prompt_learner_cfg_per_rank))
        return prompt_learners

    def forward(self, images, task_id=None):
        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sentence_embeds_list = [pl() for pl in self.prompt_learners]
        psudo_sentence_tokens_list = self.psudo_sentence_tokens_list
        logit_scale = self.logit_scale.exp()  # shared logit scale for all tasks.

        logits_list = []
        text_features_list = []
        for i in range(len(self.prompt_learners)):
            if i in task_id:
                text_features = self.text_encoder(sentence_embeds_list[i], psudo_sentence_tokens_list[i])
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ text_features.t()
                logits_list.append(logits)
                text_features_list.append(text_features)

        return logits_list, image_features, text_features_list


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
