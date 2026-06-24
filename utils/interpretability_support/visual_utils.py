import pickle
import torch

from models.ordinalclip import OrdinalCLIP
from models.clip_model_utils import load_pretrained_weights
from models.clip_model_utils import load_model

from utils.argparser import load_config
from utils.path_utils import get_data_root


def load_pretrained_model(checkpoint, descriptor="fr_benzene", text_template=None, device="cuda"):
    descriptor_info = pickle.load(open(get_data_root() / "descriptor_info.pkl", "rb"))

    # if text_template is None, model will use the pre-trained text template context
    # else, model will use the provided text_template.
    default_cfg = {
        'type': 'PlainPromptLearner',
        'num_ranks': descriptor_info[descriptor]["num_ranks"],
        'num_base_ranks': 3,
        'num_tokens_per_rank': 1,
        'num_context_tokens': 10,
        'logit_scale': None,
        'rank_tokens_position': 'tail',
        'init_rank_path': None,
        'init_context': text_template,
        'rank_specific_context': False,
        'interpolation_type': 'linear',
        'PREC': 'fp32', }

    model = OrdinalCLIP(text_encoder_name="RN50",
                        image_encoder_name="RN50",
                        prompt_learner_cfg=default_cfg)

    load_pretrained_weights(model,
                            checkpoint,
                            prompt_learnrer_idx=descriptor_info[descriptor]["index"])
    model.float()
    model.eval()
    model.to(device)
    return model


def load_finetuned_model(cfg_path, model_weights_path=None, device="cuda", verbose=False):
    cfg = load_config([cfg_path])
    model = load_model(cfg)
    load_pretrained_weights(model, model_weights_path, verbose=verbose)

    model.float()
    model.eval()
    model.to(device)
    return model


def get_ckpt_epoch(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    return checkpoint["epoch"]
