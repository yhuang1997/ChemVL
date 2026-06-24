from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torchvision import transforms

from models.clip_model_utils import load_model, load_pretrained_weights
from utils.argparser import load_config

from analysis.representation_analysis.features.label_parsing import parse_space_separated_multitask_column


def load_finetuned_model(
    cfg_path: str,
    model_weights_path: str,
    device: str = "cuda",
    *,
    pretraining_resume: Optional[str] = None,
    regression_labels_csv: Optional[str] = None,
):
    """Load finetuned CLIP for downstream t-SNE feature extraction."""
    cfg = load_config([cfg_path])
    if pretraining_resume:
        cfg.setdefault("model", {})["resume"] = str(Path(pretraining_resume).expanduser().resolve())
    if regression_labels_csv and str(cfg.get("dataset", {}).get("task_type", "")).lower() == "regression":
        cfg.setdefault("regression_scheduler", {})["labels_csv_path"] = str(
            Path(regression_labels_csv).expanduser().resolve()
        )
    model = load_model(cfg)
    load_pretrained_weights(model, model_weights_path)
    model.float().eval().to(device)
    return model


def _extract_image_text_features(
    model,
    smiles: List[str],
    *,
    targets=None,
    task_id: int = 0,
    cached_image_features: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model.eval().cuda()
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_out: Dict[str, np.ndarray] = {}
    text_out: Dict[str, np.ndarray] = {}
    text_feat: Optional[np.ndarray] = None
    targets_arr: Optional[np.ndarray] = None

    if targets is not None:
        targets_arr = np.asarray(targets)
    elif cached_image_features is not None:
        with torch.no_grad():
            text_feat = model.forward_text_only(task_id=task_id).cpu().numpy()
        text_feat = text_feat / np.linalg.norm(text_feat, axis=-1, keepdims=True)
        cached = np.array([cached_image_features[s] for s in smiles])
        sim = cached @ text_feat.T
        targets_arr = np.argmax(sim, axis=1)

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
        img_t = tfm(img).unsqueeze(0).cuda()
        with torch.no_grad():
            im = model.encode_image(img_t).cpu().numpy()[0]
        im = im / np.linalg.norm(im, axis=-1, keepdims=True)
        image_out[smi] = im

    if targets_arr is None and cached_image_features is None:
        with torch.no_grad():
            text_feat = model.forward_text_only(task_id=task_id).cpu().numpy()
        text_feat = text_feat / np.linalg.norm(text_feat, axis=-1, keepdims=True)
        stacked = np.stack([image_out[s] for s in smiles], axis=0)
        sim = stacked @ text_feat.T
        targets_arr = np.argmax(sim, axis=1)

    if text_feat is not None and targets_arr is not None:
        for i, smi in enumerate(smiles):
            text_out[smi] = text_feat[int(targets_arr[i])]

    return image_out, text_out


def build_features_from_preset(preset: Dict[str, Any]) -> Dict[str, Any]:
    stage = str(preset["mode"]).lower()
    if stage != "downstream":
        raise ValueError(f"Only mode=downstream is supported (got {stage!r})")

    common = preset.get("common", {})
    seed = int(common.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)

    cache_base = Path(str(common.get("cache_base", "./cache"))).expanduser()
    cache_base.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reducer_cfg = preset.get("reducer", {"name": "tsne", "params": {}})
    out: Dict[str, Any] = {"mode": stage, "reducer_cfg": reducer_cfg, "records": []}

    dcfg = preset["downstream"]
    data = pd.read_csv(dcfg["downstream_csv_file"])
    smiles = data["smiles"].astype(str).tolist()
    targets = data[dcfg.get("label_col", "label")].to_numpy()
    task_id = int(dcfg.get("task_id", 0))
    if dcfg.get("label_parse_space_separated", False):
        targets = parse_space_separated_multitask_column(targets, task_id)
    targets = pd.to_numeric(pd.Series(np.asarray(targets).ravel()), errors="coerce").to_numpy(dtype=np.float64)

    backbone = common.get("pretraining_resume")
    if not backbone:
        raise ValueError(
            "downstream preset needs common.pretraining_resume: absolute path to backbone weights "
            "for load_model(cfg) (config.json model.resume is often an invalid Windows path on WSL)."
        )
    reg_csv = str(Path(dcfg["downstream_csv_file"]).expanduser().resolve())
    model = load_finetuned_model(
        dcfg["finetune_cfg_path"],
        common["model_ckpt"],
        device=device,
        pretraining_resume=backbone,
        regression_labels_csv=reg_csv,
    )
    image_map, text_map = _extract_image_text_features(model, smiles, targets=None, task_id=task_id)
    image_features = np.stack([image_map[s] for s in smiles], axis=0)
    text_features = np.stack([text_map[s] for s in smiles], axis=0)

    out["records"].append(
        {
            "stage": "downstream",
            "dataset": str(dcfg.get("dataset_name", "downstream")),
            "descriptor": None,
            "smiles": smiles,
            "image_features": image_features,
            "text_features": text_features,
            "targets": np.asarray(targets),
            "combined_alpha_list": list(dcfg.get("combined_alpha_list", [0.0, 0.15, 0.5])),
            "task_id": task_id,
            "text_target_mode": common.get("text_target_mode", "maxScore"),
            "source_refs": {
                "downstream_csv_file": dcfg["downstream_csv_file"],
                "finetune_cfg_path": dcfg["finetune_cfg_path"],
                "model_ckpt": common.get("model_ckpt"),
                "cache_base": str(cache_base),
            },
        }
    )
    return out
