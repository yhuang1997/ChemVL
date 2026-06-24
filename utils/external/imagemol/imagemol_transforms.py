# SPDX-License-Identifier: MIT
"""Official ImageMol transforms (CenterCrop train/val; aug on train only)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from torchvision import transforms


def build_imagemol_transforms(cfg: Dict[str, Any], *, train: bool) -> transforms.Compose:
    image_size = int((cfg.get("model") or {}).get("imageSize") or 224)
    aug_cfg = cfg.get("data_augmentation") or {}
    use_aug = bool(aug_cfg.get("image_aug")) and train

    parts = [transforms.CenterCrop(image_size)]
    if use_aug:
        parts.extend(
            [
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(degrees=360),
                transforms.RandomHorizontalFlip(),
            ]
        )
    parts.append(transforms.ToTensor())
    mean = aug_cfg.get("Normalize_mean") or [0.485, 0.456, 0.406]
    std = aug_cfg.get("Normalize_std") or [0.229, 0.224, 0.225]
    parts.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(parts)


def build_imagemol_transform_triplet(cfg: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    train_t = build_imagemol_transforms(cfg, train=True)
    eval_t = build_imagemol_transforms(cfg, train=False)
    return train_t, eval_t, eval_t
