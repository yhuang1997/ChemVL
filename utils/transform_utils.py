from __future__ import annotations

from torchvision import transforms


def get_transforms(aug_cfg):
    # Define the transformations for the training set
    transform_list = []

    if 'Resize' in aug_cfg and aug_cfg['Resize'] is not None:
        transform_list.append(transforms.Resize(aug_cfg['Resize']))
    if 'CenterCrop' in aug_cfg and aug_cfg['CenterCrop'] is not None:
        transform_list.append(transforms.CenterCrop(aug_cfg['CenterCrop']))

    # Geometric transformation
    if 'HorizontalFlip' in aug_cfg and aug_cfg['HorizontalFlip'] is not None:
        prob = aug_cfg['HorizontalFlip'].get('p', 1.0)
        transform_list.append(transforms.RandomHorizontalFlip(p=prob))
    if 'VerticalFlip' in aug_cfg and aug_cfg['VerticalFlip'] is not None:
        prob = aug_cfg['VerticalFlip'].get('p', 1.0)
        transform_list.append(transforms.RandomVerticalFlip(p=prob))
    if 'Rotation' in aug_cfg and aug_cfg['Rotation'] is not None:
        degrees = aug_cfg['Rotation']['degrees']
        prob = aug_cfg['Rotation'].get('p', 1.0)
        transform_list.append(transforms.RandomApply([transforms.RandomRotation(degrees=degrees, fill=(255, 255, 255))],
                                                     p=prob))

    if 'RandomResizedCrop' in aug_cfg and aug_cfg['RandomResizedCrop'] is not None:
        size = aug_cfg['RandomResizedCrop']['size']
        scale = aug_cfg['RandomResizedCrop']['scale']
        ratio = aug_cfg['RandomResizedCrop']['ratio']
        prob = aug_cfg['RandomResizedCrop'].get('p', 1.0)
        transform_list.append(transforms.RandomApply([transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio)],
                                                     p=prob))
    # Color jitter
    if 'ColorJitter' in aug_cfg and aug_cfg['ColorJitter'] is not None:
        brightness = aug_cfg['ColorJitter'].get('brightness', 0)
        contrast = aug_cfg['ColorJitter'].get('contrast', 0)
        saturation = aug_cfg['ColorJitter'].get('saturation', 0)
        hue = aug_cfg['ColorJitter'].get('hue', 0)
        prob = aug_cfg['ColorJitter'].get('p', 1.0)
        transform_list.append(transforms.RandomApply(
            [transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)], p=prob))

    # Gaussian blur
    if 'GaussianBlur' in aug_cfg and aug_cfg['GaussianBlur'] is not None:
        kernel_size = aug_cfg['GaussianBlur'].get('kernel_size', 3)
        sigma = aug_cfg['GaussianBlur'].get('sigma', (0.1, 2.0))
        prob = aug_cfg['GaussianBlur'].get('p', 1.0)
        transform_list.append(
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)], p=prob))

    # Random grayscale
    if 'RandomGrayscale' in aug_cfg and aug_cfg['RandomGrayscale'] is not None:
        prob = aug_cfg['RandomGrayscale'].get('p', 1.0)
        transform_list.append(transforms.RandomApply([transforms.RandomGrayscale(p=prob)], p=prob))

    # ToTensor and Normalization is the last step of the transformation
    transform_list.append(transforms.ToTensor())

    if 'Normalize_mean' in aug_cfg and 'Normalize_std' in aug_cfg:
        transform_list.append(transforms.Normalize(mean=aug_cfg['Normalize_mean'],
                                                   std=aug_cfg['Normalize_std']))
    else:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)


def get_default_transforms():
    # This is standard transforms in CLIP
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
