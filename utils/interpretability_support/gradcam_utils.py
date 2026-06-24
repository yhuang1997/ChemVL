from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as ssim
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputTarget
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms.functional as TF
import cv2
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import LinearSegmentedColormap


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, smiles=None):
        """
        :param images: list of images (already loaded), e.g., [Image1, Image2, ..., ImageN]
        :param labels: list of labels, e.g., single label: [[1], [0], [2]]; multi-labels: [[0, 1, 0], ..., [1,1,0]]
        :param img_transformer: optional image transformer (e.g., torchvision.transforms)
        :param normalize: optional normalization function
        :param ret_index: whether to return the index
        :param smiles: optional SMILES strings corresponding to the images
        """
        self.images = images
        self.labels = labels
        self.total = len(self.images)
        self._image_transformer = transform
        self.smiles = smiles

    def __getitem__(self, index):
        img = self.images[index]
        if self._image_transformer is not None:
            img = self._image_transformer(img)

        if self.smiles is not None:
            return img, self.labels[index], self.smiles[index]
        else:
            return img, self.labels[index]

    def __len__(self):
        return self.total

def typing_batch_index_in_rgb_images(rgb_images, font_color=(255, 0, 0), font_size=12):
    typing_rgb_images = []
    for idx, image in enumerate(rgb_images):
        image_copy = image.copy()
        # Convert to PIL Image
        draw = ImageDraw.Draw(image_copy)

        text = f"#{idx + 1}"
        text_position = (10, 10)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()  # Use default font

        draw.text(text_position, text, fill=font_color, font=font)

        rgb_image = (np.array(image) / 255).astype(np.float32)
        typing_rgb_images.append(rgb_image)

    return typing_rgb_images



def get_target_fn(target_fn):
    if target_fn == "plain":
        return ClassifierOutputTarget
    elif target_fn == "softmax":
        return ClassifierOutputSoftmaxTarget
    elif target_fn == "softmax_kl_ce":
        return KLDivergenceCrossEntropyOutPutTarget
    elif target_fn == "argmax":
        return ArgmaxOutPutSoftmaxTarget
    elif target_fn == "rmse":
        return RMSEOutPutTarget
    elif target_fn == "scalar":
        return RegressionScalarTarget
    else:
        raise ValueError("Invalid target function!")


class KLDivergenceCrossEntropyOutPutTarget(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, output):
        num_ranks = output.shape[-1]
        ce_loss_weight = 1.0
        kl_loss_weight = 1.0
        ce_loss_func = nn.CrossEntropyLoss()
        kl_loss_func = nn.KLDivLoss(reduction="sum")

        y = torch.tensor(self.target, dtype=torch.long).to(output.device)
        # Cross Entropy Loss
        ce_loss = ce_loss_func(output, y)

        # KL Divergence Loss
        y_t = F.one_hot(y, num_ranks).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        y_t_reduction = (y_t * 10.0).softmax(-1)
        y_t_reduction[y_t_row_ind <= 0] = 0

        logits_t = output.t()
        kl_loss = kl_loss_func(F.log_softmax(logits_t, dim=-1), y_t_reduction) / num_slots

        return ce_loss_weight * ce_loss + kl_loss_weight * kl_loss


class ArgmaxOutPutSoftmaxTarget(nn.Module):
    def __init__(self, target=None):
        super().__init__()
        self.target = target  # This target will not be used
    def forward(self, output):
        target_categories = np.argmax(output.cpu().data.numpy(), axis=-1)
        logits = F.softmax(output, dim=-1)
        return logits[target_categories]


class RMSEOutPutTarget(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, output):
        return (output - self.target) ** 2


class RegressionScalarTarget(nn.Module):
    """Backprop through the regression scalar output directly (no GT needed)."""

    def __init__(self, target=None):
        super().__init__()

    def forward(self, output):
        return output


class ActivationsAndGradientsForCLIP(ActivationsAndGradients):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        super().__init__(model, target_layers, reshape_transform)

    def __call__(self, image, text):
        self.gradients = []
        self.activations = []
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Encode text features using CLIP instead of the model's text encoder
        text_features = self.model.clip_encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


class GradCAMForCLIP(GradCAM):
    def __init__(self, model, target_layers):
        super().__init__(model, target_layers)
        self.activations_and_grads = ActivationsAndGradientsForCLIP(
            self.model, target_layers, self.reshape_transform)

    def __call__(self,
                 input_tensor: List[torch.Tensor],
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        assert not aug_smooth, "Augmentation smoothing is not supported for GradCAM"

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def get_target_width_height(self, input_tensor: List[torch.Tensor]) -> Tuple[int, int]:
        input_image_tensor = input_tensor[0]
        width, height = input_image_tensor.size(-1), input_image_tensor.size(-2)
        return width, height

    def forward(self,
                input_tensor: List[torch.Tensor],
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        input_image_tensor = input_tensor[0]
        input_text_tensor = input_tensor[1]

        input_text_tensor = input_text_tensor.to(self.device)
        input_image_tensor = input_image_tensor.to(self.device)

        if self.compute_input_gradient:
            input_image_tensor = torch.autograd.Variable(input_image_tensor,
                                                         requires_grad=True)
            input_text_tensor = torch.autograd.Variable(input_text_tensor,
                                                        requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_image_tensor, input_text_tensor)

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                        for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)
        cam_per_layer = self.compute_cam_per_layer(input_image_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)


class GradCAMForOrdinalCLIP(GradCAM):
    def __init__(self, model, target_layers, task_id=0):
        super().__init__(model, target_layers)
        self.task_id = task_id
        # self.uses_gradients = False

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> tuple[np.ndarray, int]:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)

        # NOTE: This is a workaround to handle the case where the model outputs a single tensor. Consider refactoring for better handling.
        try:
            logits, image_features, text_features = outputs
            self.outputs = logits.detach().cpu().numpy()
        except ValueError:
            logits = outputs
            self.outputs = logits.detach().cpu().numpy()
            if len(logits.shape) == 3:
                logits = logits[:, :, self.task_id]
            # regression task
            elif len(logits.shape) == 2:
                logits = logits[:, self.task_id]


        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                        for target, output in zip(targets, logits)])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)


# Showing the metrics on top of the CAM :
def visualize_score(visualization, name, descriptor=None, target=None, pred=None, logits=None, text_template=None,
                    ssim_info=None):
    # Define constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Adjust font scale for 224*224 images
    font_color = (255, 255, 255)  # White text
    line_color = (0, 0, 0)  # Black border for text
    thickness = 1
    line_spacing = 20  # Space between lines

    # Define starting positions
    x, y = 10, 12  # Starting position for text

    # Helper function to draw text with border
    def put_text_with_border(image, text, position, font, font_scale, font_color, line_color, thickness):
        x, y = position
        cv2.putText(image, text, (x, y), font, font_scale, line_color, thickness + 1, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # # Add text elements with consistent spacing
    # put_text_with_border(visualization, f"Name: {name}", (x, y), font, font_scale, font_color, line_color, thickness)
    # y += line_spacing

    if descriptor is not None:
        put_text_with_border(visualization, f"Descriptor: {descriptor}", (x, y), font, font_scale, font_color,
                             line_color, thickness)
        y += line_spacing

    if logits is not None:
        prediction_text = f"Prediction: {pred} -> {target} ({logits:.2f})"
    else:
        prediction_text = f"Prediction: {pred:.2f} -> {target:.2f}"
    put_text_with_border(visualization, prediction_text, (x, y), font, font_scale, font_color, line_color, thickness)
    y += line_spacing

    if text_template is not None:
        put_text_with_border(visualization, f"Prompt: {text_template[:30]}...", (x, y), font, font_scale, font_color,
                             line_color, thickness)
        y += line_spacing

    if ssim_info is not None:
        put_text_with_border(visualization, f"{ssim_info}", (x, y), font, font_scale, font_color, line_color,
                             thickness)

    return visualization


def _ssim_per_sample(
    original: np.ndarray,
    hflip: np.ndarray,
    vflip: np.ndarray,
    hvflip: np.ndarray,
) -> Dict[str, float]:
    """SSIM between base attribution and flip-augmented variants (one sample)."""
    data_range = float(original.max() - original.min())
    if data_range <= 0:
        data_range = 1.0
    ssim_h = float(ssim(original, hflip, data_range=data_range))
    ssim_v = float(ssim(original, vflip, data_range=data_range))
    ssim_hv = float(ssim(original, hvflip, data_range=data_range))
    vals = [ssim_h, ssim_v, ssim_hv]
    return {
        "ssim_hflip": ssim_h,
        "ssim_vflip": ssim_v,
        "ssim_hvflip": ssim_hv,
        "ssim_mean": float(np.mean(vals)),
        "ssim_std": float(np.std(vals)),
    }


def benchmark(model, rgb_images, input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=None, info=None,
              target_fn_name="plain", task_type="classification", task_id=0, typing=False,
              return_records: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Any]]]]:
    # note: do not use aug_smooth in pytorch_grad_cam, toooooooo slow!!
    methods = [
        ("GradCAM", GradCAMForOrdinalCLIP(model=model, target_layers=target_layers, task_id=task_id)),
        # ("GradCAM++", GradCAMPlusPlusForOrdinalCLIP(model=model, target_layers=target_layers)),
        # ("EigenGradCAM", EigenGradCAMForOrdinalCLIP(model=model, target_layers=target_layers)),
        # ("EigenCAM", EigenCAMForOrdinalCLIP(model=model, target_layers=target_layers)),
        # ("AblationCAM", AblationCAMForOrdinalCLIP(model=model, target_layers=target_layers)),
    ]

    rows = len(rgb_images)
    cols = len(methods)
    resolution = 224

    if category is not None:
        assert target_fn_name in ["plain", "softmax", "softmax_kl_ce", "argmax", "rmse"], "Invalid target function name!"
        target_fn = get_target_fn(target_fn_name)
        # please refer to the function document of `model_targets` in `pytorch_grad_cam`

        targets = [target_fn(c) for c in category]

    else:
        targets = None

    images_array = np.zeros((rows * resolution, cols * resolution, 3), dtype=np.uint8)
    records: List[Dict[str, Any]] = []
    nan_ssim = {
        "ssim_hflip": float("nan"),
        "ssim_vflip": float("nan"),
        "ssim_hvflip": float("nan"),
        "ssim_mean": float("nan"),
        "ssim_std": float("nan"),
    }

    for j, (name, cam_method) in enumerate(methods):
        with cam_method:
            attributions = cam_method(input_tensor=input_tensor,
                                      targets=targets,
                                      eigen_smooth=eigen_smooth,
                                      aug_smooth=False)

            per_sample_ssim: List[Dict[str, float]] = [dict(nan_ssim) for _ in range(rows)]
            if aug_smooth:
                input_tensor_hflip = TF.hflip(input_tensor)
                attributions_hflip = cam_method(input_tensor=input_tensor_hflip,
                                                targets=targets,
                                                eigen_smooth=eigen_smooth,
                                                aug_smooth=False)
                attributions_hflip_back = TF.hflip(torch.from_numpy(attributions_hflip)).numpy()

                input_tensor_vflip = TF.vflip(input_tensor)
                attributions_vflip = cam_method(input_tensor=input_tensor_vflip,
                                                targets=targets,
                                                eigen_smooth=eigen_smooth,
                                                aug_smooth=False)
                attributions_vflip_back = TF.vflip(torch.from_numpy(attributions_vflip)).numpy()

                input_tensor_hvflip = TF.hflip(input_tensor_vflip)
                attributions_hvflip = cam_method(input_tensor=input_tensor_hvflip,
                                                 targets=targets,
                                                 eigen_smooth=eigen_smooth,
                                                 aug_smooth=False)
                attributions_hvflip_back = TF.vflip(TF.hflip(torch.from_numpy(attributions_hvflip))).numpy()

                for k in range(attributions.shape[0]):
                    per_sample_ssim[k] = _ssim_per_sample(
                        attributions[k],
                        attributions_hflip_back[k],
                        attributions_vflip_back[k],
                        attributions_hvflip_back[k],
                    )

                attributions = (attributions + attributions_hflip_back +
                                    attributions_vflip_back + attributions_hvflip_back) / 4

            if isinstance(cam_method.outputs, tuple):
                pred_logits = cam_method.outputs[task_id]
            else:
                pred_logits = cam_method.outputs
                if task_id is not None and getattr(pred_logits, "ndim", 0) > 1:
                    pred_logits = np.take(pred_logits, task_id, axis=-1)
            if task_type == "classification":
                pred_categories = np.argmax(pred_logits, axis=-1)
                softmax_scores = torch.softmax(torch.tensor(pred_logits), dim=-1)
                scores = softmax_scores[range(rows), pred_categories]
            else:
                pred_categories = np.asarray(pred_logits).reshape(-1)
                scores = None

            gt_values: Optional[List[float]] = None
            if category is not None:
                gt_values = [float(category[i]) for i in range(rows)]
            elif info is not None and info.get("descriptor_targets") is not None:
                gt_values = [float(info["descriptor_targets"][i]) for i in range(rows)]

            for i in range(rows):
                visualization = show_cam_on_image(rgb_images[i], attributions[i], use_rgb=True)
                if info is not None:
                    target = info["descriptor_targets"][i] if info.get("descriptor_targets") is not None else None
                    descriptor = info.get("descriptor", None)
                    ssim_entry = per_sample_ssim[i]
                    if aug_smooth:
                        ssim_info = (
                            f"SSIM: {ssim_entry['ssim_mean']:.2f}({ssim_entry['ssim_std']:.2f})"
                        )
                    else:
                        ssim_info = None
                    if typing:
                        visualization = visualize_score(visualization, name,
                                                        descriptor=descriptor,
                                                        pred=pred_categories[i],
                                                        logits=scores[i].item() if scores is not None else None,
                                                        target=target,
                                                        ssim_info=ssim_info,
                                                        text_template=info.get("text_template"))
                images_array[i * resolution:(i + 1) * resolution, j * resolution:(j + 1) * resolution] = visualization

                if return_records:
                    pred_val = float(np.asarray(pred_categories[i]).reshape(()))
                    gt_val = float(gt_values[i]) if gt_values is not None else float("nan")
                    rec = {
                        "pred": pred_val,
                        "gt": gt_val,
                        "abs_error": abs(pred_val - gt_val) if gt_values is not None else float("nan"),
                        **per_sample_ssim[i],
                        "panel": images_array[i * resolution:(i + 1) * resolution, :].copy(),
                    }
                    records.append(rec)

    if return_records:
        return images_array, records
    return images_array

def benchmarkv2(model, rgb_images, input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=None, info=None,
                target_fn_name="plain", task_type="classification", task_id=0, typing=False, cmap_style="base"):
    # note: do not use aug_smooth in pytorch_grad_cam, toooooooo slow!!
    methods = [
        ("GradCAM", GradCAMForOrdinalCLIP(model=model, target_layers=target_layers, task_id=task_id)),
        # ("GradCAM++", GradCAMPlusPlusForOrdinalCLIP(model=model, target_layers=target_layers)),
        # ("EigenGradCAM", EigenGradCAMForOrdinalCLIP(model=model, target_layers=target_layers)),
        # ("EigenCAM", EigenCAMForOrdinalCLIP(model=model, target_layers=target_layers)),
        # ("AblationCAM", AblationCAMForOrdinalCLIP(model=model, target_layers=target_layers)),
    ]

    rows = len(rgb_images)
    cols = len(methods)
    resolution = 224

    if category is not None:
        assert target_fn_name in ["plain", "softmax", "softmax_kl_ce", "argmax", "rmse"], "Invalid target function name!"
        target_fn = get_target_fn(target_fn_name)
        # please refer to the function document of `model_targets` in `pytorch_grad_cam`

        targets = [target_fn(c) for c in category]

    else:
        targets = None

    images_array = np.zeros((rows * resolution, cols * resolution, 3), dtype=np.uint8)
    attributions_list = []
    for j, (name, cam_method) in enumerate(methods):
        with cam_method:
            attributions = cam_method(input_tensor=input_tensor,
                                      targets=targets,
                                      eigen_smooth=eigen_smooth,
                                      aug_smooth=False)

            if aug_smooth:
                input_tensor_hflip = TF.hflip(input_tensor)
                attributions_hflip = cam_method(input_tensor=input_tensor_hflip,
                                                targets=targets,
                                                eigen_smooth=eigen_smooth,
                                                aug_smooth=False)
                attributions_hflip_back = TF.hflip(torch.from_numpy(attributions_hflip)).numpy()

                input_tensor_vflip = TF.vflip(input_tensor)
                attributions_vflip = cam_method(input_tensor=input_tensor_vflip,
                                                targets=targets,
                                                eigen_smooth=eigen_smooth,
                                                aug_smooth=False)
                attributions_vflip_back = TF.vflip(torch.from_numpy(attributions_vflip)).numpy()

                input_tensor_hvflip = TF.hflip(input_tensor_vflip)
                attributions_hvflip = cam_method(input_tensor=input_tensor_hvflip,
                                                 targets=targets,
                                                 eigen_smooth=eigen_smooth,
                                                 aug_smooth=False)
                attributions_hvflip_back = TF.vflip(TF.hflip(torch.from_numpy(attributions_hvflip))).numpy()

                ssim_results = {'ssim_hflip': [], 'ssim_vflip': [], 'ssim_hvflip': []}
                for k in range(attributions.shape[0]):
                    original = attributions[k]
                    hflip = attributions_hflip_back[k]
                    vflip = attributions_vflip_back[k]
                    hvflip = attributions_hvflip_back[k]

                    # Compute SSIM value for each sample
                    ssim_hflip = ssim(original, hflip,
                                      data_range=original.max() - original.min())

                    ssim_vflip = ssim(original, vflip,
                                      data_range=original.max() - original.min())

                    ssim_hvflip = ssim(original, hvflip,
                                       data_range=original.max() - original.min())

                    # Store results in dictionary
                    ssim_results['ssim_hflip'].append(ssim_hflip)
                    ssim_results['ssim_vflip'].append(ssim_vflip)
                    ssim_results['ssim_hvflip'].append(ssim_hvflip)

                # Average the results of four transformations as the final result
                attributions = (attributions + attributions_hflip_back +
                                    attributions_vflip_back + attributions_hvflip_back) / 4
                attributions_list.append(attributions)

            if isinstance(cam_method.outputs, tuple):
                pred_logits = cam_method.outputs[task_id]
            else:
                pred_logits = cam_method.outputs
                if task_id is not None:
                    pred_logits = np.take(pred_logits, task_id, axis=-1)
            if task_type == "classification":
                pred_categories = np.argmax(pred_logits, axis=-1)
                softmax_scores = torch.softmax(torch.tensor(pred_logits), dim=-1)
                scores = softmax_scores[range(rows), pred_categories]
            else:
                pred_categories = pred_logits
                scores = None
            for i in range(rows):
                visualization = custom_show_cam_on_image(rgb_images[i], attributions[i], use_rgb=True, cmap_style=cmap_style)
                if info is not None:
                    target = info["descriptor_targets"][i] if info["descriptor_targets"] is not None else None
                    descriptor = info.get("descriptor", None)
                    if aug_smooth:
                        this_ssim = [x[i] for x in ssim_results.values()]
                        ssim_mean, ssim_std = np.mean(this_ssim), np.std(this_ssim)
                        ssim_info = f"SSIM: {ssim_mean:.2f}({ssim_std:.2f})"
                    else:
                        ssim_info = None
                    if typing:
                        visualization = visualize_score(visualization, name,
                                                        descriptor=descriptor,
                                                        pred=pred_categories[i],
                                                        logits=scores[i].item() if scores is not None else None,
                                                        target=target,
                                                        ssim_info=ssim_info,
                                                        text_template=info["text_template"])
                images_array[i * resolution:(i + 1) * resolution, j * resolution:(j + 1) * resolution] = visualization
    return images_array, attributions_list


def resolve_normalize_mode(normalize: Any) -> str:
    """Map preset/CLI values to ``none``, ``minmax``, or ``percentile``."""
    if normalize is None or normalize is False:
        return "none"
    if isinstance(normalize, str):
        key = normalize.strip().lower()
        if key in ("", "false", "none", "off", "0"):
            return "none"
        if key in ("true", "on", "1", "percentile"):
            return "percentile"
        if key in ("minmax", "maxmin"):
            return "minmax"
        return key
    if normalize is True:
        return "percentile"
    raise ValueError(f"Unknown normalize value: {normalize!r}")


def parse_normalize_percentile(
    raw: Any,
    *,
    default_low: float = 2.0,
    default_high: float = 98.0,
) -> Tuple[float, float]:
    if raw is None:
        return default_low, default_high
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    raise ValueError(f"normalize_percentile must be [low, high], got {raw!r}")


def normalize_attribution_for_display(
    mask: np.ndarray,
    *,
    mode: str = "none",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Stretch a CAM mask for colormap display. Does not modify stored ``attributions.npy``.

    :param mode: ``none`` | ``minmax`` | ``percentile``
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    mode = resolve_normalize_mode(mode)
    if mode == "none":
        out = np.asarray(mask, dtype=np.float32)
        return np.clip(out, 0.0, 1.0)

    m = np.asarray(mask, dtype=np.float32)
    if mode == "minmax":
        lo = float(m.min())
        hi = float(m.max())
        if hi - lo < eps:
            return np.zeros_like(m)
        return np.clip((m - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)

    if mode == "percentile":
        lo = float(np.percentile(m, percentile_low))
        hi = float(np.percentile(m, percentile_high))
        if hi - lo < eps:
            return np.zeros_like(m)
        return np.clip((m - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unknown normalize mode: {mode!r}")


def upscale_attribution(attribution: np.ndarray, size=(224, 224), method: str = "lanczos", smooth: bool = True) -> np.ndarray:
    """
    Upscale an attribution or Grad-CAM mask to a target size with high-quality interpolation.

    :param attribution: Input mask (numpy array, float32, range [0, 1])
    :param size: Target size (width, height)
    :param method: Interpolation method: 'lanczos', 'cubic', or 'linear'
    :param smooth: Whether to apply a light Gaussian blur after upscaling
    :return: Upscaled mask (float32, range [0, 1])
    """
    if attribution.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {attribution.shape}")

    # choose interpolation
    if method == "lanczos":
        interp = cv2.INTER_LANCZOS4
    elif method == "cubic":
        interp = cv2.INTER_CUBIC
    elif method == "linear":
        interp = cv2.INTER_LINEAR
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # resize (OpenCV expects (width, height))
    upscaled = cv2.resize(attribution, size, interpolation=interp)

    # optional smoothing for soft boundaries
    if smooth:
        upscaled = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=0.8, sigmaY=0.8)

    # normalize to [0, 1]
    upscaled = np.clip(upscaled, 0, 1)
    if upscaled.dtype != np.float32:
        upscaled = upscaled.astype(np.float32)

    return upscaled


def custom_show_cam_on_image(img: np.ndarray,
                             mask: np.ndarray,
                             use_rgb: bool = False,
                             image_weight: float = 0.5,
                             cmap_style: str = "yor") -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :param cmap_style: Color style ('yor' for yellow–orange–red, 'jet_white' for white–jet mix)
    :returns: The default image with the cam overlay.
    """

    # Colormap selection
    if cmap_style == "base":
        return show_cam_on_image(img, mask, use_rgb=use_rgb)
    elif cmap_style == "yor":
        cmap = LinearSegmentedColormap.from_list(
            "yellow_orange_red",
            [
                (0.0,  (1.0, 1.0, 1.0, 0.0)),   # white (low)
                (0.25, (1.0, 0.95, 0.7, 1.0)),  # light yellow
                (0.5,  (1.0, 0.75, 0.3, 1.0)),  # orange
                (1.0,  (0.85, 0.0, 0.0, 1.0))   # red
            ]
        )
    elif cmap_style == "jet_white":
        # JET colormap with low values faded to white
        jet = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)
        jet = jet.squeeze().astype(np.float32)[::-1] / 255.0  # reverse colormap order
        n = len(jet)
        white = np.array([1.0, 1.0, 1.0])
        fade_ratio = 0.3
        fade_end = int(n * fade_ratio)
        for i in range(fade_end):
            t = i / fade_end
            jet[i] = (1 - t) * white + t * jet[i]
        cmap = LinearSegmentedColormap.from_list("jet_white", jet)
    else:
        raise ValueError(f"Unknown cmap_style: {cmap_style}")

    # Build RGBA heatmap
    heatmap_rgba = cmap(mask)
    alpha = np.expand_dims(mask, axis=-1)
    heatmap_rgb = np.delete(heatmap_rgba, 3, axis=2)
    heatmap_rgb = np.float32(heatmap_rgb)

    # BGR input conversion when needed
    if not use_rgb:
        heatmap_rgb = cv2.cvtColor((heatmap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        heatmap_rgb = np.float32(heatmap_rgb) / 255

    # Alpha blend overlay
    cam = (1 - alpha) * img + alpha * heatmap_rgb
    cam = image_weight * img + (1 - image_weight) * cam
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)