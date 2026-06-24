from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw


import torch
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from PIL import Image
from io import BytesIO


def _to_pil(img) -> Image.Image:
    """Convert RDKit / IPython display images to PIL.Image.Image."""
    if isinstance(img, Image.Image):
        return img

    data = getattr(img, "data", None)
    if isinstance(data, (bytes, bytearray)):
        return Image.open(BytesIO(data)).convert("RGB")

    raise TypeError(f"Unexpected image type: {type(img)}")


def smiles_to_pil_images(smiles: List[str] | np.ndarray, size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
    """Render SMILES to PIL images via RDKit."""
    pil_images: List[Image.Image] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pil_images.append(Image.new("RGB", size, color=(255, 255, 255)))
            continue
        im = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=size)
        pil_images.append(_to_pil(im).convert("RGB"))
    return pil_images


def pil_list_to_input_tensor(
    pil_images: List[Image.Image],
    device: str = "cuda",
) -> torch.Tensor:
    """Apply CLIP-style transform and stack into a tensor on `device`."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    x = torch.stack([transform(im) for im in pil_images], dim=0)
    return x.to(device)


@torch.no_grad()
def load_features(
    model,
    smiles: List[str] | np.ndarray,
    targets: Optional[List[int] | np.ndarray] = None,
    batch_size: int = 256,
    load_image_feature: bool = True,
    load_text_feature: bool = True,
    cached_image_features: Optional[Dict[str, np.ndarray]] = None,
    taskid: int = 0,
    device: str = "cuda",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Faithful to your original pipeline:
    - image: model.encode_image( transformed_mol_images )
    - text: model.forward_text_only(task_id=taskid) gives a bank of text prototypes (K, D)
    - maxScore: targets = argmax(image_features @ text_features.T)
      then each SMILES gets the corresponding prototype text feature.
    """
    model.eval()
    model = model.to(device)

    smiles = list(smiles)
    n_samples = len(smiles)

    smiles2image_features: Dict[str, np.ndarray] = {}
    smiles2text_features: Dict[str, np.ndarray] = {}

    assert load_image_feature or load_text_feature, "Must load at least one type of feature."
    if load_text_feature and targets is None:
        assert cached_image_features is not None, "cached_image_features must be provided if targets=None (maxScore mode)."

    if load_text_feature:
        text_bank = model.forward_text_only(task_id=taskid).detach().cpu().numpy()
        text_bank = text_bank / (np.linalg.norm(text_bank, axis=-1, keepdims=True) + 1e-12)

        if targets is None:
            img_mat = np.stack([cached_image_features[smi] for smi in smiles], axis=0)  # (N, D)
            sim = img_mat @ text_bank.T  # (N, K)
            targets = np.argmax(sim, axis=1)

        targets = np.asarray(targets)

    for i in range(0, n_samples, batch_size):
        batch_smiles = smiles[i:i + batch_size]

        if load_image_feature:
            pil_images = smiles_to_pil_images(batch_smiles, size=(224, 224))
            x = pil_list_to_input_tensor(pil_images, device=device)

            feats = model.encode_image(x).detach().cpu().numpy()
            feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-12)
            for j, smi in enumerate(batch_smiles):
                smiles2image_features[smi] = feats[j]

        if load_text_feature:
            batch_targets = targets[i:i + batch_size]
            proto = text_bank[batch_targets]  # (B, D)
            for j, smi in enumerate(batch_smiles):
                smiles2text_features[smi] = proto[j]

    return smiles2image_features, smiles2text_features


def array_to_smiles_dict(smiles: List[str] | np.ndarray, feats: np.ndarray) -> Dict[str, np.ndarray]:
    """Map SMILES -> feature vector."""
    smiles = list(smiles)
    assert len(smiles) == feats.shape[0]
    return {s: feats[i] for i, s in enumerate(smiles)}


def extract_image_features(model, smiles: List[str] | np.ndarray, batch_size: int = 256, device: str = "cuda") -> np.ndarray:
    """Return (N, D) image embeddings."""
    img_dict, _ = load_features(
        model,
        smiles,
        load_image_feature=True,
        load_text_feature=False,
        batch_size=batch_size,
        device=device,
    )
    return np.stack([img_dict[s] for s in list(smiles)], axis=0)


def extract_text_features(
    model,
    smiles: List[str] | np.ndarray,
    image_features_dict: Dict[str, np.ndarray],
    taskid: int = 0,
    batch_size: int = 256,
    device: str = "cuda",
    descriptor: Optional[str] = None,
    text_target_mode: str = "maxScore",
) -> np.ndarray:
    """
    Return (N, D) text embeddings under maxScore mode.

    Note: descriptor conditioning should be baked into `model`
    (create via `load_pretrained_model(..., descriptor=d)`).
    """
    if descriptor is not None:
        _ = str(descriptor)

    if text_target_mode == "maxScore":
        targets = None
    elif text_target_mode == "zeroTarget":
        targets = [0 for _ in range(len(smiles))]
    elif text_target_mode == "gtTarget":
        assert targets is not None, "targets must be provided if text_target_mode is gtTarget"
    else:
        raise ValueError(f"Invalid text target mode: {text_target_mode}")

    _, txt_dict = load_features(
        model,
        smiles,
        targets=targets,
        load_image_feature=False,
        load_text_feature=True,
        cached_image_features=image_features_dict,
        taskid=taskid,
        batch_size=batch_size,
        device=device,
    )
    return np.stack([txt_dict[s] for s in list(smiles)], axis=0)


def fit_tsne(X: np.ndarray, seed: int = 0, perplexity: int = 30) -> np.ndarray:
    """2D t-SNE embedding for numpy arrays."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")
    return tsne.fit_transform(X)


def stratified_sampling(df: pd.DataFrame, num_samples: int, descriptors: List[str], random_seed: int = 0) -> pd.DataFrame:
    """
    Robust stratified sampling (avoids hard failures when bins are empty).

    - For each descriptor, sample evenly from percentile bins.
    - If a bin is empty/insufficient, backfill from remaining pool.
    """
    rng = np.random.RandomState(random_seed)
    bins = [(0, 20), (20, 40), (40, 60), (60, 99.5), (99.5, 100)]
    per_bin = num_samples // (len(descriptors) * len(bins))
    if per_bin <= 0:
        raise ValueError("num_samples too small for requested descriptors/bins.")

    sampled_indices = set()
    picked_frames = []

    for d in descriptors:
        if d not in df.columns:
            raise ValueError(f"Descriptor column missing in dataframe: {d}")

        for p0, p1 in bins:
            lo = np.percentile(df[d].to_numpy(), p0)
            hi = np.percentile(df[d].to_numpy(), p1)

            cond = (df[d] >= lo) & (df[d] <= hi)
            available = df[cond].loc[~df[cond].index.isin(sampled_indices)]

            if len(available) < per_bin:
                remaining = df.loc[~df.index.isin(sampled_indices)]
                if len(remaining) < per_bin:
                    remaining = df
                take = remaining.sample(n=per_bin, random_state=int(rng.randint(0, 1_000_000_000)))
            else:
                take = available.sample(n=per_bin, random_state=int(rng.randint(0, 1_000_000_000)))

            sampled_indices.update(take.index)
            picked_frames.append(take)

    out = pd.concat(picked_frames, axis=0)
    if len(out) > num_samples:
        out = out.sample(n=num_samples, random_state=random_seed)
    return out


def get_descriptor_targets(smiles: np.ndarray, descriptors: List[str], get_descriptor_value_fn) -> Dict[str, np.ndarray]:
    """Compute descriptor targets via your RDKit helper for each SMILES."""
    targets: Dict[str, np.ndarray] = {}
    for d in descriptors:
        vals = [get_descriptor_value_fn(smi, [d])[d] for smi in smiles]
        targets[d] = np.asarray(vals)
    return targets


def visualize_feature_shifts_with_density(
    base_2d: np.ndarray,
    shifted_a: np.ndarray,
    shifted_b: np.ndarray,
    label_a: str,
    label_b: str,
    target_a: np.ndarray,
    target_b: np.ndarray,
    ax=None,
    n_clusters: int = 3,
    seed: int = 0,
):
    """
    Notebook-friendly visualization + cluster indices for Step3.

    Returns:
      groups = [groups_for_a, groups_for_b]
      where each is list[list[int]] with indices per cluster (KMeans on shifted embeddings).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.scatter(base_2d[:, 0], base_2d[:, 1], s=6, alpha=0.2, label="Base Distribution")
    ax.scatter(shifted_a[:, 0], shifted_a[:, 1], s=8, alpha=0.35, label=f"Distribution of {label_a}")
    ax.scatter(shifted_b[:, 0], shifted_b[:, 1], s=8, alpha=0.35, label=f"Distribution of {label_b}")
    ax.legend(frameon=False)

    km_a = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(shifted_a)
    km_b = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(shifted_b)

    groups_a = [np.where(km_a.labels_ == k)[0].tolist() for k in range(n_clusters)]
    groups_b = [np.where(km_b.labels_ == k)[0].tolist() for k in range(n_clusters)]
    return [groups_a, groups_b]


from matplotlib.colors import Normalize

def reduce_and_plot(
    reducer,
    image_features: np.ndarray,
    text_features: np.ndarray,
    targets: np.ndarray,
    fusion: str = "concatenated",
    combined_alpha: float = 0.5,
    traced=None,
    title_prefix: str = "",
    save: bool = False,
    out_dir: str = "./tsne_plots",
    dpi: int = 200,
    point_size: float = 2.0,
    alpha: float = 0.5,
):
    """
    Reduce fused (image,text) features to 2D and plot colored by targets.

    Notes
    - t-SNE does NOT support batching fit_transform. This function always runs a single fit_transform.
    - If you need large-scale, prefer UMAP.
    """
    assert fusion in ["concatenated", "combined"], "fusion must be 'concatenated' or 'combined'"
    assert image_features.shape[0] == text_features.shape[0] == len(targets), "N mismatch among inputs"

    if fusion == "combined":
        fused = (1.0 - combined_alpha) * image_features + combined_alpha * text_features
        title_fusion = f"combined(alpha={combined_alpha})"
    else:
        fused = np.concatenate([image_features, text_features], axis=1)
        title_fusion = "concatenated"

    # 2D reduction (single shot)
    reduced_2d = reducer.fit_transform(fused)

    # robust color normalization
    vmin = float(np.percentile(targets, 1.0))
    vmax = float(np.percentile(targets, 99.0))
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    sc = ax.scatter(
        reduced_2d[:, 0],
        reduced_2d[:, 1],
        c=targets,
        cmap="viridis",
        s=point_size,
        alpha=alpha,
        norm=norm,
    )

    if traced is not None:
        ax.scatter(
            reduced_2d[traced, 0],
            reduced_2d[traced, 1],
            c=np.asarray(targets)[traced],
            cmap="viridis",
            s=60,
            edgecolors="red",
            marker="*",
            alpha=0.9,
            norm=norm,
        )

    cbar = fig.colorbar(sc, ax=ax)
    ax.set_title(f"{title_prefix} | {title_fusion}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    if save:
        import os
        os.makedirs(out_dir, exist_ok=True)
        safe_name = title_prefix.replace(" ", "_").replace("/", "_")
        fig.savefig(f"{out_dir}/{safe_name}_TSNE.png", dpi=300)
        fig.savefig(f"{out_dir}/{safe_name}_TSNE.svg")

    plt.show()
    return reduced_2d