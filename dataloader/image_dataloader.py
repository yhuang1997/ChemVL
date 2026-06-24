import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def depiction_processed_subdir(depiction: Optional[Dict[str, Any]] = None) -> str:
    """
    Subdirectory under ``{dataroot}/{dataset}/processed/`` for molecular PNGs.

    Rules: ``default`` preset uses bare canvas folder (``224``, etc.).
    Other presets use ``{canvas}_{preset}`` (e.g. ``224_layout_var``, ``224_zoom_50``).

    If ``depiction`` is missing/empty, returns ``224`` (legacy layout).
    """
    if not depiction:
        return "224"
    canvas = int(depiction.get("render_canvas_px", 224))
    preset = (depiction.get("render_preset") or "default").strip()
    if preset == "default":
        return str(canvas)
    return f"{canvas}_{preset}"


class ImageDataset(Dataset):
    def __init__(self, filenames, labels, index=None, img_transformer=None, normalize=None, ret_index=False,
                 args=None, smiles=None):
        '''
        :param names: image path, e.g. ["./data/1.png", "./data/2.png", ..., "./data/n.png"]
        :param labels: labels, e.g. single label: [[1], [0], [2]]; multi-labels: [[0, 1, 0], ..., [1,1,0]]
        :param img_transformer:
        :param normalize:
        :param args:
        '''
        self.args = args
        self.filenames = filenames
        self.labels = labels
        self.total = len(self.filenames)
        self.normalize = normalize
        self._image_transformer = img_transformer
        self.ret_index = ret_index
        if index is not None:
            self.index = index
        else:
            self.index = []
            for filename in filenames:
                self.index.append(os.path.splitext(os.path.split(filename)[1])[0])
        self.smiles = smiles

    def get_image(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')

        # show img and transformed_img
        # transformed_img = self._image_transformer(img).numpy()
        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(transformed_img.transpose(1, 2, 0))
        # plt.show()
        return self._image_transformer(img)

    def __getitem__(self, index):
        data = self.get_image(index)
        if self.normalize is not None:
            data = self.normalize(data)
        if self.ret_index:
            if self.smiles is not None:
                return data, self.labels[index], self.index[index], self.smiles[index]
            else:
                return data, self.labels[index], self.index[index]
        else:
            if self.smiles is not None:
                return data, self.labels[index], self.smiles[index]
            else:
                return data, self.labels[index]

    def __len__(self):
        return self.total


def load_filenames_and_labels_multitask(image_folder, txt_file, task_type="classification"):
    assert task_type in ["classification", "regression"]
    df = pd.read_csv(txt_file)
    index = df["index"].values.astype(int)
    labels = np.array(df.label.apply(lambda x: str(x).split(' ')).tolist())
    labels = labels.astype(int) if task_type == "classification" else labels.astype(float)
    names = [os.path.join(image_folder, str(item) + ".png") for item in index]
    assert len(index) == labels.shape[0] == len(names)
    return names, labels


def get_datasets(
    dataset,
    dataroot,
    data_type="raw",
    depiction: Optional[Dict[str, Any]] = None,
):
    assert data_type in ["raw", "processed"]

    if depiction:
        from utils.depiction_constants import VALID_RENDER_PRESETS

        pr = (depiction.get("render_preset") or "default").strip()
        if pr not in VALID_RENDER_PRESETS:
            raise ValueError(
                f"Unknown dataset.depiction.render_preset {pr!r}; "
                f"expected one of {VALID_RENDER_PRESETS}"
            )

    subdir = depiction_processed_subdir(depiction)
    image_folder = os.path.join(dataroot, dataset, data_type, subdir)
    txt_file = os.path.join(dataroot, "{}/{}/{}_processed_ac.csv".format(dataset, data_type, dataset))

    assert os.path.isdir(image_folder), "{} is not a directory.".format(image_folder)
    assert os.path.isfile(txt_file), "{} is not a file.".format(txt_file)

    return image_folder, txt_file


def Smiles2Img(smis, size=224, savePath=None):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
    '''
    from utils.pretrain_image_render import smiles_to_pretrain_pil

    try:
        img = smiles_to_pretrain_pil(smis, canvas_px=size)
        if savePath is not None:
            img.save(savePath)
        return img
    except Exception:
        return None

