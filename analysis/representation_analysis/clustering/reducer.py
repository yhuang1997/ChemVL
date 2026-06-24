from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.manifold import TSNE


def build_reducer(cfg: Dict[str, Any]):
    name = str((cfg or {}).get("name", "tsne")).lower()
    params = dict((cfg or {}).get("params", {}))
    if name == "tsne":
        defaults = {"n_components": 2, "perplexity": 30, "random_state": 0}
        defaults.update(params)
        return "tsne", TSNE(**defaults), defaults
    raise ValueError(f"Unsupported reducer: {name}")


def reduce_features(features: np.ndarray, reducer_cfg: Dict[str, Any]):
    reducer_name, reducer, reducer_params = build_reducer(reducer_cfg)
    points = reducer.fit_transform(features)
    return points, reducer_name, reducer_params

