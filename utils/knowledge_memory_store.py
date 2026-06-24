"""Sharded on-disk storage for prior ``knowledge_memory`` (smiles -> CPU tensor).

Monolithic ``.pkl`` files remain readable for legacy caches. New saves with more than
``KM_SHARD_THRESHOLD`` entries use ``{stem}.km/`` shard directories to avoid OOM during
``pickle.dump`` on large datasets (e.g. HIV ~41k).
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import warnings
from typing import Any, Dict, MutableMapping

KM_FORMAT = "chemvl_km_sharded_v1"
KM_DIR_SUFFIX = ".km"
KM_SHARD_THRESHOLD = 2000
KM_KEYS_PER_SHARD = 512
KM_VALIDATE_PKL_MAX_BYTES = 2 * 1024**3


def km_pkl_path(path: str) -> str:
    return path if path.endswith(".pkl") else f"{path}.pkl"


def km_shard_dir(pkl_path: str) -> str:
    p = km_pkl_path(pkl_path)
    return p[: -len(".pkl")] + KM_DIR_SUFFIX


def _is_valid_monolithic_pkl(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    size = os.path.getsize(path)
    if size <= 0:
        return False
    if size > KM_VALIDATE_PKL_MAX_BYTES:
        return True
    try:
        with open(path, "rb") as f:
            pickle.load(f)
        return True
    except (EOFError, pickle.UnpicklingError, OSError):
        return False


def _load_monolithic_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in knowledge memory cache {path}")
    return data


def _load_sharded_km(shard_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(shard_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing knowledge memory manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != KM_FORMAT:
        raise ValueError(f"Unsupported knowledge memory format: {manifest!r}")
    out: Dict[str, Any] = {}
    for name in manifest.get("shards") or []:
        shard_path = os.path.join(shard_dir, name)
        with open(shard_path, "rb") as f:
            chunk = pickle.load(f)
        if not isinstance(chunk, dict):
            raise ValueError(f"Shard {shard_path} is not a dict")
        out.update(chunk)
    return out


def load_knowledge_memory_store(pkl_path: str) -> Dict[str, Any]:
    """Load monolithic ``.pkl`` or sharded ``.km/`` cache; return empty dict if missing."""
    pkl_path = km_pkl_path(pkl_path)
    shard_dir = km_shard_dir(pkl_path)

    if os.path.isdir(shard_dir):
        try:
            data = _load_sharded_km(shard_dir)
            print(f"Load knowledge memory from sharded cache {shard_dir} ({len(data)} keys).")
            return data
        except (OSError, json.JSONDecodeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            warnings.warn(
                f"Sharded knowledge memory at {shard_dir} unreadable ({exc}); rebuilding.",
                stacklevel=2,
            )

    if os.path.isfile(pkl_path):
        try:
            data = _load_monolithic_pkl(pkl_path)
            print(f"Load knowledge memory from {pkl_path}.")
            return data
        except (EOFError, pickle.UnpicklingError, OSError, ValueError) as exc:
            warnings.warn(
                f"Monolithic knowledge memory at {pkl_path} unreadable ({exc}); rebuilding.",
                stacklevel=2,
            )

    return {}


def _rotate_corrupt_monolithic(pkl_path: str) -> None:
    backup = pkl_path + ".corrupt"
    try:
        if not os.path.exists(backup):
            os.replace(pkl_path, backup)
            warnings.warn(f"Moved corrupt knowledge memory to {backup}.", stacklevel=2)
        else:
            os.remove(pkl_path)
    except OSError as exc:
        warnings.warn(f"Could not rotate corrupt cache {pkl_path}: {exc}", stacklevel=2)
        try:
            os.remove(pkl_path)
        except OSError:
            pass


def _save_monolithic_pkl(pkl_path: str, memory: MutableMapping[str, Any]) -> None:
    tmp = f"{pkl_path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(dict(memory), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, pkl_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _save_sharded_km(shard_dir: str, memory: MutableMapping[str, Any]) -> None:
    parent = os.path.dirname(shard_dir)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    tmp_dir = f"{shard_dir}.tmp.{os.getpid()}"
    if os.path.exists(tmp_dir):
        import shutil

        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=False)

    items = list(memory.items())
    shard_names: list[str] = []
    try:
        for i in range(0, len(items), KM_KEYS_PER_SHARD):
            chunk = dict(items[i : i + KM_KEYS_PER_SHARD])
            name = f"shard_{i // KM_KEYS_PER_SHARD:04d}.pkl"
            shard_path = os.path.join(tmp_dir, name)
            with open(shard_path, "wb") as f:
                pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
            shard_names.append(name)
            del chunk
            gc.collect()

        manifest = {
            "format": KM_FORMAT,
            "n_keys": len(items),
            "keys_per_shard": KM_KEYS_PER_SHARD,
            "shards": shard_names,
        }
        with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        if os.path.isdir(shard_dir):
            import shutil

            shutil.rmtree(shard_dir)
        os.replace(tmp_dir, shard_dir)
    finally:
        if os.path.exists(tmp_dir):
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


def save_knowledge_memory_store(pkl_path: str, memory: MutableMapping[str, Any]) -> None:
    """Persist *memory* as sharded ``.km/`` (large) or monolithic ``.pkl`` (small)."""
    if not memory:
        warnings.warn("Knowledge memory empty; skip save.", stacklevel=2)
        return

    pkl_path = km_pkl_path(pkl_path)
    shard_dir = km_shard_dir(pkl_path)
    n_keys = len(memory)

    if os.path.isfile(pkl_path) and _is_valid_monolithic_pkl(pkl_path):
        print(f"Knowledge memory cache already valid at {pkl_path}; skip save.")
        return
    if os.path.isdir(shard_dir):
        try:
            existing = _load_sharded_km(shard_dir)
            if len(existing) >= n_keys:
                print(f"Sharded knowledge memory already valid at {shard_dir}; skip save.")
                return
        except (OSError, json.JSONDecodeError, ValueError, EOFError, pickle.UnpicklingError):
            import shutil

            shutil.rmtree(shard_dir, ignore_errors=True)

    if os.path.isfile(pkl_path) and not _is_valid_monolithic_pkl(pkl_path):
        _rotate_corrupt_monolithic(pkl_path)

    if n_keys > KM_SHARD_THRESHOLD:
        _save_sharded_km(shard_dir, memory)
        if os.path.isfile(pkl_path):
            try:
                os.remove(pkl_path)
            except OSError:
                pass
        print(f"Saved sharded knowledge memory ({n_keys} keys) to {shard_dir}.")
    else:
        _save_monolithic_pkl(pkl_path, memory)
        print(f"Saved knowledge memory ({n_keys} keys) to {pkl_path}.")
