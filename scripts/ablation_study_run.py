#!/usr/bin/env python3
"""
Grid driver for ablation studies (same spirit as ``run_exps.py`` / legacy
``run_pretraining_ablation_study.py``).

How to add a **new ablation topic**
------------------------------------
Mirror ``configs/ablation_study/pretraining/``:

- ``base.json`` — shared defaults (split, training template, model defaults, …).
- ``shared/datasets/*.json`` — per-task keys (``dataset``, ``num_tasks``, ``class_names``, …).
- Optional fragments (backbones, checkpoints, strategy lines) merged via
  ``group_registry.json``.
- ``group_registry.json`` — maps **group id** → ``{ "config_chain": [ relative paths ] }``.
  Depiction / framing (single-file chains per group): see
  ``configs/ablation_study/depiction_image_ft/group_registry.json``.
  Descriptor-only (RDKit / CLIP-text MLP, no structure encoder): see
  ``configs/ablation_study/descriptor_only/group_registry.json`` and use
  ``--finetune-script scripts/experiments/ablation_descriptor_only_finetuning_uniform_training_hparams/descriptor_only_finetune.py``.

Each merged config must set ``basic.version`` to a **unique directory name** under
``log_dir_base`` (e.g. ``pretrain_ablation_graph_gin_scratch``). Use a **fresh
prefix per topic** (e.g. ``ft_strategy_...``) so different studies under the same
``--exp-name`` do not collide. Analysis scripts filter runs with ``--group-prefix``
matching that prefix.

Outputs go to ``{basic.log_dir_base}/{basic.version}/{dataset}/{timestamp}/``.
With ``--exp-name STR``, ``basic.log_dir_base`` becomes
``join(original_log_dir_base, STR)`` (single path segment; no ``..``).

**``--uniform-training-hparams``:** after merging chain + dataset JSON, replace
``cfg["training"]`` from the **first** file in the chain only; set ``runseed``;
copy ``model.dropout`` / ``model.reduction_ratio`` from that base if present.

**``--training-hparams-yaml``:** after ``--uniform-training-hparams`` (if set),
apply optional per-dataset overrides from YAML: ``defaults`` →
``datasets.<dataset>`` → ``datasets.<dataset>.<image|graph>`` (modality inferred
from ``group_id`` prefix). Unlisted datasets keep uniform base values. YAML may
override ``training.*`` and ``model.dropout`` / ``model.reduction_ratio`` only;
``training.runseed`` is always set by this runner.

**Default groups:** if you omit ``--group``, every key in the registry is used
(sorted). Pass ``--group`` one or more times to restrict.

Reproducibility logs
--------------------
On each process start, after resolving the experiment log root, the runner writes
``<log_dir_base>/_repro/`` with ``git_head.txt``, ``git_status.txt``, and an
appended ``invocations.txt``. Use ``--save-git-diff`` to also write
``git_diff.patch`` when the working tree is dirty.

Each executed finetune appends one JSON line to ``runs.jsonl`` (group, dataset,
runseed, config chain paths, exit code).

Runs are skipped when an existing timestamped directory already has
``result.json`` with ``best_valid_on_test`` for the same ``training.runseed``
(see ``--no-skip-existing``).

**``--no-save-ckpt``:** force ``basic.save_finetune_ckpt=False`` for every cell
(no ``train_best`` / ``valid_best`` checkpoints). ``result.json`` and history
CSVs are still written.

**``summary.csv``:** after each executed finetune (not on skip), rescan
``{log_dir_base}/{version}/{dataset}/`` and rewrite ``summary.csv`` with one
row per ``runseed`` (latest timestamp wins). Use ``--rebuild-summary`` to
backfill summaries for the full grid before training starts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Repo root: parent of ``scripts/``
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = str(_SCRIPT_DIR.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.argparser import load_config  # noqa: E402

from scripts.ablation_study_repro import append_run_jsonl, write_repro_bundle  # noqa: E402
from scripts.ablation_study_summary import refresh_dataset_summary  # noqa: E402
from scripts.training_hparams_yaml import (  # noqa: E402
    apply_training_hparams_yaml,
    load_training_hparams_yaml,
    warn_unknown_datasets,
)

DEFAULT_REGISTRY = "configs/ablation_study/pretraining/group_registry.json"
DEFAULT_TASK_DIR = "configs/ablation_study/shared/datasets"
DEFAULT_DATASETS: List[str] = ["bbbp", "bace", "clintox", "tox21"]
RUNSEED_START = 0
RUNSEED_END = 0
FINETUNE_SCRIPT = "finetune_moleculenet.py"

SKIP_TRIPLES: Set[Tuple[str, str, int]] = set()


def _output_has_complete_test_result(cfg: Dict[str, Any], repo_root: str) -> bool:
    log_base = cfg.get("basic", {}).get("log_dir_base")
    version = cfg.get("basic", {}).get("version")
    ds_name = cfg.get("dataset", {}).get("dataset")
    want_seed = cfg.get("training", {}).get("runseed")
    if log_base is None or version is None or ds_name is None or want_seed is None:
        return False
    if not os.path.isabs(log_base):
        log_base = os.path.normpath(os.path.join(repo_root, log_base))
    ds_dir = os.path.join(log_base, str(version), str(ds_name))
    if not os.path.isdir(ds_dir):
        return False
    for entry in os.listdir(ds_dir):
        run_dir = os.path.join(ds_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        rpath = os.path.join(run_dir, "result.json")
        cpath = os.path.join(run_dir, "config.json")
        if not os.path.isfile(rpath) or not os.path.isfile(cpath):
            continue
        try:
            with open(cpath, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("training", {}).get("runseed") != want_seed:
                continue
            with open(rpath, encoding="utf-8") as f:
                res = json.load(f)
            if isinstance(res, dict) and "best_valid_on_test" in res:
                return True
        except (json.JSONDecodeError, OSError, TypeError):
            continue
    return False


def should_skip(
    group_id: str,
    dataset: str,
    runseed: int,
    merged_cfg: Dict[str, Any],
    repo_root: str,
    *,
    skip_if_existing_result: bool = True,
) -> Optional[str]:
    if (group_id, dataset, runseed) in SKIP_TRIPLES:
        return "SKIP_TRIPLES"
    if skip_if_existing_result and _output_has_complete_test_result(merged_cfg, repo_root):
        return "existing result.json (best_valid_on_test)"
    return None


def repo_root() -> str:
    return _REPO_ROOT


def resolve_path(root: str, path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def load_registry(registry_path: str) -> Dict[str, Any]:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def validate_exp_name(name: str) -> str:
    s = name.strip()
    if not s:
        raise ValueError("exp_name must be non-empty (after stripping)")
    if ".." in s or "/" in s or "\\" in s:
        raise ValueError("exp_name must not contain '..' or path separators")
    if os.path.isabs(s):
        raise ValueError("exp_name must be a single relative label, not an absolute path")
    return s


def apply_exp_name(cfg: Dict[str, Any], exp_name: str) -> None:
    basic = cfg.setdefault("basic", {})
    orig = basic.get("log_dir_base", "results/finetuning")
    basic["log_dir_base"] = os.path.normpath(os.path.join(str(orig), exp_name))


def apply_uniform_training_from_base(
    cfg: Dict[str, Any],
    base_config_path: str,
    runseed: int,
    chain_paths: Optional[List[str]] = None,
) -> None:
    base_cfg = load_config([base_config_path])
    if "training" not in base_cfg:
        raise KeyError(f"base config missing 'training': {base_config_path}")
    overlay_training: Dict[str, Any] = {}
    if chain_paths:
        for path in chain_paths[1:]:
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                overlay_cfg = json.load(f)
            tr = overlay_cfg.get("training")
            if isinstance(tr, dict):
                overlay_training.update(tr)
    cfg["training"] = copy.deepcopy(base_cfg["training"])
    cfg["training"].update(overlay_training)
    cfg["training"]["runseed"] = runseed
    base_model = base_cfg.get("model") or {}
    for k in ("dropout", "reduction_ratio"):
        if k in base_model:
            cfg.setdefault("model", {})[k] = base_model[k]


def run_finetune(
    python_exe: str,
    finetune_script: str,
    config_path: str,
    cwd: str,
    capture_output: bool,
) -> subprocess.CompletedProcess:
    command = [python_exe, finetune_script, "--config", config_path]
    print(f"Executing: {' '.join(command)}")
    if capture_output:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
    return subprocess.run(command, cwd=cwd)


def print_process_result(result: subprocess.CompletedProcess) -> None:
    if result.returncode == 0:
        print("Finished successfully.")
    else:
        print(f"Failed with exit code {result.returncode}.")
    if result.stdout:
        print("Output:")
        print(result.stdout)
    if result.stderr:
        print("Stderr:")
        print(result.stderr)


def _chain_paths(registry: Dict[str, Any], group_id: str, root: str) -> List[str]:
    return [resolve_path(root, p) for p in registry[group_id]["config_chain"]]


def _build_cell_cfg(
    chain: List[str],
    task_json: str,
    runseed: int,
    group_id: str,
    dataset: str,
    exp_name: str,
    *,
    uniform_training: bool,
    training_hparams_yaml: Optional[Dict[str, Any]],
    no_save_ckpt: bool = False,
) -> Dict[str, Any]:
    cfg = load_config(chain + [task_json])
    cfg["training"]["runseed"] = runseed
    if uniform_training:
        apply_uniform_training_from_base(cfg, chain[0], runseed)
    if training_hparams_yaml is not None:
        apply_training_hparams_yaml(
            cfg,
            training_hparams_yaml,
            dataset=dataset,
            group_id=group_id,
        )
    cfg["training"]["runseed"] = runseed
    apply_exp_name(cfg, exp_name)
    if no_save_ckpt:
        cfg.setdefault("basic", {})["save_finetune_ckpt"] = False
    return cfg


def _repro_dir_for_cell(
    root: str,
    registry: Dict[str, Any],
    group_id: str,
    dataset: str,
    task_dir: str,
    runseed: int,
    exp_name: Optional[str],
    uniform_training: bool,
    training_hparams_yaml: Optional[Dict[str, Any]],
    no_save_ckpt: bool = False,
) -> Path:
    chain = _chain_paths(registry, group_id, root)
    task_json = os.path.join(task_dir, f"{dataset}.json")
    cfg = _build_cell_cfg(
        chain,
        task_json,
        runseed,
        group_id,
        dataset,
        exp_name or "default",
        uniform_training=uniform_training,
        training_hparams_yaml=training_hparams_yaml,
        no_save_ckpt=no_save_ckpt,
    )
    log_base = cfg.get("basic", {}).get("log_dir_base", "results/finetuning")
    if not os.path.isabs(log_base):
        log_base = os.path.normpath(os.path.join(root, log_base))
    return Path(log_base) / "_repro"


def _chain_fingerprint(chain: List[str]) -> str:
    h = hashlib.sha256()
    for p in chain:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY,
        help="Path to group_registry.json (relative to repo root or absolute).",
    )
    p.add_argument(
        "--task-dir",
        default=DEFAULT_TASK_DIR,
        help="Directory of per-dataset JSONs (relative to repo root or absolute).",
    )
    p.add_argument(
        "--group",
        action="append",
        dest="groups",
        default=None,
        help="Experimental group id (repeatable). Default: all keys in the registry.",
    )
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset ids (JSON stem under task dir).",
    )
    p.add_argument("--runseed-start", type=int, default=RUNSEED_START)
    p.add_argument("--runseed-end", type=int, default=RUNSEED_END)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List cells only; do not invoke finetuning.",
    )
    p.add_argument(
        "--capture-output",
        action="store_true",
        help="Capture stdout/stderr. Default: stream to terminal.",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--finetune-script",
        default=None,
        help=f"Path to finetune driver (default: {{repo}}/{FINETUNE_SCRIPT}).",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip when a completed result already exists for this runseed.",
    )
    p.add_argument(
        "--uniform-training-hparams",
        action="store_true",
        help="Replace training (and dropout/reduction_ratio from chain[0]) from base only.",
    )
    p.add_argument(
        "--exp-name",
        default="default",
        metavar="STR",
        help="Nest outputs under basic.log_dir_base/STR/ (single segment; no .. or /).",
    )
    p.add_argument(
        "--save-git-diff",
        action="store_true",
        help="If git working tree is dirty, write git_diff.patch under _repro/.",
    )
    p.add_argument(
        "--training-hparams-yaml",
        default=None,
        metavar="PATH",
        help="Optional YAML with per-dataset (and image/graph) training hparam overrides.",
    )
    p.add_argument(
        "--no-save-ckpt",
        action="store_true",
        help="Force basic.save_finetune_ckpt=False for every cell (no train_best/valid_best ckpts).",
    )
    p.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Before training, rescan each (group, dataset) dir and rewrite summary.csv.",
    )
    p.add_argument(
        "--no-write-summary",
        action="store_true",
        help="Do not update summary.csv after each finetune run.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    finetune_script = args.finetune_script or os.path.join(root, FINETUNE_SCRIPT)

    try:
        exp_name = validate_exp_name(args.exp_name)
    except ValueError as e:
        print(f"Invalid --exp-name: {e}", file=sys.stderr)
        return 1

    registry_path = resolve_path(root, args.registry)
    task_dir = resolve_path(root, args.task_dir)

    try:
        registry = load_registry(registry_path)
    except FileNotFoundError:
        print(f"Registry not found: {registry_path}", file=sys.stderr)
        return 1

    group_ids = sorted(registry.keys()) if args.groups is None else list(args.groups)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not group_ids:
        print("No groups to run (empty registry or empty --group list).", file=sys.stderr)
        return 1
    if not datasets:
        print("No datasets (empty --datasets).", file=sys.stderr)
        return 1

    for gid in group_ids:
        if gid not in registry:
            print(f"Unknown group {gid!r}. Registry keys: {sorted(registry)}", file=sys.stderr)
            return 1

    for dataset in datasets:
        task_json = os.path.join(task_dir, f"{dataset}.json")
        if not os.path.isfile(task_json):
            print(f"Missing task config: {task_json}", file=sys.stderr)
            return 1

    training_hparams_yaml: Optional[Dict[str, Any]] = None
    training_hparams_yaml_relpath: Optional[str] = None
    if args.training_hparams_yaml:
        yaml_path = resolve_path(root, args.training_hparams_yaml)
        if not os.path.isfile(yaml_path):
            print(f"Training hparams yaml not found: {yaml_path}", file=sys.stderr)
            return 1
        try:
            training_hparams_yaml = load_training_hparams_yaml(Path(yaml_path))
        except (OSError, ValueError) as e:
            print(f"Failed to load training hparams yaml: {e}", file=sys.stderr)
            return 1
        training_hparams_yaml_relpath = (
            os.path.relpath(yaml_path, root) if yaml_path.startswith(root) else yaml_path
        )
        warn_unknown_datasets(training_hparams_yaml, set(datasets))

    repro_dir = _repro_dir_for_cell(
        root,
        registry,
        group_ids[0],
        datasets[0],
        task_dir,
        args.runseed_start,
        exp_name,
        args.uniform_training_hparams,
        training_hparams_yaml,
        no_save_ckpt=args.no_save_ckpt,
    )
    write_repro_bundle(
        repro_dir,
        sys.argv,
        root,
        save_git_diff=args.save_git_diff,
    )

    if args.rebuild_summary:
        seen: Set[Tuple[str, str]] = set()
        for dataset in datasets:
            task_json = os.path.join(task_dir, f"{dataset}.json")
            for group_id in group_ids:
                key = (group_id, dataset)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    cfg = _build_cell_cfg(
                        _chain_paths(registry, group_id, root),
                        task_json,
                        args.runseed_start,
                        group_id,
                        dataset,
                        exp_name,
                        uniform_training=args.uniform_training_hparams,
                        training_hparams_yaml=training_hparams_yaml,
                        no_save_ckpt=args.no_save_ckpt,
                    )
                    summary_path = refresh_dataset_summary(cfg, root)
                    print(f"[rebuild-summary] {group_id} / {dataset} -> {summary_path}")
                except (KeyError, OSError, json.JSONDecodeError, ValueError) as e:
                    print(f"[rebuild-summary] failed {group_id} / {dataset}: {e}", file=sys.stderr)
                    return 1

    failures = 0

    for dataset in datasets:
        task_json = os.path.join(task_dir, f"{dataset}.json")

        for group_id in group_ids:
            chain = _chain_paths(registry, group_id, root)
            for p in chain:
                if not os.path.isfile(p):
                    print(f"[{group_id}] missing chain file: {p}", file=sys.stderr)
                    return 1

            for runseed in range(args.runseed_start, args.runseed_end + 1):
                try:
                    cfg = _build_cell_cfg(
                        chain,
                        task_json,
                        runseed,
                        group_id,
                        dataset,
                        exp_name,
                        uniform_training=args.uniform_training_hparams,
                        training_hparams_yaml=training_hparams_yaml,
                        no_save_ckpt=args.no_save_ckpt,
                    )
                except (KeyError, OSError, json.JSONDecodeError, ValueError) as e:
                    print(f"[{group_id}] build cell config failed: {e}", file=sys.stderr)
                    return 1

                skip_reason = should_skip(
                    group_id,
                    dataset,
                    runseed,
                    cfg,
                    root,
                    skip_if_existing_result=not args.no_skip_existing,
                )
                if skip_reason is not None:
                    print(f"Skip {group_id} / {dataset} / runseed={runseed} ({skip_reason})")
                    continue

                if args.dry_run:
                    tr = cfg.get("training", {})
                    model = cfg.get("model", {})
                    basic = cfg.get("basic", {})
                    print(
                        f"[dry-run] group={group_id} dataset={dataset} runseed={runseed} "
                        f"version={basic.get('version')} strategy={tr.get('finetune_strategy')} "
                        f"log_dir_base={basic.get('log_dir_base')} "
                        f"save_finetune_ckpt={basic.get('save_finetune_ckpt')} "
                        f"epochs={tr.get('epochs')} patience={tr.get('patience')} "
                        f"batch_size={tr.get('batch_size')} optimizer={tr.get('optimizer')} "
                        f"lr={tr.get('lr')} weight_decay={tr.get('weight_decay')} "
                        f"dropout={model.get('dropout')} reduction_ratio={model.get('reduction_ratio')}"
                    )
                    continue

                fd, tmp_path = tempfile.mkstemp(
                    prefix=f"ablation_{group_id}_{dataset}_{runseed}_", suffix=".json"
                )
                t0 = time.time()
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=4)
                    proc = run_finetune(
                        args.python,
                        finetune_script,
                        tmp_path,
                        cwd=root,
                        capture_output=args.capture_output,
                    )
                    if args.capture_output:
                        print_process_result(proc)
                    if proc.returncode != 0:
                        failures += 1
                    if not args.no_write_summary:
                        try:
                            summary_path = refresh_dataset_summary(
                                cfg,
                                root,
                                exit_code=proc.returncode,
                                after_ts=t0,
                            )
                            print(f"Updated summary: {summary_path}")
                        except (KeyError, OSError, ValueError) as e:
                            print(
                                f"Warning: failed to update summary for {group_id}/{dataset}: {e}",
                                file=sys.stderr,
                            )
                    append_run_jsonl(
                        repro_dir,
                        {
                            "group_id": group_id,
                            "dataset": dataset,
                            "runseed": runseed,
                            "config_chain": [os.path.relpath(p, root) if p.startswith(root) else p for p in chain],
                            "chain_fingerprint": _chain_fingerprint(chain),
                            "training_hparams_yaml": training_hparams_yaml_relpath,
                            "exit_code": proc.returncode,
                        },
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
