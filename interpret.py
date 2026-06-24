#!/usr/bin/env python3
"""
Preset-driven interpretability showcase CLI.

Examples::

    python interpret.py list
    python interpret.py visual run --preset analysis/interpret/presets/case_moleculenet_curated.yaml
    python interpret.py knowledge run --preset analysis/interpret/presets/knowledge_cases.yaml --case Cebaracetam
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from analysis.interpret.presets_loader import PRESETS_DIR, list_presets, load_preset, resolve_data_path
from analysis.interpret.runners.knowledge_single import run_knowledge_case
from analysis.interpret.runners.visual_case import run_case_panel
from analysis.interpret.runners.visual_testset import run_testset_gallery


def _default_output_dir(preset: Dict[str, Any]) -> Path:
    return Path("results") / "interpret" / str(preset["id"])


def _cmd_list(_args: argparse.Namespace) -> None:
    rows = list_presets()
    if not rows:
        print(f"No presets found under {PRESETS_DIR}")
        return
    print(f"{'id':<32} {'mode':<22} preset")
    print("-" * 90)
    for row in rows:
        print(f"{row['id']:<32} {row['mode']:<22} {row['file']}")


def _cmd_visual_run(args: argparse.Namespace) -> None:
    preset = load_preset(args.preset)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(preset)
    mode = str(preset.get("mode", ""))
    device = args.device

    if mode == "case_panel":
        paths = run_case_panel(preset, output_dir, case_filter=args.case, device=device)
    elif mode == "testset_gallery":
        paths = run_testset_gallery(
            preset,
            output_dir,
            max_molecules=args.max_molecules,
            seed=args.seed,
            device=device,
        )
    else:
        raise SystemExit(f"Unsupported visual preset mode: {mode!r}")

    print(f"Wrote {len(paths)} figure(s) to {output_dir.resolve()}")


def _cmd_knowledge_run(args: argparse.Namespace) -> None:
    if not args.preset and not (args.smiles and args.ckpt and args.dataset):
        raise SystemExit(
            "knowledge run requires --preset (with optional --case) "
            "or --smiles + --ckpt + --dataset bbbp|bace"
        )
    if args.preset:
        preset = load_preset(args.preset)
    else:
        preset = {
            "id": "knowledge_direct",
            "top_k": 15,
            "task_id": args.task_id,
            "batch_size": 4,
            "num_workers": 2,
            "figsize_cm": (18.0, 6.0),
            "dpi": 150,
            "formats": ["png"],
            "checkpoint_root": "{CHEMVL_DATA_ROOT}/checkpoints/finetuning/presets/knowledge_prompt_tuning",
        }

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(preset)
    ckpt = resolve_data_path(args.ckpt) if args.ckpt else None
    paths = run_knowledge_case(
        preset,
        output_dir,
        case_filter=args.case,
        smiles=args.smiles,
        ckpt_path=ckpt,
        dataset=args.dataset,
        device=args.device,
    )
    print(f"Wrote {len(paths)} figure(s) to {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ChemVL interpretability showcase (preset-driven visual + knowledge attention)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List available interpret presets")
    list_parser.set_defaults(func=_cmd_list)

    visual = sub.add_parser("visual", help="Grad-CAM visual interpretability")
    visual_sub = visual.add_subparsers(dest="visual_command", required=True)
    visual_run = visual_sub.add_parser("run", help="Run a visual preset")
    visual_run.add_argument("--preset", type=Path, required=True, help="Path to preset YAML")
    visual_run.add_argument("--output-dir", type=Path, default=None)
    visual_run.add_argument("--case", default=None, help="Run one case by name or id")
    visual_run.add_argument("--max-molecules", type=int, default=None)
    visual_run.add_argument("--seed", type=int, default=None)
    visual_run.add_argument("--device", default=None)
    visual_run.set_defaults(func=_cmd_visual_run)

    knowledge = sub.add_parser("knowledge", help="Knowledge-attention bar plots")
    knowledge_sub = knowledge.add_subparsers(dest="knowledge_command", required=True)
    knowledge_run = knowledge_sub.add_parser("run", help="Run knowledge-attention inference")
    knowledge_run.add_argument("--preset", type=Path, default=None)
    knowledge_run.add_argument("--case", default=None)
    knowledge_run.add_argument("--smiles", default=None)
    knowledge_run.add_argument("--ckpt", default=None, help="Finetune checkpoint (.pth)")
    knowledge_run.add_argument("--dataset", choices=("bbbp", "bace"), default=None)
    knowledge_run.add_argument("--task-id", type=int, default=0)
    knowledge_run.add_argument("--output-dir", type=Path, default=None)
    knowledge_run.add_argument("--device", default=None)
    knowledge_run.set_defaults(func=_cmd_knowledge_run)

    return parser


def main() -> None:
    from utils.notebook_quiet_logging import apply_notebook_quiet_logging

    apply_notebook_quiet_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
