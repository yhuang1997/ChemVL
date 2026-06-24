import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, open_dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from ordinalclip.runner.data import RegressionDataModule, MultiRegressionDataModule
from ordinalclip.runner.runner import Runner, MultiTaskRunner
from ordinalclip.utils.logging import get_logger, setup_file_handle_for_all_logger

logger = get_logger(__name__)


class _ClearCudaCacheBeforeValidation(pl.Callback):
    """Reduce fragmentation between train and val when VRAM is tight (Linux/Windows)."""

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(cfg: DictConfig):
    pl.seed_everything(cfg.runner_cfg.seed, True)
    output_dir = Path(cfg.runner_cfg.output_dir)
    setup_file_handle_for_all_logger(str(output_dir / "run.log"))

    save_mem = bool(cfg.get("pretrain_save_memory_on_validation", True))
    use_cache_clear = cfg.trainer_cfg.get("accelerator") == "gpu" and save_mem
    callbacks = load_callbacks(output_dir, use_cuda_cache_clear=use_cache_clear)
    logger.info(
        f"save_memory_on_validation={save_mem} "
        f"(val/test CPU offload + pre-val empty_cache: {save_mem})"
    )
    loggers = load_loggers(output_dir)

    deterministic = True
    logger.info(f"`deterministic` flag: {deterministic}")

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        deterministic=deterministic,
        **OmegaConf.to_container(cfg.trainer_cfg),
    )

    if cfg.trainer_cfg.fast_dev_run is True:
        from IPython.core.debugger import set_trace

        set_trace()

    runner = None
    regression_datamodule = build_datamodule(cfg)
    if cfg.multitask:
        runner = MultiTaskRunner(**OmegaConf.to_container(cfg.runner_cfg))
    else:
        runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))
    # Training
    if not cfg.test_only:
        logger.info("Start training.")
        trainer.fit(model=runner, datamodule=regression_datamodule)

        logger.info("End training.")

    # Testing
    ckpt_paths = list((output_dir / "ckpts").glob("*.ckpt"))
    if len(ckpt_paths) == 0:
        logger.info("zero shot")
        if runner is None:
            runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))
        trainer.test(model=runner, datamodule=regression_datamodule)
        logger.info(f"End zero shot.")

    for ckpt_path in ckpt_paths:
        logger.info(f"Start testing ckpt: {ckpt_path}.")

        # no need to load weights in runner wrapper
        # OmegaConf DictConfig doesn't have `mds()`; iterate its keys and disable all
        # `init_*_weights` so Runner.load_weights won't preload anything during testing.
        for k in list(cfg.runner_cfg.load_weights_cfg.keys()):
            cfg.runner_cfg.load_weights_cfg[k] = None
        cfg.runner_cfg.ckpt_path = str(ckpt_path)

        if runner is None:
            runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))

        runner = runner.load_from_checkpoint(str(ckpt_path), **OmegaConf.to_container(cfg.runner_cfg))
        trainer.test(model=runner, datamodule=regression_datamodule)

        logger.info(f"End testing ckpt: {ckpt_path}.")


def load_loggers(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "tb_logger").mkdir(exist_ok=True, parents=True)
    (output_dir / "csv_logger").mkdir(exist_ok=True, parents=True)
    loggers = []
    # tb_logger = pl_loggers.TensorBoardLogger(
    #     str(output_dir),
    #     name="tb_logger",
    # )
    loggers.append(
        pl_loggers.CSVLogger(
            str(output_dir),
            name="csv_logger",
        )
    )

    return loggers


def load_callbacks(output_dir, use_cuda_cache_clear: bool = False):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "ckpts").mkdir(exist_ok=True, parents=True)

    callbacks = []
    if use_cuda_cache_clear:
        callbacks.append(_ClearCudaCacheBeforeValidation())
    callbacks.append(
        ModelCheckpoint(
            monitor="val_mae_exp_metric",
            dirpath=str(output_dir / "ckpts"),
            filename="{epoch:02d}-{val_mae_exp_metric:.4f}",
            verbose=True,
            save_last=True,
            save_top_k=-1,
            mode="min",
            save_weights_only=True,
        )
    )
    return callbacks


def setup_output_dir_for_training(output_dir):
    output_dir = Path(output_dir)

    if output_dir.stem.startswith("version_"):
        output_dir = output_dir.parent
    output_dir = output_dir / f"version_{get_version(output_dir)}"

    return output_dir


def get_version(path: Path):
    versions = path.glob("version_*")
    return len(list(versions))


def build_datamodule(cfg):
    data_cfg = OmegaConf.to_container(cfg.data_cfg, resolve=True)
    representation = data_cfg.pop("representation", "image")
    if representation == "graph":
        try:
            from ordinalclip.runner.data_graph import (
                GraphMultiRegressionDataModule,
                GraphRegressionDataModule,
            )
        except ImportError as exc:
            raise ImportError("Graph data support requires ordinalclip.runner.data_graph") from exc

        datamodule_cls = GraphMultiRegressionDataModule if cfg.multitask else GraphRegressionDataModule
        data_cfg.pop("transforms_cfg", None)
    elif representation == "image":
        datamodule_cls = MultiRegressionDataModule if cfg.multitask else RegressionDataModule
    else:
        raise ValueError(f"Unknown data representation: {representation}")

    return datamodule_cls(**data_cfg)


def parse_cfg(args, instantialize_output_dir=True):
    cfg = OmegaConf.merge(*[OmegaConf.load(config_) for config_ in args.config])
    extra_cfg = OmegaConf.from_dotlist(args.cfg_options)
    cfg = OmegaConf.merge(cfg, extra_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.create())

    # Setup data root
    data_cfg = cfg.data_cfg
    representation = data_cfg.get("representation", "image")
    data_root = data_cfg.get("data_root", None)
    if data_root is None:
        raise ValueError("Please set data_cfg.data_root")

    data_root = Path(data_root)
    paths = data_cfg.get("paths", None)
    if paths is None:
        raise ValueError("Please set data_cfg.paths")
    resolved_paths = {}
    for name, rel_path in paths.items():
        if rel_path is None:
            continue
        resolved_paths[name] = str(data_root / rel_path)
    data_cfg.update(resolved_paths)
    data_cfg.pop("paths", None)

    if representation == "graph":
        metadata_file = data_cfg.get("graph_metadata_file", None)
        if metadata_file is None:
            raise ValueError("Graph-based training requires data_cfg.graph_metadata_file")
        data_cfg["graph_metadata_file"] = str(data_root / metadata_file)
    else:
        data_cfg.pop("graph_metadata_file", None)

    data_cfg.pop("data_root", None)

    # Setup output_dir
    output_dir = Path(cfg.runner_cfg.output_dir if args.output_dir is None else args.output_dir)
    if instantialize_output_dir:
        if not args.test_only:
            output_dir = setup_output_dir_for_training(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    seed = args.seed if args.seed is not None else cfg.runner_cfg.seed
    cli_cfg = OmegaConf.create(
        dict(
            config=args.config,
            test_only=args.test_only,
            runner_cfg=dict(seed=seed, output_dir=str(output_dir)),
            trainer_cfg=dict(fast_dev_run=args.debug),
        )
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Default on for tight VRAM; --fast-validation skips empty_cache + CPU val offload.
    save_mem = not getattr(args, "fast_validation", False)
    with open_dict(cfg):
        cfg.pretrain_save_memory_on_validation = bool(save_mem)
    if cfg.multitask:
        with open_dict(cfg.runner_cfg):
            cfg.runner_cfg.save_memory_on_validation = bool(save_mem)

    if instantialize_output_dir:
        OmegaConf.save(cfg, str(output_dir / "config.yaml"))
    return cfg


if __name__ == "__main__":
    ## --config
    ## ./ordinalclip/configs/default.yaml
    ## --config
    ## ./ordinalclip/configs/base_cfgs/data_cfg/datasets/mol-1M-7mds/local.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", type=str, default=[])
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--fast-validation",
        action="store_true",
        default=False,
        help="Faster validation/test on large VRAM: disable pre-val cuda empty_cache and keep multitask val/test "
        "tensors on GPU (no CPU offload).",
    )
    parser.add_argument(
        "--cfg_options",
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    cfg = parse_cfg(args, instantialize_output_dir=True)

    logger.info("Start.")
    main(cfg)
    logger.info("End.")
