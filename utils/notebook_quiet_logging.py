"""Optional quiet logging for notebook demos and subprocess CLIs."""

from __future__ import annotations

import logging
import os


def notebook_quiet_logging_enabled() -> bool:
    flag = os.environ.get("CHEMVL_NOTEBOOK_QUIET", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def apply_notebook_quiet_logging() -> None:
    """Silence verbose INFO logs from OrdinalCLIP / Lightning in the current process."""
    if not notebook_quiet_logging_enabled():
        return
    # Blocks INFO and below for all loggers in this process (demo notebooks only).
    logging.disable(logging.INFO)
    level = logging.WARNING
    for name in (
        "ordinalclip",
        "pytorch_lightning",
        "lightning",
        "torch",
        "PIL",
        "matplotlib",
    ):
        logging.getLogger(name).setLevel(level)


def notebook_quiet_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for subprocess calls from demo notebooks (``run_cmd``)."""
    env = os.environ.copy()
    env["CHEMVL_NOTEBOOK_QUIET"] = "1"
    if extra:
        env.update(extra)
    return env
