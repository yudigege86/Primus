###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
SpecForge pre-train preparation hook for primus-cli direct.

Invoked by ``runner/helpers/hooks/train/pretrain/prepare_experiment.sh``.

The important job here is the run mode. ``runner/primus-cli-direct.sh`` defaults
to ``torchrun --nproc_per_node ${GPUS_PER_NODE:-8}``, but SpecForge spawns its
own distributed workers. Nesting the two launchers oversubscribes every GPU, so
this hook emits ``env.RUN_MODE=single`` (and ``env.GPUS_PER_NODE=1``) to make
Primus launch a single plain-Python process that then execs SpecForge.

It also runs a light, GPU-free preflight on the offline hidden-states directory
so a bad path fails in seconds instead of after the container spins up.
"""

import argparse
import os
from pathlib import Path

from primus.backends.specforge.argument_builder import flatten_overrides
from primus.core.launcher.parser import load_primus_config
from runner.helpers.hooks.train.pretrain.utils import log_error_and_exit, log_info


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Primus SpecForge environment")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (kept for compatibility; not used by this hook)",
    )
    parser.add_argument(
        "--backend_path",
        type=str,
        default=None,
        help="Optional override for the SpecForge checkout; takes precedence over SPECFORGE_ROOT.",
    )
    return parser.parse_known_args()


def resolve_specforge_root(cli_path, pre_trainer_cfg):
    """Pick the SpecForge checkout: CLI > config > SPECFORGE_ROOT."""

    for source, value in (
        ("--backend_path", cli_path),
        ("pre_trainer.specforge_root", getattr(pre_trainer_cfg, "specforge_root", None)),
        ("SPECFORGE_ROOT", os.getenv("SPECFORGE_ROOT")),
    ):
        if value and Path(value).is_dir():
            log_info(f"SpecForge root from {source}: {value}")
            return Path(value).resolve()
        if value:
            log_info(f"SpecForge root from {source} does not exist, skipping: {value}")
    return None


def preflight_hidden_states(pre_trainer_cfg):
    """Fail fast when the offline hidden-states cache is missing or empty.

    Offline DFlash training reads pre-captured hidden states from disk. This is
    advisory: when no path is configured we assume an online/other recipe and
    let SpecForge validate its own config.
    """

    # Overrides may be written either nested (`data: {hidden_states_path: ...}`)
    # or with dotted Hydra keys; flatten_overrides normalizes both.
    overrides = flatten_overrides(getattr(pre_trainer_cfg, "specforge_overrides", None))
    hidden_states = overrides.get("data.hidden_states_path") or overrides.get("hidden_states_path")
    if not hidden_states:
        hidden_states = os.getenv("HIDDEN_STATES_PATH")

    if not hidden_states:
        log_info("No hidden-states path configured; skipping offline preflight.")
        return

    path = Path(str(hidden_states))
    if not path.is_dir():
        log_error_and_exit(f"Hidden-states path is not a directory: {path}")
    if not any(path.iterdir()):
        log_error_and_exit(f"Hidden-states path is empty: {path}")
    log_info(f"Hidden-states preflight OK: {path}")


def main():
    args, unknown = parse_args()

    exp_path = Path(args.config).resolve()
    log_info(f"PRIMUS_PATH: {Path(args.primus_path).resolve()}")
    log_info(f"DATA_PATH: {Path(args.data_path).resolve()}")
    log_info(f"EXP: {exp_path}")

    if not exp_path.is_file():
        log_error_and_exit(f"EXP file not found: {exp_path}")

    primus_config, _ = load_primus_config(args, unknown)

    try:
        pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    except Exception:
        log_error_and_exit("Missing required module config: pre_trainer")

    if not getattr(pre_trainer_cfg, "specforge_config", None):
        log_error_and_exit("Missing required field: pre_trainer.specforge_config")

    preflight_hidden_states(pre_trainer_cfg)

    specforge_root = resolve_specforge_root(args.backend_path, pre_trainer_cfg)
    if specforge_root is not None:
        print(f"env.SPECFORGE_ROOT={specforge_root}")
        print(f"extra.backend_path={specforge_root}")

    # SpecForge self-launches its workers; keep Primus to one plain-Python
    # process so the two launchers do not nest.
    log_info("Exposing run mode via env.RUN_MODE=single (SpecForge owns its launcher)")
    print("env.RUN_MODE=single")
    print("env.GPUS_PER_NODE=1")


if __name__ == "__main__":
    log_info("========== Prepare SpecForge Env (pre-train hook) ==========")
    main()
