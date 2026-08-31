###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
SpecForgePretrainTrainer: Primus wrapper for SpecForge draft-model training.

SpecForge spawns and manages its own distributed workers, so this trainer does
not run a training loop in-process. It replaces the Primus process with the
SpecForge CLI via ``os.execvp``, which keeps a single process tree and lets
SpecForge's exit code propagate to the scheduler unchanged.

Because ``execvp`` never returns, ``cleanup()`` only runs on the error path.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Optional

from primus.backends.specforge.argument_builder import build_specforge_argv
from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0, warning_rank_0


class SpecForgePretrainTrainer(BaseTrainer):
    """Trainer that hands off to the SpecForge CLI."""

    def __init__(self, backend_args: Any):
        super().__init__(backend_args=backend_args)
        self.argv: Optional[list[str]] = None
        self.workdir: Optional[str] = None
        log_rank_0("Initialized SpecForgePretrainTrainer")

    def setup(self):
        log_rank_0("SpecForgePretrainTrainer.setup()")

    def init(self):
        """Build the SpecForge argv and validate the entrypoint is reachable."""

        self.argv = build_specforge_argv(self.backend_args)
        self.workdir = getattr(self.backend_args, "specforge_root", None)

        if shutil.which(self.argv[0]) is None:
            raise RuntimeError(
                f"[Primus:specforge] Entrypoint '{self.argv[0]}' not found on PATH. "
                "Install SpecForge in the training image, or set "
                "'specforge_entrypoint' in the pre_trainer module config."
            )

        log_rank_0(f"SpecForge command: {' '.join(self.argv)}")
        if self.workdir:
            log_rank_0(f"SpecForge cwd: {self.workdir}")
        else:
            warning_rank_0(
                "[Primus:specforge] No SpecForge root resolved; relative paths inside the "
                "SpecForge config may not resolve. Set SPECFORGE_ROOT or 'specforge_root'."
            )

    def train(self):
        """Replace this process with the SpecForge CLI. Does not return."""

        if self.argv is None:
            raise RuntimeError("SpecForgePretrainTrainer.init() must be called before train().")

        if self.workdir:
            os.chdir(self.workdir)

        log_rank_0("Handing off to SpecForge (exec).")
        os.execvp(self.argv[0], self.argv)
