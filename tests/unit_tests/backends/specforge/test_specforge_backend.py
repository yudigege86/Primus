###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the SpecForge backend (CPU only, no SpecForge install required).

Coverage:
    1. Adapter registration and lazy load via BackendRegistry
    2. Experiment YAML -> `specforge train` argv
    3. SpecForge working-directory resolution
    4. The pretrain hook emits env.RUN_MODE=single so Primus skips torchrun
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.specforge.argument_builder import (
    build_specforge_argv,
    flatten_overrides,
    resolve_specforge_root,
)
from primus.backends.specforge.specforge_adapter import SpecForgeAdapter
from primus.backends.specforge.specforge_pretrain_trainer import (
    SpecForgePretrainTrainer,
)
from primus.core.backend.backend_registry import BackendRegistry
from primus.core.launcher.parser import PrimusParser

PRIMUS_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_CONFIG = PRIMUS_ROOT / "examples" / "specforge" / "configs" / "qwen3.5-4b-dflash-offline.yaml"
PREPARE_HOOK = PRIMUS_ROOT / "runner/helpers/hooks/train/pretrain/specforge/prepare.py"


@pytest.fixture
def specforge_checkout(tmp_path):
    """A directory that resolve_specforge_root() recognizes as a SpecForge tree."""
    root = tmp_path / "SpecForge"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'specforge'\n")
    return root


@pytest.fixture
def hidden_states(tmp_path):
    """A non-empty directory standing in for captured hidden states."""
    path = tmp_path / "hidden_states"
    path.mkdir()
    (path / "shard-0000.pt").write_bytes(b"")
    return path


@pytest.fixture
def experiment_env(monkeypatch, specforge_checkout, hidden_states, tmp_path):
    """Environment the example experiment YAML interpolates."""
    env = {
        "SPECFORGE_ROOT": str(specforge_checkout),
        "SPECFORGE_CONFIG": str(specforge_checkout / "configs" / "qwen3.5-4b-dflash.yaml"),
        "HIDDEN_STATES_PATH": str(hidden_states),
        "OUTPUT_DIR": str(tmp_path / "out"),
        "MAX_STEPS": "20",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


def load_pre_trainer_params(config_path):
    """Parse an experiment YAML into the params the adapter receives."""
    from primus.core.config.primus_config import get_module_config, load_primus_config

    cfg = load_primus_config(Path(config_path))
    module = get_module_config(cfg, "pre_trainer")
    assert module is not None, "pre_trainer module missing from config"
    return module


class TestRegistration:
    def test_adapter_registered_on_import(self):
        import primus.backends.specforge  # noqa: F401

        assert BackendRegistry.has_adapter("specforge")
        assert BackendRegistry._adapters["specforge"] is SpecForgeAdapter

    def test_get_adapter_lazy_loads_backend(self):
        original = BackendRegistry._adapters.copy()
        try:
            BackendRegistry._adapters.pop("specforge", None)
            adapter = BackendRegistry.get_adapter("specforge")
        finally:
            BackendRegistry._adapters = original

        assert isinstance(adapter, SpecForgeAdapter)
        assert adapter.framework == "specforge"

    def test_load_trainer_class(self):
        adapter = SpecForgeAdapter()
        assert adapter.load_trainer_class("pretrain") is SpecForgePretrainTrainer
        with pytest.raises(ValueError, match="Unsupported stage"):
            adapter.load_trainer_class("sft")

    def test_setup_backend_path_tolerates_missing_checkout(self, monkeypatch):
        """SpecForge ships as a wheel, so a missing third_party/ dir is not fatal."""
        monkeypatch.delenv("SPECFORGE_ROOT", raising=False)
        monkeypatch.delenv("BACKEND_PATH", raising=False)
        adapter = SpecForgeAdapter()
        assert adapter.setup_backend_path() == ""

    def test_setup_backend_path_uses_specforge_root(self, monkeypatch, specforge_checkout):
        monkeypatch.setenv("SPECFORGE_ROOT", str(specforge_checkout))
        adapter = SpecForgeAdapter()
        resolved = adapter.setup_backend_path()
        assert Path(resolved) == specforge_checkout.resolve()


class TestOverrideFlattening:
    def test_nested_and_dotted_keys_both_flatten(self):
        flat = flatten_overrides({"training": {"max_steps": 20}, "data.path": "/x"})
        assert flat == {"training.max_steps": "20", "data.path": "/x"}

    def test_booleans_use_lowercase_hydra_form(self):
        assert flatten_overrides({"model": {"use_liger_kernel": False}}) == {
            "model.use_liger_kernel": "false"
        }

    def test_namespace_input(self):
        ns = SimpleNamespace(training=SimpleNamespace(max_steps=5))
        assert flatten_overrides(ns) == {"training.max_steps": "5"}


class TestArgvBuilder:
    def test_config_is_required(self):
        with pytest.raises(ValueError, match="specforge_config"):
            build_specforge_argv(SimpleNamespace())

    def test_argv_shape(self):
        params = SimpleNamespace(
            specforge_config="/sf/configs/a.yaml",
            specforge_overrides={"training.max_steps": 20},
            output_dir="/out",
        )
        argv = build_specforge_argv(params)
        assert argv[:4] == ["specforge", "train", "--config", "/sf/configs/a.yaml"]
        assert "training.max_steps=20" in argv
        assert "output_dir=/out" in argv

    def test_explicit_output_dir_override_wins(self):
        params = SimpleNamespace(
            specforge_config="/sf/a.yaml",
            specforge_overrides={"output_dir": "/explicit"},
            output_dir="/alias",
        )
        assert "output_dir=/explicit" in build_specforge_argv(params)
        assert "output_dir=/alias" not in build_specforge_argv(params)

    def test_custom_entrypoint(self):
        params = SimpleNamespace(specforge_config="/sf/a.yaml", specforge_entrypoint="specforge-wrapper")
        assert build_specforge_argv(params)[0] == "specforge-wrapper"


class TestWorkdirResolution:
    def test_explicit_param_wins(self, specforge_checkout, tmp_path, monkeypatch):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("SPECFORGE_ROOT", str(other))
        params = SimpleNamespace(specforge_root=str(specforge_checkout))
        assert resolve_specforge_root(params) == specforge_checkout

    def test_env_fallback(self, specforge_checkout):
        params = SimpleNamespace()
        env = {"SPECFORGE_ROOT": str(specforge_checkout)}
        assert resolve_specforge_root(params, env=env) == specforge_checkout

    def test_inferred_from_config_ancestor(self, specforge_checkout):
        config = specforge_checkout / "configs" / "dflash.yaml"
        config.write_text("{}")
        params = SimpleNamespace(specforge_config=str(config))
        assert resolve_specforge_root(params, env={}) == specforge_checkout

    def test_unresolvable_returns_none(self):
        params = SimpleNamespace(specforge_config="relative/path.yaml")
        assert resolve_specforge_root(params, env={}) is None


class TestExampleExperiment:
    def test_yaml_converts_to_specforge_argv(self, experiment_env):
        module = load_pre_trainer_params(EXAMPLE_CONFIG)
        assert module.framework == "specforge"

        adapter = SpecForgeAdapter()
        backend_args = adapter.convert_config(module.params)

        argv = build_specforge_argv(backend_args)
        assert argv[:3] == ["specforge", "train", "--config"]
        assert argv[3] == experiment_env["SPECFORGE_CONFIG"]
        assert "training.max_steps=20" in argv
        assert "model.use_liger_kernel=false" in argv
        assert f"data.hidden_states_path={experiment_env['HIDDEN_STATES_PATH']}" in argv
        assert f"output_dir={experiment_env['OUTPUT_DIR']}" in argv
        assert backend_args.specforge_root == experiment_env["SPECFORGE_ROOT"]

    def test_trainer_rejects_missing_entrypoint(self, experiment_env):
        backend_args = SimpleNamespace(
            specforge_config=experiment_env["SPECFORGE_CONFIG"],
            specforge_entrypoint="specforge-does-not-exist",
            specforge_root=experiment_env["SPECFORGE_ROOT"],
            specforge_overrides={},
        )
        trainer = SpecForgePretrainTrainer(backend_args=backend_args)
        with pytest.raises(RuntimeError, match="not found on PATH"):
            trainer.init()


class TestPrepareHook:
    """The hook must tell primus-cli to skip torchrun; SpecForge self-launches."""

    def run_hook(self, env_overrides, extra_args=()):
        env = dict(os.environ)
        env.update(env_overrides)
        env["PYTHONPATH"] = os.pathsep.join([str(PRIMUS_ROOT), env.get("PYTHONPATH", "")])
        return subprocess.run(
            [
                sys.executable,
                str(PREPARE_HOOK),
                "--config",
                str(EXAMPLE_CONFIG),
                "--data_path",
                str(PRIMUS_ROOT / "data"),
                "--primus_path",
                str(PRIMUS_ROOT),
                *extra_args,
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(PRIMUS_ROOT),
        )

    def test_emits_single_run_mode(self, experiment_env):
        result = self.run_hook(experiment_env)
        assert result.returncode == 0, result.stderr
        assert "env.RUN_MODE=single" in result.stdout
        assert "env.GPUS_PER_NODE=1" in result.stdout
        assert f"env.SPECFORGE_ROOT={experiment_env['SPECFORGE_ROOT']}" in result.stdout

    def test_fails_fast_on_empty_hidden_states(self, experiment_env, tmp_path):
        empty = tmp_path / "empty_hidden_states"
        empty.mkdir()
        result = self.run_hook({**experiment_env, "HIDDEN_STATES_PATH": str(empty)})
        assert result.returncode != 0
        assert "empty" in result.stderr.lower()

    def test_fails_fast_on_missing_hidden_states(self, experiment_env, tmp_path):
        result = self.run_hook({**experiment_env, "HIDDEN_STATES_PATH": str(tmp_path / "nope")})
        assert result.returncode != 0
        assert "not a directory" in result.stderr.lower()


class TestPrimusParserAcceptsSpecForge:
    def test_framework_dispatch_name(self, experiment_env):
        """prepare_experiment.sh routes on this exact string."""
        parser = PrimusParser()
        cfg = parser.parse(SimpleNamespace(config=str(EXAMPLE_CONFIG)))
        assert cfg.get_module_config("pre_trainer").framework == "specforge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
