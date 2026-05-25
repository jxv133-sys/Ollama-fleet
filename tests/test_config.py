"""Unit tests for ollama_fleet/config.py.

Tests cover:
- Default values for all settings models
- Valid TOML loading
- Field constraint enforcement (out-of-range values exit non-zero)
- Missing env-var file exits non-zero
- Missing fallback file returns defaults
- load_settings() env-var and fallback behaviour
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from ollama_fleet.config import (
    FleetSettings,
    MemoryConfig,
    OllamaConfig,
    SchedulerConfig,
    ToolsConfig,
    UIConfig,
    WorkspaceConfig,
    load_settings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_toml(tmp_path: Path, content: str) -> Path:
    """Write *content* to a temp TOML file and return its path."""
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_ollama_defaults(self) -> None:
        cfg = OllamaConfig()
        assert cfg.base_url == "http://192.168.50.142:7869/v1"
        assert cfg.planner_model == "hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M"
        assert cfg.coder_model == "hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M"
        assert cfg.summarizer_model == "hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M"
        assert cfg.critic_model == "hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M"
        assert cfg.tester_model == "hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M"
        assert cfg.timeout == 600.0

    def test_scheduler_defaults(self) -> None:
        cfg = SchedulerConfig()
        assert cfg.retry_limit == 3
        assert cfg.max_concurrent_tasks == 4
        assert cfg.poll_interval == 5.0
        assert cfg.max_critique_revision_loops == 3
        assert cfg.stall_timeout == 600.0

    def test_memory_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.max_context_tokens == 8192
        assert cfg.episodic_window == 5

    def test_workspace_defaults(self) -> None:
        cfg = WorkspaceConfig()
        assert cfg.base_path == "./workspaces"

    def test_ui_defaults(self) -> None:
        cfg = UIConfig()
        assert cfg.refresh_rate == 1.0

    def test_tools_defaults(self) -> None:
        cfg = ToolsConfig()
        assert cfg.command_timeout == 60
        assert cfg.git_enabled is True

    def test_fleet_settings_defaults(self) -> None:
        settings = FleetSettings()
        assert isinstance(settings.ollama, OllamaConfig)
        assert isinstance(settings.scheduler, SchedulerConfig)
        assert isinstance(settings.memory, MemoryConfig)
        assert isinstance(settings.workspace, WorkspaceConfig)
        assert isinstance(settings.ui, UIConfig)
        assert isinstance(settings.tools, ToolsConfig)


# ---------------------------------------------------------------------------
# Valid TOML loading
# ---------------------------------------------------------------------------


class TestValidToml:
    def test_load_empty_toml_uses_defaults(self, tmp_path: Path) -> None:
        p = write_toml(tmp_path, "")
        settings = FleetSettings.from_toml(str(p))
        assert settings.ollama.timeout == 600.0

    def test_load_partial_section(self, tmp_path: Path) -> None:
        p = write_toml(
            tmp_path,
            """
            [ollama]
            base_url = "http://192.168.1.10:11434"
            timeout = 900.0
            """,
        )
        settings = FleetSettings.from_toml(str(p))
        assert settings.ollama.base_url == "http://192.168.1.10:11434"
        assert settings.ollama.timeout == 900.0
        # Other sections keep defaults
        assert settings.scheduler.retry_limit == 3

    def test_load_all_sections(self, tmp_path: Path) -> None:
        p = write_toml(
            tmp_path,
            """
            [ollama]
            base_url = "http://localhost:11434"
            planner_model = "mistral"
            coder_model = "codellama"
            summarizer_model = "llama3"
            timeout = 1200.0

            [scheduler]
            retry_limit = 5
            max_concurrent_tasks = 8
            poll_interval = 2.0
            max_critique_revision_loops = 5
            stall_timeout = 300.0

            [memory]
            max_context_tokens = 16384
            episodic_window = 10

            [workspace]
            base_path = "/tmp/workspaces"

            [ui]
            refresh_rate = 2.0

            [tools]
            command_timeout = 120
            git_enabled = false
            """,
        )
        settings = FleetSettings.from_toml(str(p))
        assert settings.ollama.planner_model == "mistral"
        assert settings.ollama.coder_model == "codellama"
        assert settings.ollama.timeout == 1200.0
        assert settings.scheduler.retry_limit == 5
        assert settings.scheduler.max_concurrent_tasks == 8
        assert settings.memory.max_context_tokens == 16384
        assert settings.workspace.base_path == "/tmp/workspaces"
        assert settings.ui.refresh_rate == 2.0
        assert settings.tools.command_timeout == 120
        assert settings.tools.git_enabled is False

    def test_critic_and_tester_model_optional(self, tmp_path: Path) -> None:
        p = write_toml(
            tmp_path,
            """
            [ollama]
            critic_model = "llama3"
            tester_model = "codellama"
            """,
        )
        settings = FleetSettings.from_toml(str(p))
        assert settings.ollama.critic_model == "llama3"
        assert settings.ollama.tester_model == "codellama"


# ---------------------------------------------------------------------------
# Field constraint enforcement — out-of-range values must exit(1)
# ---------------------------------------------------------------------------


class TestFieldConstraints:
    def _assert_exits(self, tmp_path: Path, content: str) -> None:
        p = write_toml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            FleetSettings.from_toml(str(p))
        assert exc_info.value.code != 0

    def test_timeout_below_minimum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[ollama]\ntimeout = 299.9")

    def test_timeout_above_maximum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[ollama]\ntimeout = 3600.1")

    def test_timeout_at_minimum_is_valid(self, tmp_path: Path) -> None:
        p = write_toml(tmp_path, "[ollama]\ntimeout = 300.0")
        settings = FleetSettings.from_toml(str(p))
        assert settings.ollama.timeout == 300.0

    def test_timeout_at_maximum_is_valid(self, tmp_path: Path) -> None:
        p = write_toml(tmp_path, "[ollama]\ntimeout = 3600.0")
        settings = FleetSettings.from_toml(str(p))
        assert settings.ollama.timeout == 3600.0

    def test_retry_limit_below_minimum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[scheduler]\nretry_limit = 0")

    def test_retry_limit_above_maximum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[scheduler]\nretry_limit = 11")

    def test_max_concurrent_tasks_below_minimum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[scheduler]\nmax_concurrent_tasks = 0")

    def test_max_concurrent_tasks_above_maximum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[scheduler]\nmax_concurrent_tasks = 33")

    def test_max_context_tokens_below_minimum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[memory]\nmax_context_tokens = 1023")

    def test_max_context_tokens_above_maximum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[memory]\nmax_context_tokens = 131073")

    def test_refresh_rate_below_minimum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[ui]\nrefresh_rate = 0.09")

    def test_refresh_rate_above_maximum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[ui]\nrefresh_rate = 10.1")

    def test_command_timeout_below_minimum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[tools]\ncommand_timeout = 0")

    def test_command_timeout_above_maximum(self, tmp_path: Path) -> None:
        self._assert_exits(tmp_path, "[tools]\ncommand_timeout = 3601")

    def test_stderr_contains_field_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = write_toml(tmp_path, "[ollama]\ntimeout = 100.0")
        with pytest.raises(SystemExit):
            FleetSettings.from_toml(str(p))
        captured = capsys.readouterr()
        assert "timeout" in captured.err

    def test_stderr_contains_expected_range(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = write_toml(tmp_path, "[scheduler]\nretry_limit = 0")
        with pytest.raises(SystemExit):
            FleetSettings.from_toml(str(p))
        captured = capsys.readouterr()
        # Should mention the range [1, 10]
        assert "1" in captured.err
        assert "10" in captured.err


# ---------------------------------------------------------------------------
# Missing file behaviour
# ---------------------------------------------------------------------------


class TestMissingFile:
    def test_missing_env_var_file_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = str(tmp_path / "nonexistent.toml")
        with pytest.raises(SystemExit) as exc_info:
            FleetSettings.from_toml(missing, _from_env=True)
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "nonexistent.toml" in captured.err

    def test_missing_fallback_file_returns_defaults(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "nonexistent.toml")
        # _from_env=False (default) → missing file → return defaults, no exit
        settings = FleetSettings.from_toml(missing, _from_env=False)
        assert settings.ollama.timeout == 600.0
        assert settings.scheduler.retry_limit == 3


# ---------------------------------------------------------------------------
# load_settings()
# ---------------------------------------------------------------------------


class TestLoadSettings:
    def test_load_settings_uses_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = write_toml(tmp_path, "[ollama]\ntimeout = 1800.0")
        monkeypatch.setenv("OLLAMA_FLEET_CONFIG", str(p))
        settings = load_settings()
        assert settings.ollama.timeout == 1800.0

    def test_load_settings_fallback_missing_returns_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OLLAMA_FLEET_CONFIG", raising=False)
        # Change cwd so ./config.toml doesn't exist
        monkeypatch.chdir(tmp_path)
        settings = load_settings()
        assert settings.ollama.timeout == 600.0

    def test_load_settings_env_var_missing_file_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_FLEET_CONFIG", str(tmp_path / "missing.toml"))
        with pytest.raises(SystemExit) as exc_info:
            load_settings()
        assert exc_info.value.code != 0

    def test_load_settings_fallback_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OLLAMA_FLEET_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.toml").write_text(
            "[scheduler]\nretry_limit = 7\n", encoding="utf-8"
        )
        settings = load_settings()
        assert settings.scheduler.retry_limit == 7
