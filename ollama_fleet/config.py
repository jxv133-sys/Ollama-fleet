"""Configuration system — Pydantic settings and TOML loader.

Loads configuration from a TOML file. The path is read from the
``OLLAMA_FLEET_CONFIG`` environment variable, falling back to ``./config.toml``.

On any validation error the module writes the offending field name and its
expected range to *stderr* and exits with a non-zero status code.
"""

from __future__ import annotations

import os
import sys
import tomllib
from typing import Any

from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Sub-section models
# ---------------------------------------------------------------------------


class OllamaConfig(BaseModel):
    """Settings for the Ollama server and model assignments."""

    base_url: str = "http://localhost:11434"
    planner_model: str = "llama3"
    coder_model: str = "llama3"
    summarizer_model: str = "llama3"
    # None means "default to coder_model" at runtime
    critic_model: str | None = None
    tester_model: str | None = None
    timeout: float = Field(default=600.0, ge=300.0, le=3600.0)


class SchedulerConfig(BaseModel):
    """Settings for the task scheduler."""

    retry_limit: int = Field(default=3, ge=1, le=10)
    max_concurrent_tasks: int = Field(default=4, ge=1, le=32)
    poll_interval: float = Field(default=5.0, ge=0.1, le=5.0)
    max_critique_revision_loops: int = Field(default=3, ge=1, le=10)
    stall_timeout: float = Field(default=600.0, ge=1.0)


class MemoryConfig(BaseModel):
    """Settings for the memory system."""

    max_context_tokens: int = Field(default=8192, ge=1024, le=131072)
    episodic_window: int = Field(default=5, ge=1)


class WorkspaceConfig(BaseModel):
    """Settings for the workspace manager."""

    base_path: str = "./workspaces"


class UIConfig(BaseModel):
    """Settings for the terminal UI."""

    refresh_rate: float = Field(default=1.0, ge=0.1, le=10.0)


class ToolsConfig(BaseModel):
    """Settings for the tool runtime."""

    command_timeout: int = Field(default=60, ge=1, le=3600)
    git_enabled: bool = True


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

# Human-readable descriptions of constrained fields used in error messages.
_FIELD_RANGES: dict[str, str] = {
    "ollama.timeout": "float in [300.0, 3600.0]",
    "scheduler.retry_limit": "int in [1, 10]",
    "scheduler.max_concurrent_tasks": "int in [1, 32]",
    "scheduler.poll_interval": "float in [0.1, 5.0]",
    "scheduler.max_critique_revision_loops": "int in [1, 10]",
    "scheduler.stall_timeout": "float >= 1.0",
    "memory.max_context_tokens": "int in [1024, 131072]",
    "memory.episodic_window": "int >= 1",
    "ui.refresh_rate": "float in [0.1, 10.0]",
    "tools.command_timeout": "int in [1, 3600]",
}


def _format_validation_errors(exc: ValidationError, config_path: str) -> str:
    """Return a human-readable error string for a Pydantic ValidationError."""
    lines: list[str] = [f"Configuration error in '{config_path}':"]
    for error in exc.errors():
        # loc is a tuple like ('ollama', 'timeout') or ('scheduler', 'retry_limit')
        loc_parts = [str(p) for p in error["loc"]]
        field_path = ".".join(loc_parts)
        expected = _FIELD_RANGES.get(field_path, error.get("msg", "invalid value"))
        lines.append(f"  field '{field_path}': expected {expected}")
    return "\n".join(lines)


class FleetSettings(BaseModel):
    """Top-level configuration for Ollama Fleet."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @classmethod
    def from_toml(cls, path: str, *, _from_env: bool = False) -> "FleetSettings":
        """Load and validate settings from a TOML file.

        Parameters
        ----------
        path:
            Filesystem path to the TOML configuration file.
        _from_env:
            Internal flag — set to ``True`` when *path* came from the
            ``OLLAMA_FLEET_CONFIG`` environment variable so that a missing
            file produces a more specific error message.

        Raises
        ------
        SystemExit
            On any file-not-found or validation error.
        """
        try:
            with open(path, "rb") as fh:
                raw: dict[str, Any] = tomllib.load(fh)
        except FileNotFoundError:
            if _from_env:
                print(
                    f"Error: OLLAMA_FLEET_CONFIG is set but the file '{path}' does not exist.",
                    file=sys.stderr,
                )
            else:
                # Fall-back path missing — use all defaults silently.
                return cls()
            sys.exit(1)
        except tomllib.TOMLDecodeError as exc:
            print(f"Error: Failed to parse TOML file '{path}': {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            print(_format_validation_errors(exc, path), file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Module-level convenience loader
# ---------------------------------------------------------------------------


def load_settings() -> FleetSettings:
    """Load :class:`FleetSettings` from the configured TOML path.

    Reads the ``OLLAMA_FLEET_CONFIG`` environment variable; falls back to
    ``./config.toml`` when the variable is not set.  Exits non-zero on any
    error (missing env-var file, invalid TOML, or validation failure).
    """
    env_path = os.environ.get("OLLAMA_FLEET_CONFIG")
    if env_path is not None:
        return FleetSettings.from_toml(env_path, _from_env=True)
    return FleetSettings.from_toml("./config.toml", _from_env=False)
