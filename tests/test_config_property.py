"""Property-based tests for config validation error completeness.

**Property 23: Config Validation Error Completeness**
**Validates: Requirements 12.3, 12.4**

For any TOML configuration containing an out-of-range value for a constrained
field, FleetSettings.from_toml() SHALL:
  - exit with a non-zero status code, AND
  - write to stderr a message that includes the field name and the expected range.
"""

from __future__ import annotations

import io
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_fleet.config import FleetSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_toml_to_dir(directory: str, content: str) -> str:
    """Write *content* to a TOML file inside *directory* and return its path."""
    p = Path(directory) / "config.toml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


def assert_validation_error(
    toml_content: str,
    field_name: str,
    range_hints: list[str],
) -> None:
    """Assert that loading *toml_content* exits non-zero and mentions the field + range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_toml_to_dir(tmpdir, toml_content)
        # Capture stderr by temporarily redirecting it
        old_stderr = sys.stderr
        sys.stderr = captured = io.StringIO()
        try:
            with pytest.raises(SystemExit) as exc_info:
                FleetSettings.from_toml(path)
        finally:
            sys.stderr = old_stderr
        err_output = captured.getvalue()

    # Requirement 12.3 / 12.4: exit with non-zero status code
    assert exc_info.value.code != 0, "Expected non-zero exit code on validation error"
    # Requirement 12.3 / 12.4: stderr must contain the field name
    assert field_name in err_output, (
        f"Expected field name '{field_name}' in stderr, got: {err_output!r}"
    )
    # Requirement 12.3 / 12.4: stderr must contain the expected range
    for hint in range_hints:
        assert hint in err_output, (
            f"Expected range hint '{hint}' in stderr, got: {err_output!r}"
        )


# ---------------------------------------------------------------------------
# Property 23: Config Validation Error Completeness
# ---------------------------------------------------------------------------


class TestConfigValidationErrorCompleteness:
    """Property 23: Config Validation Error Completeness.

    **Validates: Requirements 12.3, 12.4**

    For every constrained field, any out-of-range value must produce:
      - a non-zero exit code
      - stderr output containing the field name and the expected range
    """

    # --- [ollama] timeout: valid range [300.0, 3600.0] ---

    @given(
        timeout=st.fixed_dictionaries(
            {
                "value": st.one_of(
                    st.floats(min_value=-1e9, max_value=299.99, allow_nan=False, allow_infinity=False),
                    st.floats(min_value=3600.01, max_value=1e9, allow_nan=False, allow_infinity=False),
                )
            }
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_timeout_out_of_range(
        self,
        timeout: dict,
    ) -> None:
        """Out-of-range ollama.timeout must exit non-zero and mention field + range."""
        toml = f"[ollama]\ntimeout = {timeout['value']}\n"
        assert_validation_error(
            toml,
            field_name="timeout",
            range_hints=["300", "3600"],
        )

    # --- [scheduler] retry_limit: valid range [1, 10] ---

    @given(
        cfg=st.fixed_dictionaries(
            {
                "retry_limit": st.one_of(
                    st.integers(min_value=-1000, max_value=0),
                    st.integers(min_value=11, max_value=1000),
                )
            }
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_retry_limit_out_of_range(
        self,
        cfg: dict,
    ) -> None:
        """Out-of-range scheduler.retry_limit must exit non-zero and mention field + range."""
        toml = f"[scheduler]\nretry_limit = {cfg['retry_limit']}\n"
        assert_validation_error(
            toml,
            field_name="retry_limit",
            range_hints=["1", "10"],
        )

    # --- [scheduler] max_concurrent_tasks: valid range [1, 32] ---

    @given(
        cfg=st.fixed_dictionaries(
            {
                "max_concurrent_tasks": st.one_of(
                    st.integers(min_value=-1000, max_value=0),
                    st.integers(min_value=33, max_value=1000),
                )
            }
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_max_concurrent_tasks_out_of_range(
        self,
        cfg: dict,
    ) -> None:
        """Out-of-range scheduler.max_concurrent_tasks must exit non-zero and mention field + range."""
        toml = f"[scheduler]\nmax_concurrent_tasks = {cfg['max_concurrent_tasks']}\n"
        assert_validation_error(
            toml,
            field_name="max_concurrent_tasks",
            range_hints=["1", "32"],
        )

    # --- [memory] max_context_tokens: valid range [1024, 131072] ---

    @given(
        cfg=st.fixed_dictionaries(
            {
                "max_context_tokens": st.one_of(
                    st.integers(min_value=-1000, max_value=1023),
                    st.integers(min_value=131073, max_value=500000),
                )
            }
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_max_context_tokens_out_of_range(
        self,
        cfg: dict,
    ) -> None:
        """Out-of-range memory.max_context_tokens must exit non-zero and mention field + range."""
        toml = f"[memory]\nmax_context_tokens = {cfg['max_context_tokens']}\n"
        assert_validation_error(
            toml,
            field_name="max_context_tokens",
            range_hints=["1024", "131072"],
        )

    # --- [ui] refresh_rate: valid range [0.1, 10.0] ---

    @given(
        cfg=st.fixed_dictionaries(
            {
                "refresh_rate": st.one_of(
                    st.floats(min_value=-1e6, max_value=0.09, allow_nan=False, allow_infinity=False),
                    st.floats(min_value=10.01, max_value=1e6, allow_nan=False, allow_infinity=False),
                )
            }
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_refresh_rate_out_of_range(
        self,
        cfg: dict,
    ) -> None:
        """Out-of-range ui.refresh_rate must exit non-zero and mention field + range."""
        toml = f"[ui]\nrefresh_rate = {cfg['refresh_rate']}\n"
        assert_validation_error(
            toml,
            field_name="refresh_rate",
            range_hints=["0.1", "10.0"],
        )

    # --- [tools] command_timeout: valid range [1, 3600] ---

    @given(
        cfg=st.fixed_dictionaries(
            {
                "command_timeout": st.one_of(
                    st.integers(min_value=-1000, max_value=0),
                    st.integers(min_value=3601, max_value=100000),
                )
            }
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_command_timeout_out_of_range(
        self,
        cfg: dict,
    ) -> None:
        """Out-of-range tools.command_timeout must exit non-zero and mention field + range."""
        toml = f"[tools]\ncommand_timeout = {cfg['command_timeout']}\n"
        assert_validation_error(
            toml,
            field_name="command_timeout",
            range_hints=["1", "3600"],
        )

    # --- Combined: multiple out-of-range fields in one config ---

    @given(
        cfg=st.fixed_dictionaries(
            {
                "timeout": st.one_of(
                    st.floats(min_value=-1e6, max_value=299.99, allow_nan=False, allow_infinity=False),
                    st.floats(min_value=3600.01, max_value=1e6, allow_nan=False, allow_infinity=False),
                ),
                "retry_limit": st.one_of(
                    st.integers(min_value=-100, max_value=0),
                    st.integers(min_value=11, max_value=100),
                ),
            }
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_multiple_out_of_range_fields(
        self,
        cfg: dict,
    ) -> None:
        """Multiple out-of-range fields: exit non-zero and stderr mentions at least one field."""
        toml = (
            f"[ollama]\ntimeout = {cfg['timeout']}\n"
            f"[scheduler]\nretry_limit = {cfg['retry_limit']}\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_toml_to_dir(tmpdir, toml)
            old_stderr = sys.stderr
            sys.stderr = captured = io.StringIO()
            try:
                with pytest.raises(SystemExit) as exc_info:
                    FleetSettings.from_toml(path)
            finally:
                sys.stderr = old_stderr
            err_output = captured.getvalue()

        assert exc_info.value.code != 0
        # At least one of the two invalid fields must appear in stderr
        assert "timeout" in err_output or "retry_limit" in err_output, (
            f"Expected at least one field name in stderr, got: {err_output!r}"
        )
