"""Unit tests for ollama_fleet/ollama/client.py.

Tests cover:
- Successful streaming accumulation (single and multi-chunk)
- Stop on "done": true mid-stream
- OllamaHTTPError raised for 4xx/5xx responses
- OllamaConnectionError raised for connection failures
- OllamaTimeoutError raised for timeout failures
- Empty lines in stream are skipped
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ollama_fleet.ollama.client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaTimeoutError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_lines(*chunks: dict) -> list[str]:
    """Encode a sequence of chunk dicts as newline-delimited JSON strings."""
    return [json.dumps(c) for c in chunks]


def _mock_stream_response(lines: list[str], status_code: int = 200):
    """Return an async context manager that yields a mock streaming response."""

    async def _aiter_lines():
        for line in lines:
            yield line

    resp = MagicMock()
    resp.status_code = status_code
    resp.aiter_lines = _aiter_lines
    resp.aread = AsyncMock(return_value=b"error body")

    @asynccontextmanager
    async def _stream_ctx(*args, **kwargs):
        yield resp

    mock_client = MagicMock()
    mock_client.stream = _stream_ctx

    @asynccontextmanager
    async def _client_ctx():
        yield mock_client

    return _client_ctx


# ---------------------------------------------------------------------------
# Successful streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_single_chunk():
    """A single-chunk response with done=true returns the response text."""
    lines = _make_stream_lines({"response": "hello", "done": True})
    ctx = _mock_stream_response(lines)

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", ctx):
        client = OllamaClient()
        result = await client.generate("llama3", "say hello", timeout=30.0)

    assert result == "hello"


@pytest.mark.asyncio
async def test_generate_multi_chunk_accumulation():
    """Multiple chunks are concatenated in order."""
    lines = _make_stream_lines(
        {"response": "foo", "done": False},
        {"response": "bar", "done": False},
        {"response": "baz", "done": True},
    )
    ctx = _mock_stream_response(lines)

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", ctx):
        client = OllamaClient()
        result = await client.generate("llama3", "prompt", timeout=30.0)

    assert result == "foobarbaz"


@pytest.mark.asyncio
async def test_generate_stops_at_done_true():
    """Chunks after done=true are not included in the result."""
    lines = _make_stream_lines(
        {"response": "part1", "done": False},
        {"response": "part2", "done": True},
        # This chunk should never be consumed
        {"response": "SHOULD_NOT_APPEAR", "done": False},
    )
    ctx = _mock_stream_response(lines)

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", ctx):
        client = OllamaClient()
        result = await client.generate("llama3", "prompt", timeout=30.0)

    assert result == "part1part2"
    assert "SHOULD_NOT_APPEAR" not in result


@pytest.mark.asyncio
async def test_generate_skips_empty_lines():
    """Empty lines in the stream are silently skipped."""
    lines = ["", json.dumps({"response": "hello", "done": True}), ""]
    ctx = _mock_stream_response(lines)

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", ctx):
        client = OllamaClient()
        result = await client.generate("llama3", "prompt", timeout=30.0)

    assert result == "hello"


@pytest.mark.asyncio
async def test_generate_missing_response_field():
    """Chunks without a 'response' key contribute empty string."""
    lines = _make_stream_lines(
        {"done": False},  # no 'response' key
        {"response": "world", "done": True},
    )
    ctx = _mock_stream_response(lines)

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", ctx):
        client = OllamaClient()
        result = await client.generate("llama3", "prompt", timeout=30.0)

    assert result == "world"


# ---------------------------------------------------------------------------
# HTTP error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 500, 502, 503])
async def test_generate_raises_http_error_for_error_status(status_code: int):
    """4xx and 5xx responses raise OllamaHTTPError with correct status_code."""
    ctx = _mock_stream_response([], status_code=status_code)

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", ctx):
        client = OllamaClient()
        with pytest.raises(OllamaHTTPError) as exc_info:
            await client.generate("llama3", "prompt", timeout=30.0)

    assert exc_info.value.status_code == status_code
    assert isinstance(exc_info.value.body, str)


@pytest.mark.asyncio
async def test_http_error_contains_body():
    """OllamaHTTPError.body contains the response body text."""
    ctx = _mock_stream_response([], status_code=404)

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", ctx):
        client = OllamaClient()
        with pytest.raises(OllamaHTTPError) as exc_info:
            await client.generate("llama3", "prompt", timeout=30.0)

    # body is decoded from the mock aread() return value b"error body"
    assert exc_info.value.body == "error body"


# ---------------------------------------------------------------------------
# Connection error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_raises_connection_error_on_connect_error():
    """httpx.ConnectError is mapped to OllamaConnectionError."""

    @asynccontextmanager
    async def _failing_client():
        mock = MagicMock()

        @asynccontextmanager
        async def _stream(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")
            yield  # make it a generator

        mock.stream = _stream
        yield mock

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", _failing_client):
        client = OllamaClient()
        with pytest.raises(OllamaConnectionError):
            await client.generate("llama3", "prompt", timeout=30.0)


@pytest.mark.asyncio
async def test_generate_raises_connection_error_on_connect_timeout():
    """httpx.ConnectTimeout is mapped to OllamaConnectionError (pre-response TCP timeout)."""

    @asynccontextmanager
    async def _failing_client():
        mock = MagicMock()

        @asynccontextmanager
        async def _stream(*args, **kwargs):
            raise httpx.ConnectTimeout("Timed out connecting")
            yield

        mock.stream = _stream
        yield mock

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", _failing_client):
        client = OllamaClient()
        with pytest.raises(OllamaConnectionError):
            await client.generate("llama3", "prompt", timeout=30.0)


# ---------------------------------------------------------------------------
# Timeout error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_raises_timeout_error_on_read_timeout():
    """httpx.ReadTimeout is mapped to OllamaTimeoutError."""

    @asynccontextmanager
    async def _failing_client():
        mock = MagicMock()

        @asynccontextmanager
        async def _stream(*args, **kwargs):
            raise httpx.ReadTimeout("Read timed out")
            yield

        mock.stream = _stream
        yield mock

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", _failing_client):
        client = OllamaClient()
        with pytest.raises(OllamaTimeoutError):
            await client.generate("llama3", "prompt", timeout=30.0)


@pytest.mark.asyncio
async def test_generate_raises_timeout_error_on_pool_timeout():
    """httpx.PoolTimeout is mapped to OllamaTimeoutError."""

    @asynccontextmanager
    async def _failing_client():
        mock = MagicMock()

        @asynccontextmanager
        async def _stream(*args, **kwargs):
            raise httpx.PoolTimeout("Pool timed out")
            yield

        mock.stream = _stream
        yield mock

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", _failing_client):
        client = OllamaClient()
        with pytest.raises(OllamaTimeoutError):
            await client.generate("llama3", "prompt", timeout=30.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_client_default_base_url():
    """OllamaClient defaults to http://localhost:11434."""
    client = OllamaClient()
    assert client.base_url == "http://localhost:11434"


def test_client_custom_base_url():
    """OllamaClient accepts a custom base_url."""
    client = OllamaClient(base_url="http://192.168.50.142:7869/v1")
    assert client.base_url == "http://192.168.50.142:7869/v1"


def test_client_strips_trailing_slash():
    """Trailing slashes are stripped from base_url."""
    client = OllamaClient(base_url="http://localhost:11434/")
    assert client.base_url == "http://localhost:11434"


@pytest.mark.asyncio
async def test_generate_posts_to_correct_endpoint():
    """generate() sends a POST to /api/generate with correct payload."""
    captured: dict = {}

    @asynccontextmanager
    async def _capturing_client():
        mock = MagicMock()

        @asynccontextmanager
        async def _stream(method, url, *, json, timeout):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout

            resp = MagicMock()
            resp.status_code = 200

            async def _aiter_lines():
                yield json_line({"response": "ok", "done": True})

            resp.aiter_lines = _aiter_lines
            yield resp

        mock.stream = _stream
        yield mock

    def json_line(d: dict) -> str:
        return json.dumps(d)

    with patch("ollama_fleet.ollama.client.httpx.AsyncClient", _capturing_client):
        client = OllamaClient(base_url="http://myhost:11434")
        await client.generate("mymodel", "my prompt", timeout=60.0)

    assert captured["method"] == "POST"
    assert captured["url"] == "http://myhost:11434/api/generate"
    assert captured["json"]["model"] == "mymodel"
    assert captured["json"]["prompt"] == "my prompt"
    assert captured["json"]["format"] == "json"
    assert captured["timeout"] == 60.0
