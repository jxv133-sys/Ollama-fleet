"""Async HTTP client for the Ollama REST API with streaming support."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when the Ollama server is unreachable.

    Covers connection refused, DNS resolution failure, and TCP-level timeouts
    that occur before a response is received.
    """


class OllamaHTTPError(Exception):
    """Raised when the Ollama server returns a 4xx or 5xx HTTP response.

    Attributes:
        status_code: The HTTP status code returned by the server.
        body: The raw response body as a string.
    """

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Ollama HTTP error {status_code}: {body}")


class OllamaTimeoutError(Exception):
    """Raised when the per-request timeout is exceeded during a generation call.

    This covers read timeouts and pool timeouts that occur after the connection
    is established but before the full response is received.
    """


class OllamaClient:
    """Async client for the Ollama REST API.

    Uses ``httpx.AsyncClient`` with streaming to accumulate line-delimited JSON
    chunks from the ``/api/generate`` endpoint.

    Args:
        base_url: Base URL of the Ollama server. Defaults to
            ``http://localhost:11434``.
    """

    def __init__(self, base_url: str = "http://192.168.50.142:7869") -> None:
        self.base_url = base_url.rstrip("/")

    async def generate(self, model: str, prompt: str, timeout: float) -> str:
        """Send a generation request and return the accumulated response text.

        Streams the response from ``POST /api/generate``, accumulating the
        ``"response"`` field from each JSON chunk until ``"done": true`` is
        received.

        Args:
            model: The Ollama model name to use (e.g. ``"llama3"``).
            prompt: The prompt string to send to the model.
            timeout: Per-request timeout in seconds. When exceeded, the
                in-flight request is cancelled and :exc:`OllamaTimeoutError`
                is raised.

        Returns:
            The concatenated response text from all streamed chunks.

        Raises:
            OllamaConnectionError: If the server is unreachable (connection
                refused, DNS failure, or TCP timeout before a response).
            OllamaHTTPError: If the server returns a 4xx or 5xx status code.
            OllamaTimeoutError: If the per-request timeout is exceeded.
        """
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "format": "json"}

        try:
            timeout_config = httpx.Timeout(timeout, connect=10.0, read=timeout, write=timeout, pool=timeout)
            try:
                client_cm = httpx.AsyncClient(http2=False, trust_env=False, headers={"Accept-Encoding": "identity"})
            except TypeError:
                # Some test helpers patch AsyncClient with a callable that doesn't
                # accept kwargs. Fall back to calling without kwargs so tests
                # and older httpx versions continue to work.
                client_cm = httpx.AsyncClient()

            async with client_cm as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    timeout=timeout,
                ) as resp:
                    # Raise typed error for 4xx/5xx before reading the body.
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise OllamaHTTPError(
                            status_code=resp.status_code,
                            body=body.decode("utf-8", errors="replace"),
                        )

                    full_response: list[str] = []
                    done = False

                    async def _process_line(line: str) -> bool:
                        line = line.strip()
                        if not line:
                            return False
                        try:
                            chunk_obj = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                "OllamaClient.invalid_stream_chunk line=%r",
                                line,
                            )
                            return False

                        response_chunk = chunk_obj.get("response", "")
                        thinking_chunk = chunk_obj.get("thinking", "")
                        if response_chunk:
                            full_response.append(response_chunk)
                        elif thinking_chunk:
                            full_response.append(thinking_chunk)
                        return bool(chunk_obj.get("done"))

                    # Prefer byte-level iteration to correctly handle servers that
                    # stream partial JSON tokens across chunks. This is more robust
                    # across httpx versions and real-world Ollama servers. Fall
                    # back to line-level iteration or a final read if bytes
                    # iteration is unavailable.
                    buffer = bytearray()
                    if hasattr(resp, "aiter_bytes"):
                        async for chunk in resp.aiter_bytes():
                            if not chunk:
                                continue
                            buffer.extend(chunk)
                            while True:
                                newline_index = buffer.find(b"\n")
                                if newline_index == -1:
                                    break
                                line_bytes = bytes(buffer[:newline_index])
                                del buffer[: newline_index + 1]
                                line_str = line_bytes.decode("utf-8", errors="replace")
                                logger.debug("OllamaClient.stream_line %r", line_str)
                                if await _process_line(line_str):
                                    done = True
                                    break
                            if done:
                                break
                        if not done and buffer:
                            tail = buffer.decode("utf-8", errors="replace").strip()
                            if tail:
                                logger.debug("OllamaClient.stream_tail %r", tail)
                                await _process_line(tail)
                    elif hasattr(resp, "aiter_lines"):
                        async for line in resp.aiter_lines():
                            logger.debug("OllamaClient.stream_line %r", line)
                            if await _process_line(line):
                                done = True
                                break
                    else:
                        # Last-resort: read the remaining body in one go.
                        tail = await resp.aread()
                        tail_str = tail.decode("utf-8", errors="replace").strip()
                        if tail_str:
                            logger.debug("OllamaClient.stream_read %r", tail_str)
                            await _process_line(tail_str)

                    joined = "".join(full_response).strip()
                    if not joined:
                        return joined

                    # If the accumulated chunks look like JSON, try to
                    # validate them before returning so callers (agents)
                    # that expect a structured payload can proceed.
                    first_char = joined[0]
                    if first_char in ("{", "["):
                        try:
                            json.loads(joined)
                            logger.debug("OllamaClient.generated_valid_json")
                            return joined
                        except json.JSONDecodeError:
                            logger.debug("OllamaClient.generated_invalid_json_attempt", extra={"sample": joined[:200]})

                    return joined

        except OllamaHTTPError:
            # Re-raise our own typed errors without wrapping them.
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Request to Ollama timed out after {timeout}s: {exc}"
            ) from exc

    async def list_models(self) -> list[dict[str, Any]]:
        """Return the list of models available on the Ollama server.

        Performs a GET request to ``/api/models`` and returns the parsed JSON
        response which is expected to be a list of model metadata objects.
        """
        endpoints = ["/api/models", "/models"]
        last_err: Exception | None = None
        async with httpx.AsyncClient() as client:
            for ep in endpoints:
                url = f"{self.base_url}{ep}"
                try:
                    resp = await client.get(url, timeout=10.0)
                    if resp.status_code == 404:
                        # try the next candidate endpoint
                        last_err = OllamaHTTPError(status_code=resp.status_code, body=resp.text)
                        continue
                    if resp.status_code >= 400:
                        raise OllamaHTTPError(status_code=resp.status_code, body=resp.text)
                    return resp.json()
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last_err = OllamaConnectionError(f"Could not connect to Ollama at {self.base_url}: {exc}")
                    break

        # If we get here, none of the endpoints succeeded. Raise the last error if present,
        # otherwise return an empty list to allow callers to fall back.
        if last_err:
            raise last_err
        return []
