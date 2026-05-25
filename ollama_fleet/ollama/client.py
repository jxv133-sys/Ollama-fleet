"""Async HTTP client for the Ollama REST API with streaming support."""

from __future__ import annotations

import json

import httpx


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

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
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
            async with httpx.AsyncClient() as client:
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
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        full_response.append(chunk.get("response", ""))
                        if chunk.get("done"):
                            break

                    return "".join(full_response)

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
