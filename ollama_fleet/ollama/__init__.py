"""Ollama package — async HTTP client for the Ollama REST API."""

from ollama_fleet.ollama.client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaTimeoutError,
)

__all__ = [
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaHTTPError",
    "OllamaTimeoutError",
]
