from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ollama_fleet.config import FleetSettings
from ollama_fleet.memory.episodic import EpisodicEntry, EpisodicMemory
from ollama_fleet.memory.long_term import LongTermEntry, LongTermMemory
from ollama_fleet.ollama.client import OllamaClient
from ollama_fleet.db.database import Database


@dataclass
class ActiveContext:
    task_description: str
    active_files: list[str]
    file_contents: dict[str, str]
    episodic_summaries: list[str]


class MemorySystem:
    def __init__(self, db: Database, settings: FleetSettings, client: OllamaClient | None = None) -> None:
        self._memory = EpisodicMemory(db)
        self._long_term = LongTermMemory(db)
        self._settings = settings
        self._client = client

    async def assemble_context(
        self,
        task_description: str,
        job_id: str,
        active_files: list[str] | None = None,
        file_contents: dict[str, str] | None = None,
    ) -> ActiveContext:
        active_files = active_files or []
        file_contents = file_contents or {}
        recent_entries = await self._memory.get_recent(job_id, self._settings.memory.episodic_window)
        episodic_summaries = [entry.summary_text for entry in recent_entries]

        if self._estimate_tokens(self._context_text(file_contents, episodic_summaries)) > self._settings.memory.max_context_tokens:
            if self._client is not None:
                compressed = await self._compress_context(file_contents, episodic_summaries)
                episodic_summaries = compressed
            else:
                episodic_summaries = episodic_summaries[-self._settings.memory.episodic_window:]

        return ActiveContext(
            task_description=task_description,
            active_files=active_files,
            file_contents=file_contents,
            episodic_summaries=episodic_summaries,
        )

    async def save_episodic(self, entry: EpisodicEntry) -> None:
        await self._memory.save(entry)

    async def search_long_term(self, job_id: str, query: str) -> list[LongTermEntry]:
        return await self._long_term.search(job_id, query)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _context_text(self, file_contents: dict[str, str], summaries: list[str]) -> str:
        return "\n".join(file_contents.values()) + "\n" + "\n".join(summaries)

    async def _compress_context(self, file_contents: dict[str, str], summaries: list[str]) -> list[str]:
        if self._client is None:
            return summaries[-self._settings.memory.episodic_window:]

        prompt = (
            "Compress the following context into a concise list of episodic summaries. "
            "Preserve important details and remove redundant text.\n\n"
        )
        for path, content in file_contents.items():
            prompt += f"File: {path}\n{content}\n\n"
        if summaries:
            prompt += "Previous summaries:\n" + "\n".join(summaries)

        response = await self._client.generate(
            model=self._settings.ollama.summarizer_model,
            prompt=prompt,
            timeout=self._settings.ollama.timeout,
        )
        return [response.strip()]
