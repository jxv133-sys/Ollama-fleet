from __future__ import annotations

import asyncio
from typing import Any, Callable

EventHandler = Callable[[dict[str, Any]], Any]


class UIEventBus:
    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def publish(self, event: dict[str, Any]) -> None:
        for handler in list(self._handlers):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                continue
