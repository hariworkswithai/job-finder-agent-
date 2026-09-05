"""Job search event streaming.

Two concerns, cleanly separated:

- `JobSearchState` is plain task state (what has been done, what was found).
- `EventStream` is a thin asyncio queue that serializes state changes into
  SSE events for the frontend.

Only concise, safe messages are ever emitted. Hidden reasoning (chain of
thought) is never exposed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field


@dataclass
class JobSearchState:
    status: str = "running"  # running | completed | degraded | error
    role: str = ""
    location: str = ""
    experience: str = ""
    skills: str = ""
    salary: str = ""
    work_mode: str = ""
    current_step: str = ""
    completed_steps: list[str] = field(default_factory=list)
    jobs_found: int = 0
    model: str = ""
    ai_available: bool = True

    def to_public(self) -> dict:
        return {
            "status": self.status,
            "role": self.role,
            "location": self.location,
            "experience": self.experience,
            "current_step": self.current_step,
            "completed_steps": list(self.completed_steps),
            "jobs_found": self.jobs_found,
            "model": self.model,
            "ai_available": self.ai_available,
        }


class EventStream:
    """Frontend-facing event queue. Handlers push events; the SSE generator pops."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()

    def emit(self, type_: str, data: dict) -> None:
        self._queue.put_nowait({"type": type_, **data})

    def activity(self, kind: str, message: str) -> None:
        self.emit("activity", {"kind": kind, "message": message})

    def progress(self, percent: int, label: str) -> None:
        self.emit("progress", {"percent": percent, "label": label})

    def step(self, index: int, total: int, label: str) -> None:
        if total > 0:
            self.emit("step", {"index": index, "total": total, "label": label})

    def error_event(self, message: str, detail: str = "") -> None:
        self.emit("error", {"message": message, "detail": detail})

    def result(self, payload: dict) -> None:
        self.emit("result", {"payload": payload})

    def done(self) -> None:
        self._queue.put_nowait(None)

    async def poll(self):
        return await self._queue.get()


def sse_frame(item: dict) -> str:
    return f"data: {json.dumps(item, ensure_ascii=False)}\n\n"