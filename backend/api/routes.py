"""HTTP API for the Job Finder Agent.

`POST /api/search` streams a real-time investigation via SSE.
`GET /api/health` reports service readiness.
The OpenRouter API key never leaves the backend.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..agent.agent import JobFinderAgent
from ..ai.openrouter import OpenRouterClient, OpenRouterError
from ..state import EventStream, sse_frame

router = APIRouter()

MAX_ROLE_LEN = 100
MAX_LOCATION_LEN = 100
MAX_EXPERIENCE_LEN = 80
MAX_SKILLS_LEN = 500
MAX_SALARY_LEN = 100
MAX_WORK_MODE_LEN = 60


class SearchRequest(BaseModel):
    role: str = "Data Analyst"
    location: str = "India"
    experience: str = "Fresher"
    skills: str = ""
    salary: str = ""
    work_mode: str = ""


@router.get("/api/health")
async def health() -> dict:
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    return {
        "ok": True,
        "openrouter_key_configured": bool(key) and key != "your_key_here",
        "model": os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini",
    }


@router.post("/api/search")
async def search(body: SearchRequest):
    # Bound user input so we never pass unbounded text to the model.
    request = {
        "role": (body.role or "").strip()[:MAX_ROLE_LEN],
        "location": (body.location or "").strip()[:MAX_LOCATION_LEN],
        "experience": (body.experience or "").strip()[:MAX_EXPERIENCE_LEN],
        "skills": (body.skills or "").strip()[:MAX_SKILLS_LEN],
        "salary": (body.salary or "").strip()[:MAX_SALARY_LEN],
        "work_mode": (body.work_mode or "").strip()[:MAX_WORK_MODE_LEN],
    }

    async def stream():
        events = EventStream()
        client: OpenRouterClient | None = None
        try:
            try:
                client = OpenRouterClient()
            except OpenRouterError as exc:
                client = None
                events.activity(
                    "warn", f"OpenRouter unavailable: {exc}. Bounded results only."
                )

            agent = JobFinderAgent(request, events, client)
            task = asyncio.create_task(agent.run())

            while True:
                item = await events.poll()
                if item is None:
                    break
                yield sse_frame(item)
            await task
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )