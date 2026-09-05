"""FastAPI entry point for the Job Finder Agent.

Serves the backend API and the static frontend from the same origin.

Run from the project root:
    python -m backend.main            # or
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

from .api.routes import router  # noqa: E402

app = FastAPI(title="Job Finder Agent", version="1.0.0")
app.include_router(router)

FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))

if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)