# Job Finder Agent

An AI-powered job search assistant. You enter the role, location and your
skills once, and a bounded agent runs a live web search, validates real
listings, and returns a ranked list of jobs plus a written market report —
never fabricating a company, salary or URL.

It runs a single bounded **UNDERSTAND → SEARCH PLAN → WEB SEARCH → ANALYZE →
VERIFY → REPORT** loop:

- **UNDERSTAND** — the request is parsed into role, location, experience,
  skills, salary and work mode.
- **SEARCH PLAN** — the agent proposes targeted search queries for current job
  postings.
- **WEB SEARCH** — live searches via OpenRouter's server-side web search;
  results come back with real URLs.
- **ANALYZE** — jobs are extracted (AI or deterministic fallback), de-duplicated,
  and scored against your skills.
- **VERIFY** — duplicates and low-signal listings are filtered out in review.
- **REPORT** — a final markdown report: verified job list, market insights,
  salary context, and a skills-gap analysis.

Every extracted field is grounded in real web results. If the AI service is
unavailable or a search returns nothing usable, the app warns and continues
with fully deterministic, honest fallbacks ("Not specified" is used when a
value cannot be found — nothing is ever invented).

## Features

- Web UI served from the same origin as the API (FastAPI + static frontend).
- Live SSE progress stream while the agent works (progress %, activity feed,
  pipeline map).
- Verified job cards with a skill-match score, market insights chips, and a
  skills-gap breakdown.
- Markdown report preview + one-click download.
- Bounded loop with hard ceilings (max steps, max search rounds, deadline) so
  the agent always terminates and never spends unlimited credits.
- OpenRouter AI optional throughout — deterministic fallbacks keep the app
  working when the AI is slow, down, or out of credits.

## Project layout

```
backend/
  agent/           agent loop, prompts/validators, skill matching, deterministic fallbacks
  ai/              OpenRouter client (chat, JSON, server-side web search)
  api/             FastAPI routes (search, health)
  state.py         shared search state + SSE event stream
  main.py          FastAPI entry point (serves frontend too)
frontend/
  index.html       landing, research, and results views
  app.js           SSE client + progress UI + results rendering
  styles.css       dark command-center theme
```

## Requirements

- Python 3.10+ (developed on 3.12)
- `pip install -r backend/requirements.txt`

## Setup

1. Copy the environment template and add your OpenRouter key:

   ```bash
   cp backend/.env.example backend/.env
   ```

   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```

2. Install dependencies (requires a virtual environment or `--user`):

   ```bash
   pip install -r backend/requirements.txt
   ```

## Run

```bash
python -m backend.main
```

Then open http://127.0.0.1:8000.

For development with auto-reload:

```bash
uvicorn backend.main:app --reload
```

## Using it

1. Fill in the search form: role, location, experience level, your skills,
   desired salary, and work mode. Click **Search Jobs**.
2. Watch the research panel stream progress through the six pipeline stages.
3. Review the results: verified job cards (title, company, location, salary,
   experience, posted date, URL), the skill-match score per job, market
   insights and salary context.
4. Download the full markdown report for a written summary and skills-gap
   analysis.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/api/search` | JSON `{role, location, ...}` → SSE stream (agent progress + result) |
| GET | `/api/health` | server + OpenRouter key status |

Example search body:

```json
{
  "role": "Data Analyst",
  "location": "India",
  "experience": "Fresher",
  "skills": "Python, SQL, Excel",
  "salary": "5 LPA",
  "work_mode": "Remote"
}
```

The SSE stream ends with a `result` event containing `{jobs, analysis,
report_markdown, status, degraded}` (job URLs always point to real pages found
during the search — they are never invented).

## Agent safety limits

| Setting | Value |
| --- | --- |
| Max agent steps per run | 6 |
| Max search rounds | 3 |
| Min reliable jobs before extra rounds | 3 |
| Hard deadline per run | 240 s |
| OpenRouter request retries (backoff) | 3 |

## Security notes

- `backend/.env` holds your real API key and is gitignored — never commit it.
- The API key is only ever read on the backend; the frontend never sees it.
- Only the extracted job facts are sent to the LLM (as a compact digest), never
  browsing history or unrelated data.
- No database and no user accounts — nothing to leak; every run is stateless.