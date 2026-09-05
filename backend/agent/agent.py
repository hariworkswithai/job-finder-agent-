"""The Job Finder Agent — a single bounded agent loop.

Pipeline:

    1. understand the user's request
    2. build a search plan (LLM)
    3. search the web per plan query (OpenRouter web search)
    4. extract jobs from search output (LLM)
    5. deduplicate + analyze (LLM) + deterministic skill match
    6. verify listings (LLM)
    7. re-search if too few reliable results (bounded)
    8. generate the final report (LLM)

Guarantees: maximum steps, graceful degradation when AI/search fails, no
fabrication (missing data = "Not specified"), safe public status events only.
"""

from __future__ import annotations

import re
import time

from ..ai.openrouter import OpenRouterClient, OpenRouterError
from ..state import EventStream, JobSearchState, sse_frame
from .prompts import (
    SYSTEM_ANALYSIS,
    SYSTEM_JOB_EXTRACTION,
    SYSTEM_REPORT,
    SYSTEM_SEARCH_PLAN,
    SYSTEM_VERIFY,
    analysis_validator,
    job_extraction_validator,
    search_plan_validator,
    user_analysis_prompt,
    user_job_extraction_prompt,
    user_report_prompt,
    user_search_plan_prompt,
    user_verify_prompt,
    verify_validator,
)
from .skill_match import (
    compute_skill_match,
    compute_skill_match_score,
)

MAX_AGENT_STEPS = 6          # hard bound on loop iterations
MAX_SEARCH_ROUNDS = 3        # how many full search rounds we allow
MIN_RELIABLE_JOBS = 3
AGENT_DEADLINE_SECONDS = 240.0
SEARCH_MAX_TOKENS = 900
EXTRACT_MAX_TOKENS = 2000
ANALYSIS_MAX_TOKENS = 1200
VERIFY_MAX_TOKENS = 1500
REPORT_MAX_TOKENS = 3000


class JobFinderAgent:
    def __init__(
        self,
        request: dict,
        events: EventStream,
        client: OpenRouterClient | None,
    ):
        self.request = request
        self.events = events
        self.client = client
        self.state = JobSearchState(**request)
        self.jobs: list[dict] = []
        self.verified_jobs: list[dict] = []
        self.search_rounds = 0
        self._step = 0

    # ---------------------------------------------------------------- public
    async def run(self) -> None:
        started = time.monotonic()
        try:
            self.state.status = "running"
            self.events.activity("ok", "Request understood")
            self.events.activity("info", f"Looking for {self.request['role']} in {self.request['location']}")

            # ---- step 1: understand
            self.events.step(self._next_step(), TOTAL_STEPS, "Understanding request")
            await self._understand()

            # ---- step 2: search plan
            self.events.step(self._next_step(), TOTAL_STEPS, "Creating search plan")
            await self._plan_searches()

            # ---- main loop: search -> extract -> (analyze/verify) with repeats
            reporting = False
            while self._step < MAX_AGENT_STEPS:
                if time.monotonic() - started > AGENT_DEADLINE_SECONDS:
                    self.events.activity("info", "Deadline reached — finalizing with what we have.")
                    break
                if reporting:
                    break
                if self.search_rounds >= MAX_SEARCH_ROUNDS:
                    break

                await self._search_round()
                if not self.jobs:
                    self.events.activity("warn", "No jobs collected — trying another round.")
                    continue

                analysis = await self._analyze(only_market=True)
                self.events.step(self._next_step(), TOTAL_STEPS, "Verifying results")
                await self._verify()
                if not self.verified_jobs:
                    self.events.activity("warn", "All listings were unreliable — re-searching.")
                    continue

                need_more = len(self.verified_jobs) < MIN_RELIABLE_JOBS and self.search_rounds < MAX_SEARCH_ROUNDS
                if need_more:
                    self.events.activity("info", "Too few reliable jobs — searching again.")
                    continue

                reporting = True
                await self._finalize(analysis)
                break

            if not reporting:
                await self._finalize_fallback()

        except Exception as exc:
            self.state.status = "error"
            self.events.error_event("Job search failed unexpectedly.", f"{type(exc).__name__}: {exc}")
        finally:
            self.events.done()

    # ------------------------------------------------------------ phases
    async def _understand(self) -> None:
        """Normalize the user request into a clean search-request dict."""
        role = (self.request.get("role") or "").strip()
        location = (self.request.get("location") or "").strip()
        experience = (self.request.get("experience") or "").strip()
        skills = (self.request.get("skills") or "").strip()
        salary = (self.request.get("salary") or "").strip()
        work_mode = (self.request.get("work_mode") or "").strip()

        self.request["role"] = role or "Software Engineer"
        self.request["location"] = location or "Remote"
        self.request["experience"] = experience or "Not specified"
        self.request["skills"] = skills
        self.request["salary"] = salary
        self.request["work_mode"] = work_mode

    async def _plan_searches(self) -> None:
        """Build the list of search queries (LLM or a deterministic fallback)."""
        queries: list[str] = []
        if self.client is not None:
            try:
                data = await self.client.chat_json(
                    SYSTEM_SEARCH_PLAN,
                    user_search_plan_prompt(self.request),
                    temperature=0.3,
                    max_tokens=SEARCH_MAX_TOKENS,
                    validator=search_plan_validator,
                )
                queries = [str(q) for q in data["queries"]]
            except OpenRouterError:
                self.events.activity("info", "Search plan via AI unavailable — using standard queries.")

        if not queries:
            queries = self._fallback_queries()

        self.queries = queries
        self.events.activity("ok", f"Search plan created ({len(queries)} queries)")

    async def _search_round(self) -> None:
        """Run one round of web searches for all planned queries."""
        self.search_rounds += 1
        roles = []
        for idx, query in enumerate(self.queries, start=1):
            self.events.activity("action", f"Searching: {query}")
            if self.client is not None:
                try:
                    result = await self.client.web_search(query, max_results=8)
                    for block in result:
                        roles.append(_block_to_text(block))
                except OpenRouterError as exc:
                    self.events.activity("warn", f"Search failed for '{query}': {exc}")
            else:
                self.events.activity("warn", f"No AI endpoint configured — skipping search '{query}'.")

        if not roles:
            return

        search_output = "\n\n---\n\n".join(roles)
        extract_jobs = await self._extract_jobs(search_output)
        self._dedupe_into(extract_jobs)
        self.state.jobs_found = len(self.jobs)
        self.events.activity("ok", f"{len(self.jobs)} unique jobs collected")

    async def _extract_jobs(self, search_output: str) -> list[dict]:
        """Extract structured jobs from raw search output (LLM or deterministic)."""
        if self.client is not None:
            tries = 0
            while tries < 2:
                tries += 1
                try:
                    data = await self.client.chat_json(
                        SYSTEM_JOB_EXTRACTION,
                        user_job_extraction_prompt(self.request["role"], search_output),
                        temperature=0.1,
                        max_tokens=EXTRACT_MAX_TOKENS,
                        validator=job_extraction_validator,
                    )
                    return data["jobs"]
                except OpenRouterError:
                    break
            self.events.activity("warn", "AI job extraction unavailable — falling back to raw listings.")

        return deterministic_extract(search_output, role=self.request["role"])

    async def _analyze(self, only_market: bool = True) -> dict:
        """Analyze collected jobs (LLM analysis + deterministic skill match)."""
        self.events.activity("action", "Analyzing job market")
        analysis: dict = {}

        if self.client is not None and self.jobs:
            try:
                analysis = await self.client.chat_json(
                    SYSTEM_ANALYSIS,
                    user_analysis_prompt(self.jobs),
                    temperature=0.2,
                    max_tokens=ANALYSIS_MAX_TOKENS,
                    validator=analysis_validator,
                )
            except OpenRouterError:
                analysis = {}

        if not analysis:
            analysis = deterministic_analysis(self.jobs)

        self.skill_match = compute_skill_match(self.request.get("skills") or "", self.jobs)
        self.analysis = analysis
        self.events.activity("ok", "Market analysis complete")
        return analysis

    async def _verify(self) -> None:
        """Verify the collected jobs (LLM, bounded); drop unreliable ones."""
        if not self.jobs:
            return

        verified: list[dict] = []
        if self.client is not None:
            try:
                data = await self.client.chat_json(
                    SYSTEM_VERIFY,
                    user_verify_prompt(self.jobs, self.request),
                    temperature=0.1,
                    max_tokens=VERIFY_MAX_TOKENS,
                    validator=verify_validator,
                )
                verified = [j for j in data.get("jobs", []) if j.get("verified")]
            except OpenRouterError:
                self.events.activity("warn", "AI verification unavailable — keeping all collected jobs.")

        if not verified:
            verified = self.jobs

        cleaned = self._dedupe_verified(verified)
        for job in cleaned:
            job["match_score"] = compute_skill_match_score(
                self.request.get("skills") or "", job
            )
        self.verified_jobs = cleaned[:25]
        self.state.jobs_found = len(self.verified_jobs)
        self.events.activity("ok", f"{len(self.verified_jobs)} verified jobs after review")

    async def _finalize(self, analysis: dict) -> None:
        """Build + publish the final report."""
        self.state.current_step = "report"
        self.events.step(self._next_step(), TOTAL_STEPS, "Generating report")
        self.events.activity("action", "Preparing final report...")

        report_md = ""
        if self.client is not None and self.verified_jobs:
            try:
                report_md = await self.client.chat(
                    SYSTEM_REPORT,
                    user_report_prompt(
                        self.request,
                        self.verified_jobs,
                        analysis,
                        getattr(self, "skill_match", {}),
                    ),
                    temperature=0.3,
                    max_tokens=REPORT_MAX_TOKENS,
                )
            except OpenRouterError:
                report_md = ""

        if not report_md:
            report_md = build_report_markdown(
                self.request,
                self.verified_jobs,
                analysis,
                getattr(self, "skill_match", {}),
            )

        if not self.verified_jobs:
            self.state.status = "degraded"
            self.events.activity("warn", "No reliable job listings could be verified.")
        else:
            self.state.status = "completed"
        payload = {
            "request": self.request,
            "jobs": self.verified_jobs,
            "analysis": analysis,
            "skill_match": getattr(self, "skill_match", {}),
            "report_markdown": report_md,
            "jobs_found": len(self.verified_jobs),
            "model": self.client.model if self.client else "unavailable",
            "ai_available": self.client is not None,
            "search_rounds": self.search_rounds,
        }
        self.events.result(payload)
        self.events.activity("ok", "Report generated")

    async def _finalize_fallback(self) -> None:
        """Emit a graceful (possibly empty) result when the loop hit bounds."""
        await self._finalize(getattr(self, "analysis", {}))

    # ------------------------------------------------------------ helpers
    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def _fallback_queries(self) -> list[str]:
        role = self.request["role"]
        loc = self.request["location"]
        exp = self.request["experience"]
        q1 = f"{role} jobs in {loc} {exp} hiring 2026"
        q2 = f"{role} job openings {loc} -apply site:linkedin.com OR site:indeed.com OR site:naukri.com"
        q3 = f"{role} fresher jobs {loc} salary requirements skills"
        if exp.lower() in ("fresher", "entry level", "entry-level", "0-1"):
            q4 = f"entry level {role} remote jobs 2026"
        else:
            q4 = f"{role} {exp} {loc} job vacancy"
        return [q1, q2, q3, q4][:4]

    def _dedupe_into(self, candidates: list[dict]) -> list[dict]:
        """Adds new jobs after de-duping against existing ones by title+company."""
        for job in candidates:
            key = _job_key(job)
            if key and not _contains_key(self.jobs, key):
                self.jobs.append(job)
        return self.jobs

    def _dedupe_verified(self, jobs: list[dict]) -> list[dict]:
        seen: set[tuple] = set()
        out: list[dict] = []
        for job in jobs:
            key = _job_key(job)
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(job)
        return out


TOTAL_STEPS = 7


# ------------------------------------------------------------------ fallbacks


def _block_to_text(block) -> str:
    """Turn a web-search block (dict with title/content/url) into searchable text."""
    if not isinstance(block, dict):
        return str(block).strip()
    title = (block.get("title") or "").strip()
    content = (block.get("content") or "").strip()
    url = (block.get("url") or "").strip()
    if url:
        head = title if title else "Listing"
        return f"{head}\n{content}\nURL: {url}"
    if content:
        return content
    return title


def deterministic_extract(text: str, role: str) -> list[dict]:
    """Deterministic extractor — preserves real URLs when the AI is down.

    Only keeps links that look like actual job pages (job boards, /careers,
    /jobs, /jobs/<id>). Title and company are guessed from nearby text only
    when there is strong signal (e.g. "Job Title:" labels); otherwise the
    requested role is used as the title. No fabricated data.
    """
    jobs: list[dict] = []
    seen_urls: set[str] = set()

    for block in re.split(r"\n\s*---\s*\n+", text):
        block_urls = list(_URL_RE.finditer(block))
        if not block_urls:
            continue
        for m in block_urls:
            url = _clean_url(m.group(0))
            if not url or url in seen_urls:
                continue
            if not _JOB_URL_HINT.search(url):
                continue
            seen_urls.add(url)

            window = block[max(0, m.start() - 500): m.end() + 120]
            title = _guess_title(window, role)
            if title == "Not specified":
                continue
            jobs.append({
                "title": title,
                "company": _guess_company(window),
                "location": _guess_location(window),
                "experience": _guess_experience(window),
                "skills": "Not specified",
                "salary": _guess_salary(window),
                "job_type": _guess_job_type(window),
                "posted_date": _guess_posted(window),
                "url": url,
            })
            if len(jobs) >= 20:
                return jobs
    return jobs


_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")
_JOB_URL_HINT = re.compile(
    r"(linkedin\.com/.+?(jobs|job)|indeed\.com|naukri\.com|glassdoor\.com|"
    r"monster\.|careers?\.|jobs\.|/job|/jobs|/careers|hiring|vacanc)",
    re.IGNORECASE,
)


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?")[:120]


def _guess_title(window: str, role: str) -> str:
    title_match = re.search(
        r"\b(?:job\s*title|position|role)\b\s*[:\-]\s*([A-Za-z][^\n|,]{2,60})",
        window,
        re.IGNORECASE,
    )
    if title_match:
        candidate = _tidy(title_match.group(1))
        if len(candidate) > 2:
            return candidate
    return (role or "Not specified").strip()[:80] or "Not specified"


def _guess_company(window: str) -> str:
    company_match = re.search(
        r"(?<!\S)\b(?:company|organisation|organization)\b\s*[:\-]\s*([A-Z][A-Za-z0-9&.'\- ]{1,50})",
        window,
        re.IGNORECASE,
    )
    if company_match:
        candidate = _tidy(company_match.group(1))
        if candidate and candidate.lower() not in ("the", "and"):
            return candidate[:60]
    at_match = re.search(
        r"(?<!\S)\bat\b\s+([A-Z][A-Za-z0-9&.'\-]{2,40})",
        window,
    )
    if at_match:
        return _tidy(at_match.group(1))[:60]
    return "Not specified"


def _guess_location(window: str) -> str:
    loc_match = re.search(
        r"\blocation\b\s*[:\-]\s*([^,\n;|]{2,50})",
        window,
        re.IGNORECASE,
    )
    if loc_match:
        return _tidy(loc_match.group(1))
    return "Not specified"


def _guess_salary(window: str) -> str:
    direct = re.search(
        r"(?:₹|rs\.?|inr\b)\s*\d[\d,.]*\s*(?:[-–]\s*\d[\d,.]*\s*)?(?:lpa|lakh\w*)?"
        r"|\b\d[\d,.]*\s*[-–]\s*\d[\d,.]*\s*(?:lpa|lakh\w*)\b",
        window,
        re.IGNORECASE,
    )
    if direct:
        return _tidy(direct.group(0))
    label = re.search(
        r"\b(?:salary|pay|compensation|ctc|lpa)\b\s*[:\-]\s*([^\n;|]{2,40})",
        window,
        re.IGNORECASE,
    )
    if label:
        candidate = _tidy(label.group(1))
        if re.search(r"[\d₹$€£]", candidate) or re.search(r"\b(?:lpa|ctc|lakh\w*)\b", candidate, re.IGNORECASE):
            return candidate
    return "Not specified"


def _guess_job_type(window: str) -> str:
    type_match = re.search(
        r"\b(full[- ]?time|part[- ]?time|contract|freelance|internship|temporary|permanent)\b",
        window,
        re.IGNORECASE,
    )
    return type_match.group(1).replace("-", " ").title() if type_match else "Not specified"


def _guess_posted(window: str) -> str:
    posted_match = re.search(
        r"\b(?:published|posted)\s*(?:on\s*)?[:\-]?\s*"
        r"((?:\d{1,2}\s+(?:day|week|month|hour)s?\s+ago)|\b(?:today|yesterday)\b|"
        r"(?:\w+\s+\d{1,2},?\s+\d{2,4})|\b(?:\d{1,2}[-\s/]\w+[-\s/]\d{2,4})\b)",
        window,
        re.IGNORECASE,
    )
    if posted_match:
        return _tidy(posted_match.group(1))
    return "Not specified"


def _guess_experience(window: str) -> str:
    exp_match = re.search(
        r"\b(?:experience|exp\.?)\b\s*[:\-]?\s*([^\n;|]{2,30})",
        window,
        re.IGNORECASE,
    )
    if exp_match:
        candidate = _tidy(exp_match.group(1))
        if re.search(r"fresher|entry|intern|\d|\+|year|yr", candidate, re.IGNORECASE):
            return candidate
    return "Not specified"


def _tidy(text: str) -> str:
    return " ".join(text.strip().split())[:80]


def deterministic_analysis(jobs: list[dict]) -> dict:
    titles: list[str] = []
    companies: list[str] = []
    locations: list[str] = []
    for job in jobs:
        titles.append(str(job.get("title") or "Not specified"))
        companies.append(str(job.get("company") or "Not specified"))
        locations.append(str(job.get("location") or "Not specified"))

    requested_skills: list[str] = ["Not specified"]
    technologies = ["Not specified"]
    fresher_friendly = any(
        str(job.get("experience") or "").lower() in ("fresher", "entry", "0-1", "entry level")
        or "fresh" in str(job.get("experience") or "").lower()
        for job in jobs
    )
    remote = [str(j.get("location")) for j in jobs
              if "remote" in str(j.get("location")).lower()
              or "remote" in str(j.get("job_type")).lower()]

    return {
        "most_requested_skills": requested_skills[:10],
        "most_requested_technologies": technologies[:10],
        "companies_hiring": _top(companies, 10),
        "common_locations": _top(locations, 8),
        "experience_requirements": _top([str(j.get("experience") or "Not specified") for j in jobs], 6),
        "fresher_friendly": fresher_friendly,
        "remote_opportunities": remote[:6],
        "salary_insights": _salary_summary(jobs),
    }


def _salary_summary(jobs: list[dict]) -> str:
    salaries = [str(j.get("salary") or "") for j in jobs if j.get("salary") and str(j.get("salary")) != "Not specified"]
    if not salaries:
        return "Not enough data available."
    ranges = [s for s in salaries if re.search(r"\d", s)]
    if not ranges:
        return "Not enough data available."
    return "; ".join(ranges[:5])


def _top(values: list[str], n: int) -> list[str]:
    from collections import Counter

    counted = Counter(v for v in values if v and v != "Not specified")
    return [v for v, _ in counted.most_common(n)]


def _job_key(job: dict) -> tuple | None:
    title = (str(job.get("title") or "")).strip().lower()
    company = (str(job.get("company") or "")).strip().lower()
    if not title or title == "not specified":
        return None
    return (title, company)


def _contains_key(jobs: list[dict], key: tuple) -> bool:
    for job in jobs:
        if _job_key(job) == key:
            return True
    return False


def build_report_markdown(request: dict, jobs: list[dict], analysis: dict, skill_match: dict) -> str:
    lines: list[str] = []
    lines.append("# JOB FINDER REPORT\n")
    lines.append("### Search\n")
    lines.append(f"Role: {request.get('role') or 'Not specified'}")
    lines.append(f"Location: {request.get('location') or 'Not specified'}")
    lines.append(f"Experience: {request.get('experience') or 'Not specified'}\n")
    lines.append("### Top Job Opportunities\n")
    if not jobs:
        lines.append("No reliable job listings could be verified. Try a broader role, location, or rerun the search.")
    for i, job in enumerate(jobs, start=1):
        lines.append(f"**{i}. {job.get('title') or 'Not specified'}**")
        lines.append(f"- Company: {job.get('company') or 'Not specified'}")
        lines.append(f"- Location: {job.get('location') or 'Not specified'}")
        lines.append(f"- Experience: {job.get('experience') or 'Not specified'}")
        lines.append(f"- Skills: {job.get('skills') or 'Not specified'}")
        lines.append(f"- Salary: {job.get('salary') or 'Not specified'}")
        lines.append(f"- Job type: {job.get('job_type') or 'Not specified'}")
        lines.append(f"- Posted date: {job.get('posted_date') or 'Not specified'}")
        url = job.get("url")
        lines.append(f"- URL: {url if url and url != 'Not specified' else 'Not specified'}\n")
    lines.append("### Market Insights\n")
    lines.append(f"- Most requested skills: {_list_to_na(analysis.get('most_requested_skills'))}")
    lines.append(f"- Most requested technologies: {_list_to_na(analysis.get('most_requested_technologies'))}")
    lines.append(f"- Common experience requirements: {_list_to_na(analysis.get('experience_requirements'))}")
    lines.append(f"- Hiring companies: {_list_to_na(analysis.get('companies_hiring'))}")
    lines.append(f"- Salary information: {analysis.get('salary_insights') or 'Not enough data available.'}\n")
    lines.append("### Your Skill Match\n")
    lines.append(f"- Skills you already have: {_list_to_na(skill_match.get('matching'))}")
    lines.append(f"- Skills commonly requested that you are missing: {_list_to_na(skill_match.get('missing'))}")
    lines.append(f"- Recommended skills to learn: {_list_to_na(skill_match.get('recommended'))}\n")
    lines.append("### Recommendations\n")
    if jobs:
        lines.append(
            f"Prioritize the {min(3, len(jobs))} top-listed opportunities that best match your "
            f"skills and experience. Focus on applying to roles where your current skills overlap "
            f"with the requirements."
        )
        if skill_match.get("recommended"):
            lines.append(
                f"To strengthen your profile, learn: {', '.join(skill_match['recommended'][:5])}."
            )
        else:
            lines.append("No specific skill gaps were detected in the collected listings.")
    else:
        lines.append("No reliable listings were found; broaden the search and try again.")
    return "\n".join(lines)


def _list_to_na(values) -> str:
    if isinstance(values, list) and values:
        return ", ".join(str(v) for v in values[:10])
    return "Not specified"