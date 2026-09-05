"""Prompt templates for the Job Finder Agent."""

from __future__ import annotations

# ------------------------------------------------------------------ search plan

SYSTEM_SEARCH_PLAN = """You are the planning module of a Job Finder Agent.
You receive a user's job-search request and must produce a structured search
plan: a list of focused web-search queries that will surface current job
postings matching the request.

Rules:
- Return ONLY valid JSON, no prose, no markdown fences.
- Produce 3 to 5 distinct queries.
- Queries must be realistic web-search strings (e.g. "Data Analyst jobs in
  India for freshers"). Include relevant sites like LinkedIn, Indeed, Naukri
  when appropriate.
- Never invent job postings here. This step only plans searches."""

SYSTEM_JOB_EXTRACTION = """You are the analysis module of a Job Finder Agent.
You receive the raw text output of a web search for job postings. Extract real
job opportunities into structured JSON.

Rules:
- Return ONLY valid JSON, no prose, no markdown fences.
- Structure: {"jobs": [ {...} ]}
- Each job object MUST have exactly these string fields:
  title, company, location, experience, skills, salary, job_type, posted_date, url
- Use "Not specified" for any field that is not present in the search results.
- NEVER invent, guess, or fabricate any information. If a salary is not
  mentioned, salary must be "Not specified". If a URL is not present, url must
  be "Not specified".
- Only include jobs that genuinely appear in the search output.
- Skills should be a comma-separated string of technologies/tools when
  mentioned."""

SYSTEM_ANALYSIS = """You are the market-analysis module of a Job Finder Agent.
You receive a list of confirmed job listings. Produce actionable market
insights in structured JSON.

Return ONLY valid JSON with this exact shape:
{
  "most_requested_skills": ["...", "..."],
  "most_requested_technologies": ["...", "..."],
  "companies_hiring": ["...", "..."],
  "common_locations": ["...", "..."],
  "experience_requirements": ["...", "..."],
  "fresher_friendly": true,
  "remote_opportunities": ["...", "..."],
  "salary_insights": "..."
}
- Base every claim on the provided job data. Do not extrapolate.
- If a dimension cannot be analyzed (e.g. no salary data), use an empty list
  or "Not enough data available."
- salary_insights must summarize salary ranges ONLY when supported by the data;
  otherwise say "Not enough data available." Never invent salary figures."""

SYSTEM_VERIFY = """You are the verification module of a Job Finder Agent.
You receive a list of extracted job listings and the user's original request.

Check each job for:
1. Relevance to the user's role, location, and experience level.
2. Suspicious or impossible data (e.g. no title, missing URL).
3. Internal inconsistencies (title/company mismatches, unsupported salaries).
4. Duplicates (same title + same company).

Return ONLY valid JSON with this exact shape:
{
  "jobs": [ { ...same job objects, with an added field "verified": true/false and "reason": "..." } ],
  "needs_more_search": false,
  "reason": "..."
}
Set needs_more_search=true ONLY when too few reliable jobs remain (fewer than
3) and a further search could plausibly find more. Preserve every original
field; only add "verified" and "reason"."""

SYSTEM_REPORT = """You are the report module of a Job Finder Agent.
You receive verified job listings plus analysis plus the user's skills.

Produce a final markdown job report. Structure it exactly like this:

# JOB FINDER REPORT

### Search

Role: ...
Location: ...
Experience: ...

### Top Job Opportunities

For each job, a numbered block with:
- Job title
- Company
- Location
- Experience
- Skills
- Salary
- Job type
- Posted date
- URL

Use "Not specified" for missing values. Never invent salaries or URLs.

### Market Insights

- Most requested skills
- Most requested technologies
- Common experience requirements
- Hiring companies
- Salary information

### Your Skill Match

- Skills you already have (from the user's skills that match job requirements)
- Skills commonly requested that you are missing
- Recommended skills to learn

### Recommendations

Practical advice: which jobs to prioritize and what the user should learn next.

Return ONLY the markdown report, nothing else."""

# ------------------------------------------------------------------ validators


def search_plan_validator(data: dict) -> dict:
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("search plan must contain a non-empty 'queries' list")
    cleaned = [str(q).strip() for q in queries if str(q).strip()]
    if not cleaned:
        raise ValueError("search plan queries are empty")
    return {"queries": cleaned[:6]}


def job_extraction_validator(data: dict) -> dict:
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("job extraction must contain a 'jobs' list")
    cleaned = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        row = {
            "title": _clean(job.get("title")),
            "company": _clean(job.get("company")),
            "location": _clean(job.get("location")),
            "experience": _clean(job.get("experience")),
            "skills": _clean(job.get("skills")),
            "salary": _clean(job.get("salary")),
            "job_type": _clean(job.get("job_type")),
            "posted_date": _clean(job.get("posted_date")),
            "url": _clean(job.get("url")),
        }
        if row["title"] and row["title"] != "Not specified":
            cleaned.append(row)
    if not cleaned:
        raise ValueError("job extraction returned no valid jobs")
    return {"jobs": cleaned}


def analysis_validator(data: dict) -> dict:
    for key in (
        "most_requested_skills",
        "most_requested_technologies",
        "companies_hiring",
        "common_locations",
        "experience_requirements",
        "remote_opportunities",
    ):
        value = data.get(key)
        if not isinstance(value, list):
            data[key] = []
        else:
            data[key] = [str(v) for v in value]
    data["fresher_friendly"] = bool(data.get("fresher_friendly"))
    if not isinstance(data.get("salary_insights"), str):
        data["salary_insights"] = "Not enough data available."
    return data


def verify_validator(data: dict) -> dict:
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("verification must contain a 'jobs' list")
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if "verified" not in job:
            job["verified"] = True
        if "reason" not in job:
            job["reason"] = ""
        for key in (
            "title", "company", "location", "experience",
            "skills", "salary", "job_type", "posted_date", "url",
        ):
            job[key] = _clean(job.get(key))
    data["needs_more_search"] = bool(data.get("needs_more_search"))
    if not isinstance(data.get("reason"), str):
        data["reason"] = ""
    return data


# ------------------------------------------------------------------ helpers


def _clean(value) -> str:
    if value is None:
        return "Not specified"
    text = str(value).strip()
    if not text or text.lower() in ("none", "null", "n/a", "na"):
        return "Not specified"
    return text


# ------------------------------------------------------------------ user prompts


def user_search_plan_prompt(request: dict) -> str:
    return (
        "User job-search request:\n"
        f"- Role: {request.get('role')}\n"
        f"- Location: {request.get('location')}\n"
        f"- Experience: {request.get('experience')}\n"
        f"- Skills: {request.get('skills')}\n"
        f"- Salary preference: {request.get('salary')}\n"
        f"- Work mode: {request.get('work_mode')}\n\n"
        "Create a focused web-search plan (3-5 queries) that will find current "
        "job postings matching this request."
    )


def user_job_extraction_prompt(query: str, search_output: str) -> str:
    return (
        f"Web search performed for: {query}\n\n"
        f"Search results text:\n\n{search_output[:12000]}\n\n"
        "Extract all real job postings found in the results into the jobs "
        "array. Use 'Not specified' for fields that are not present."
    )


def user_analysis_prompt(jobs: list[dict]) -> str:
    lines = [f"{i + 1}. {_job_line(job)}" for i, job in enumerate(jobs)]
    return (
        "Collected job listings:\n\n" + "\n".join(lines) + "\n\n"
        "Analyze these jobs and produce market insights based only on this data."
    )


def user_verify_prompt(jobs: list[dict], request: dict) -> str:
    lines = [f"{i + 1}. {_job_line(job)} (url: {job.get('url')})" for i, job in enumerate(jobs)]
    return (
        f"User request: {request.get('role')} | {request.get('location')} | "
        f"{request.get('experience')}\n\n"
        "Job listings to verify:\n\n" + "\n".join(lines) + "\n\n"
        "Flag unverifiable, irrelevant, duplicated, or suspicious listings. "
        "Keep only jobs that are plausibly real and relevant."
    )


def user_report_prompt(request: dict, jobs: list[dict], analysis: dict, skill_match: dict) -> str:
    job_lines = [f"{i + 1}. {_job_line(job)}" for i, job in enumerate(jobs)]
    analysis_lines = _flatten_analysis(analysis)
    skill_lines = _flatten_skill_match(skill_match)
    return (
        "Original request:\n"
        f"- Role: {request.get('role')}\n"
        f"- Location: {request.get('location')}\n"
        f"- Experience: {request.get('experience')}\n"
        f"- User skills: {request.get('skills')}\n"
        f"- Salary preference: {request.get('salary')}\n"
        f"- Work mode: {request.get('work_mode')}\n\n"
        "Verified jobs:\n\n" + "\n".join(job_lines) + "\n\n"
        "Market analysis:\n\n" + analysis_lines + "\n\n"
        "Skill match:\n\n" + skill_lines + "\n\n"
        "Produce the final markdown job report."
    )


# ------------------------------------------------------------------ helpers


def _job_line(job: dict) -> str:
    skills = job.get("skills") or "Not specified"
    return (
        f"Title: {job.get('title')} | Company: {job.get('company')} | "
        f"Location: {job.get('location')} | Experience: {job.get('experience')} | "
        f"Skills: {skills} | Salary: {job.get('salary')} | "
        f"Type: {job.get('job_type')} | Posted: {job.get('posted_date')}"
    )


def _flatten_analysis(analysis: dict) -> str:
    if not analysis:
        return "No analysis available."
    parts = []
    if analysis.get("most_requested_skills"):
        parts.append("Most requested skills: " + ", ".join(analysis["most_requested_skills"]))
    if analysis.get("most_requested_technologies"):
        parts.append("Most requested technologies: " + ", ".join(analysis["most_requested_technologies"]))
    if analysis.get("companies_hiring"):
        parts.append("Companies hiring: " + ", ".join(analysis["companies_hiring"]))
    if analysis.get("common_locations"):
        parts.append("Common locations: " + ", ".join(analysis["common_locations"]))
    if analysis.get("experience_requirements"):
        parts.append("Experience requirements: " + ", ".join(analysis["experience_requirements"]))
    if analysis.get("remote_opportunities"):
        parts.append("Remote opportunities: " + ", ".join(analysis["remote_opportunities"]))
    parts.append(f"Fresher friendly: {bool(analysis.get('fresher_friendly'))}")
    parts.append(f"Salary insights: {analysis.get('salary_insights') or 'Not enough data available.'}")
    return "\n".join(parts)


def _flatten_skill_match(skill_match: dict) -> str:
    if not skill_match:
        return "No skill match available."
    parts = []
    if skill_match.get("matching"):
        parts.append("Matching skills: " + ", ".join(skill_match["matching"]))
    if skill_match.get("missing"):
        parts.append("Missing skills: " + ", ".join(skill_match["missing"]))
    if skill_match.get("recommended"):
        parts.append("Recommended to learn: " + ", ".join(skill_match["recommended"]))
    return "\n".join(parts) if parts else "No skill match available."