"""Skill-matching analysis for the Job Finder Agent.

Deterministic, rule-based. Uses normalized keyword matching between the user's
skills and the skills requested by collected jobs. No AI needed here.
"""

from __future__ import annotations

import re
from collections import Counter

_SKILL_ALIASES = {
    "python": {"python", "py"},
    "sql": {"sql", "sql server", "mysql", "postgresql", "postgres", "plsql"},
    "r": {"r", "r lang", "r language", "rstudio"},
    "excel": {"excel", "ms excel", "spreadsheet", "spreadsheets"},
    "power bi": {"power bi", "powerbi"},
    "tableau": {"tableau"},
    "pandas": {"pandas"},
    "numpy": {"numpy"},
    "matplotlib": {"matplotlib"},
    "seaborn": {"seaborn"},
    "scikit-learn": {"scikit-learn", "scikit learn", "sklearn"},
    "tensorflow": {"tensorflow", "tf"},
    "keras": {"keras"},
    "pytorch": {"pytorch"},
    "machine learning": {"machine learning", "ml"},
    "deep learning": {"deep learning", "dl"},
    "nlp": {"nlp", "natural language processing"},
    "data analysis": {"data analysis", "data analytics"},
    "statistics": {"statistics", "statistical analysis", "stats"},
    "data visualization": {"data visualization", "dataviz", "data viz"},
    "aws": {"aws", "amazon web services"},
    "azure": {"azure"},
    "gcp": {"gcp", "google cloud"},
    "docker": {"docker"},
    "kubernetes": {"kubernetes", "k8s"},
    "git": {"git", "github", "bitbucket"},
    "linux": {"linux", "unix"},
    "java": {"java"},
    "javascript": {"javascript", "js"},
    "typescript": {"typescript", "ts"},
    "html": {"html", "css", "html/css"},
    "react": {"react", "reactjs", "react js"},
    "node.js": {"node.js", "nodejs", "node js", "node"},
    "flask": {"flask"},
    "django": {"django"},
    "fastapi": {"fastapi", "fast api"},
    "spark": {"spark", "apache spark", "pyspark"},
    "hadoop": {"hadoop"},
    "kafka": {"kafka"},
    "airflow": {"airflow"},
    "etl": {"etl"},
    "big data": {"big data"},
    "linux server": {"linux"},
    "powerpoint": {"powerpoint", "ppt"},
    "communication": {"communication", "presentation", "presentations", "storytelling"},
    "teamwork": {"teamwork", "collaboration", "collaborative", "team player"},
    "problem solving": {"problem solving", "analytical skills", "analytical"},
    "tableau": {"tableau"},
    "looker": {"looker", "looker studio"},
    "snowflake": {"snowflake"},
    "gcp": {"gcp"},
    "mlops": {"mlops"},
    "power bi": {"power bi"},
    "sap": {"sap"},
    "oracle": {"oracle"},
}

_SKILL_WORDS = {
    "python": {"python", "py"},
    "sql": {"sql", "mysql", "postgres"},
    "r": {"rstudio"},
    "excel": {"excel"},
    "power bi": {"powerbi", "power bi"},
    "tableau": {"tableau"},
    "pandas": {"pandas"},
    "numpy": {"numpy"},
    "scikit-learn": {"sklearn", "scikit"},
    "tensorflow": {"tensorflow"},
    "keras": {"keras"},
    "pytorch": {"pytorch"},
    "machine learning": {"machine learning", "ml"},
    "deep learning": {"deep learning"},
    "nlp": {"nlp"},
    "data analysis": {"data analysis"},
    "statistics": {"statistics"},
    "data visualization": {"data visualization"},
    "aws": {"aws"},
    "azure": {"azure"},
    "gcp": {"google cloud"},
    "docker": {"docker"},
    "kubernetes": {"kubernetes", "k8s"},
    "git": {"git"},
    "linux": {"linux"},
    "java": {"java"},
    "javascript": {"javascript", "js"},
    "typescript": {"typescript"},
    "react": {"react"},
    "node.js": {"nodejs", "node"},
    "flask": {"flask"},
    "django": {"django"},
    "fastapi": {"fastapi"},
    "spark": {"spark"},
    "hadoop": {"hadoop"},
    "kafka": {"kafka"},
    "airflow": {"airflow"},
    "etl": {"etl"},
    "big data": {"big data"},
    "communication": {"communication"},
    "teamwork": {"teamwork", "collaboration"},
    "problem solving": {"problem solving"},
    "looker": {"looker"},
    "snowflake": {"snowflake"},
    "mlops": {"mlops"},
    "sap": {"sap"},
    "oracle": {"oracle"},
}


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def _tokenize_skills(raw: str) -> set[str]:
    norm = normalize(raw)
    found: set[str] = set()
    for canonical, aliases in _SKILL_ALIASES.items():
        for alias in aliases:
            if alias in norm:
                found.add(canonical)
                break
    return found


def _extract_job_skill_keywords(skills_text: str) -> set[str]:
    norm = normalize(skills_text)
    found: set[str] = set()
    for canonical, words in _SKILL_WORDS.items():
        for word in words:
            if word in norm:
                found.add(canonical)
                break
    return found


def compute_skill_match(user_skills_raw: str, jobs: list[dict]) -> dict:
    """Compare user skills against skills requested across the jobs.

    Returns {"matching": [...], "missing": [...], "recommended": [...]}.
    """
    user_skills = _tokenize_skills(user_skills_raw)
    job_skills: Counter[str] = Counter()
    for job in jobs:
        job_skills.update(_extract_job_skill_keywords(job.get("skills") or ""))

    matching = sorted(user_skills & set(job_skills))

    missing_candidates = [s for s, _ in job_skills.most_common() if s not in user_skills]
    missing = missing_candidates[:10]
    recommended = missing[:6]

    return {
        "matching": matching,
        "missing": missing,
        "recommended": recommended,
        "user_skills_count": len(user_skills),
        "job_skill_rank": [{"skill": s, "count": c} for s, c in job_skills.most_common(15)],
    }


def compute_skill_match_score(user_skills_raw: str, job: dict) -> int:
    """0-100 score of how well the user's skills fit a single job."""
    user_skills = _tokenize_skills(user_skills_raw)
    if not user_skills:
        return 0
    job_skills = _extract_job_skill_keywords(job.get("skills") or "")
    if not job_skills:
        return 50
    match_count = len(user_skills & job_skills)
    return round(100 * min(1.0, match_count / max(len(job_skills), 1)))