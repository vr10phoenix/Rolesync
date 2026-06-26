import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from __future__ import annotations


class CandidateDocument:
    """Container about one candidate."""
    candidate_id: str
    text: str                     
    metadata: dict[str, Any] = field(default_factory=dict)  
    raw: dict[str, Any] = field(default_factory=dict)        


def _fmt_date(date_str: str | None) -> str:
    if not date_str:
        return "present"
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %Y")
    except ValueError:
        return date_str


def _months_to_years(months: int | float | None) -> str:
    if not months:
        return "unknown duration"
    years = months / 12
    return f"{years:.1f} years" if years >= 1 else f"{int(months)} months"


def _build_career_section(career_history: list[dict]) -> str:
    """
    Builds the career section of the document
    """
    if not career_history:
        return "CAREER HISTORY: none provided."

    blocks = []
    for i, role in enumerate(career_history):
        title = role.get("title", "Unknown title")
        company = role.get("company", "Unknown company")
        industry = role.get("industry", "unknown industry")
        size = role.get("company_size", "unknown size")
        start = _fmt_date(role.get("start_date"))
        end = "present" if role.get("is_current") else _fmt_date(role.get("end_date"))
        dur = _months_to_years(role.get("duration_months"))
        desc = (role.get("description") or "").strip()
        current_tag = " (current role)" if role.get("is_current") else ""

        block = (
            f"ROLE {i + 1}: {title} at {company}{current_tag}, "
            f"a {size}-employee company in the {industry} industry. "
            f"Tenure: {start} to {end} ({dur}). "
            f"Description: {desc}"
        )
        blocks.append(block)

    return "CAREER HISTORY:\n" + "\n\n".join(blocks)


def _build_skills_section(skills: list[dict]) -> str:
    # function that adds logical texts on skills
    if not skills:
        return "SKILLS: none listed."
    ranked = sorted(
        skills,
        key=lambda s: (s.get("endorsements", 0) or 0) + (s.get("duration_months", 0) or 0) / 3,
        reverse=True,
    )
    lines = []
    for s in ranked:
        lines.append(
            f"{s.get('name', 'Unknown')} ({s.get('proficiency', 'unknown')} level, "
            f"{s.get('duration_months', 0)} months experience, "
            f"{s.get('endorsements', 0)} endorsements)"
        )
    return "SKILLS (ranked by evidenced depth, not just self-reported list)\:\n" + "; ".join(lines)


def _build_education_section(education: list[dict]) -> str:
    if not education:
        return "EDUCATION: none provided."
    lines = []
    for e in education:
        lines.append(
            f"{e.get('degree', '')} in {e.get('field_of_study', '')} from "
            f"{e.get('institution', 'unknown institution')} "
            f"({e.get('start_year', '?')}-{e.get('end_year', '?')}, "
            f"grade: {e.get('grade', 'n/a')}, tier: {e.get('tier', 'unknown')})"
        )
    return "EDUCATION:\n" + "; ".join(lines)


def _build_signals_section(signals: dict) -> str:
    """
    Redrob signal filter section
    """
    if not signals:
        return "ACTIVITY/AVAILABILITY SIGNALS: none provided."

    last_active = signals.get("last_active_date", "unknown")
    open_to_work = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", None)
    notice = signals.get("notice_period_days", "unknown")
    relocate = signals.get("willing_to_relocate", False)
    work_mode = signals.get("preferred_work_mode", "unspecified")
    salary = signals.get("expected_salary_range_inr_lpa", {}) or {}
    github = signals.get("github_activity_score", None)

    parts = [
        f"Last active on platform: {last_active}.",
        f"Open to work: {open_to_work}.",
        f"Recruiter response rate: {response_rate}." if response_rate is not None else "",
        f"Notice period: {notice} days.",
        f"Willing to relocate: {relocate}. Preferred work mode: {work_mode}.",
        f"Expected salary band (INR LPA): {salary.get('min', 'n/a')}-{salary.get('max', 'n/a')}.",
        f"GitHub activity score: {github}." if github is not None and github >= 0 else "No verifiable GitHub activity.",
    ]
    return "ACTIVITY/AVAILABILITY SIGNALS:\n" + " ".join(p for p in parts if p)


def build_candidate_text(candidate: dict) -> str:
    """
    Assemble the full text document for one candidate.
    """
    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []
    certifications = candidate.get("certifications", []) or []
    languages = candidate.get("languages", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    header = (
        f"CANDIDATE PROFILE\n"
        f"Name: {profile.get('anonymized_name', 'Unknown')}\n"
        f"Headline: {profile.get('headline', 'n/a')}\n"
        f"Current role: {profile.get('current_title', 'n/a')} at "
        f"{profile.get('current_company', 'n/a')} "
        f"({profile.get('current_industry', 'n/a')}, "
        f"{profile.get('current_company_size', 'n/a')} employees)\n"
        f"Location: {profile.get('location', 'n/a')}, {profile.get('country', 'n/a')}\n"
        f"Total experience: {profile.get('years_of_experience', 'n/a')} years\n"
        f"Summary: {(profile.get('summary') or '').strip()}"
    )

    sections = [
        header,
        _build_career_section(career_history),
        _build_skills_section(skills),
        _build_education_section(education),
        # "CERTIFICATIONS: " + (", ".join(certifications) if certifications else "none."),
        "CERTIFICATIONS: " + (", ".join(c.get('name', 'Unknown Cert') for c in certifications) if certifications else "none."),
        "LANGUAGES: " + (", ".join(f"{l.get('language')} ({l.get('proficiency')})" for l in languages) if languages else "none."),
        _build_signals_section(signals),
    ]

    return "\n\n".join(sections)


def build_candidate_document(candidate: dict) -> CandidateDocument:
    """
    main function that sequentially builds the Candidate profile with comments and 
     logical text.
    """
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []

    text = build_candidate_text(candidate)

    metadata = {
        "candidate_id": candidate.get("candidate_id"),
        "name": profile.get("anonymized_name"),
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "current_industry": profile.get("current_industry"),
        "years_of_experience": profile.get("years_of_experience"),
        "location": profile.get("location"),
        "country": profile.get("country"),
        "companies": [r.get("company") for r in career_history],
        "titles": [r.get("title") for r in career_history],
        "industries": [r.get("industry") for r in career_history],
        "skill_names": [s.get("name") for s in skills],
        "last_active_date": signals.get("last_active_date"),
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "notice_period_days": signals.get("notice_period_days"),
        "willing_to_relocate": signals.get("willing_to_relocate"),
        "preferred_work_mode": signals.get("preferred_work_mode"),
        "expected_salary_range_inr_lpa": signals.get("expected_salary_range_inr_lpa"),
        "github_activity_score": signals.get("github_activity_score"),
    }

    return CandidateDocument(
        candidate_id=candidate.get("candidate_id"),
        text=text,
        metadata=metadata,
        raw=candidate,
    )


def load_candidates(path: str) -> list[dict]:
    """Loads either a JSON array of candidates or a JSONL file (one object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":

    candidates = load_candidates("sample_candidates.json")