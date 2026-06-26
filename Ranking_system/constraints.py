from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any
from document_builder import load_candidates

# Refernece terms required for varipus checks : 

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra", "mindtree",
    "ltimindtree", "l&t infotech", "mphasis", "ibm global services",
}

CORE_RETRIEVAL_TECH = {
    "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "vector database", "vector search",
    "ann", "approximate nearest neighbor", "hybrid search", "embeddings",
}

EVAL_FRAMEWORK_TECH = {
    "ndcg", "mrr", "map", "a/b test", "offline-to-online", "offline evaluation",
    "ranking evaluation", "learning to rank", "ltr",
}

LLM_SURFACE_ONLY_MARKERS = {
    "langchain", "chatgpt", "openai api", "llamaindex", "prompt engineering",
}

CV_SPEECH_ROBOTICS_ONLY = {
    "image classification", "computer vision", "speech recognition", "tts",
    "robotics", "gans", "object detection", "ocr",
}

NLP_IR_MARKERS = {
    "nlp", "natural language processing", "information retrieval", "search",
    "ranking", "recommendation", "retrieval", "embeddings", "llm",
    "fine-tuning llms", "lora",
}

PREFERRED_LOCATIONS = {"noida", "pune"}
ACCEPTABLE_LOCATIONS = {"noida", "pune", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram", "new delhi"}


def _text_blob(candidate: dict) -> str:
    """Flatten everything searchable into one lowercase string for keyword checks."""
    parts = [candidate.get("profile", {}).get("summary", ""),
             candidate.get("profile", {}).get("headline", "")]
    for role in candidate.get("career_history", []) or []:
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    for s in candidate.get("skills", []) or []:
        parts.append(s.get("name", ""))
    return " ".join(parts).lower()


def _months_since(date_str: str | None, ref: date) -> float | None:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (ref.year - d.year) * 12 + (ref.month - d.month)


def _is_consulting_only_career(candidate: dict) -> bool:
    companies = [r.get("company", "").strip().lower() for r in candidate.get("career_history", []) or []]
    if not companies:
        return False
    return all(c in CONSULTING_FIRMS for c in companies)


def _has_prior_product_company_experience(candidate: dict) -> bool:
    companies = [r.get("company", "").strip().lower() for r in candidate.get("career_history", []) or []]
    return any(c not in CONSULTING_FIRMS for c in companies)


def _job_hopping_titles_chasing(candidate: dict) -> bool:
    """
    job hopping and title chasing flag
    """
    history = candidate.get("career_history", []) or []
    if len(history) < 3:
        return False
    durations = [r.get("duration_months", 0) or 0 for r in history]
    avg_tenure = sum(durations) / len(durations)
    seniority_words = ["junior", "engineer", "senior", "staff", "lead", "principal", "head", "director"]

    def seniority_rank(title: str) -> int:
        title_l = (title or "").lower()
        for i, w in enumerate(seniority_words):
            if w in title_l:
                return i
        return -1

    ranks = [seniority_rank(r.get("title", "")) for r in history]
    ranks = [r for r in ranks if r >= 0]
    escalating = len(ranks) >= 2 and ranks == sorted(ranks) and ranks[-1] > ranks[0]
    return avg_tenure <= 18 and escalating


def _cv_speech_robotics_dominant_without_nlp(candidate: dict) -> bool:
    blob = _text_blob(candidate)
    cv_hits = sum(1 for kw in CV_SPEECH_ROBOTICS_ONLY if kw in blob)
    nlp_hits = sum(1 for kw in NLP_IR_MARKERS if kw in blob)
    return cv_hits >= 3 and nlp_hits == 0


def _recent_langchain_only(candidate: dict, ref: date) -> bool:
    """
    recent AI hype frameworks flag
    """
    blob_recent, blob_older = "", ""
    for role in candidate.get("career_history", []) or []:
        months_ago = _months_since(role.get("start_date"), ref)
        text = f"{role.get('title','')} {role.get('description','')}".lower()
        if months_ago is not None and months_ago <= 12:
            blob_recent += text
        else:
            blob_older += text

    recent_is_langchain = any(kw in blob_recent for kw in LLM_SURFACE_ONLY_MARKERS)
    older_has_real_ml = any(kw in blob_older for kw in CORE_RETRIEVAL_TECH | EVAL_FRAMEWORK_TECH | NLP_IR_MARKERS)
    has_any_older_role = bool(blob_older.strip())
    return recent_is_langchain and not older_has_real_ml and not has_any_older_role


def _stale_architect_not_coding(candidate: dict, ref: date) -> bool:
    """
    Senior title and no production code in the most recent role
    """
    history = candidate.get("career_history", []) or []
    if not history:
        return False
    current = next((r for r in history if r.get("is_current")), history[0])
    title = (current.get("title") or "").lower()
    desc = (current.get("description") or "").lower()
    months_in_role = current.get("duration_months", 0) or 0
    leadership_words = ["architecture", "tech lead", "engineering manager", "head of", "director"]
    coding_words = ["implemented", "built", "wrote", "coded", "shipped", "developed", "designed and built"]
    is_leadership_title = any(w in title for w in leadership_words)
    has_coding_evidence = any(w in desc for w in coding_words)
    return is_leadership_title and not has_coding_evidence and months_in_role >= 18


def _pure_research_no_production(candidate: dict) -> bool:
    blob = _text_blob(candidate)
    research_markers = ["research scientist", "research lab", "academic", "phd", "publication", "postdoc"]
    production_markers = ["production", "deployed", "shipped", "real users", "live system", "scale"]
    has_research = any(m in blob for m in research_markers)
    has_production = any(m in blob for m in production_markers)
    return has_research and not has_production


class ConstraintResult:
    candidate_id: str
    disqualified: bool
    disqualifier_reasons: list[str] = field(default_factory=list)
    soft_score: float = 0.0           # 0-1, blended with embedding similarity downstream
    soft_score_breakdown: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)   # informational, non-disqualifying notes


def evaluate_candidate(candidate: dict, reference_date: date | None = None) -> ConstraintResult:
    ref = reference_date or date.today()
    cid = candidate.get("candidate_id", "unknown")
    blob = _text_blob(candidate)
    signals = candidate.get("redrob_signals", {}) or {}
    profile = candidate.get("profile", {}) or {}

    reasons: list[str] = []
    flags: list[str] = []

    # flags 
    if _pure_research_no_production(candidate):
        reasons.append("Pure research background with no detectable production deployment evidence.")

    if _recent_langchain_only(candidate, ref):
        reasons.append("AI exposure looks limited to recent (<12mo) LangChain/OpenAI-API usage with no "
                        "pre-LLM-era production ML evidence.")

    if _stale_architect_not_coding(candidate, ref):
        reasons.append("Current role reads as architecture/tech-lead without recent hands-on coding evidence.")

    if _is_consulting_only_career(candidate):
        reasons.append("Entire visible career history is at consulting/services firms "
                        "(TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini-class) with no product-company experience.")

    if _cv_speech_robotics_dominant_without_nlp(candidate):
        reasons.append("Primary expertise reads as computer vision/speech/robotics with no NLP/IR exposure.")

    if _job_hopping_titles_chasing(candidate):
        reasons.append("Career pattern reads as title-chasing: frequent switches (~<=18mo avg tenure) with "
                        "escalating seniority titles.")

    # visa or location flag
    country = (profile.get("country") or "").strip().lower()
    if country and country != "india":
        if not signals.get("willing_to_relocate", False):
            reasons.append(f"Located outside India ({profile.get('country')}) with no visa sponsorship available "
                            f"and not flagged as willing to relocate.")
        else:
            flags.append("Outside India but willing to relocate — case-by-case per JD, not auto-disqualified.")

    disqualified = len(reasons) > 0

    # soft scoring
    breakdown: dict[str, float] = {}

    #Core retrieval/vector-DB production evidence
    retrieval_hits = sum(1 for kw in CORE_RETRIEVAL_TECH if kw in blob)
    breakdown["retrieval_vectordb_evidence"] = min(retrieval_hits / 4, 1.0) * 0.30

    # Evaluation framework rigor
    eval_hits = sum(1 for kw in EVAL_FRAMEWORK_TECH if kw in blob)
    breakdown["eval_framework_evidence"] = min(eval_hits / 2, 1.0) * 0.15

    # Experience band fit 5-9 yrs as given
    yoe = profile.get("years_of_experience", 0) or 0
    if 5 <= yoe <= 9:
        exp_score = 1.0
    elif yoe < 5:
        exp_score = max(0.0, 1 - (5 - yoe) / 5)
    else:
        exp_score = max(0.0, 1 - (yoe - 9) / 8)
    breakdown["experience_band_fit"] = exp_score * 0.10

    # consulting flag
    breakdown["product_company_experience"] = (0.10 if _has_prior_product_company_experience(candidate) else 0.0)

    # add on skills flag
    nice_to_have_kws = {"lora", "qlora", "peft", "xgboost", "learning to rank",
                         "recruiting", "marketplace", "open-source", "open source", "github"}
    nice_hits = sum(1 for kw in nice_to_have_kws if kw in blob)
    breakdown["nice_to_have_signals"] = min(nice_hits / 3, 1.0) * 0.10

    # Location 'fit' parameter
    loc = (profile.get("location") or "").lower()
    if any(p in loc for p in PREFERRED_LOCATIONS):
        loc_score = 1.0
    elif any(p in loc for p in ACCEPTABLE_LOCATIONS):
        loc_score = 0.6
    elif signals.get("willing_to_relocate"):
        loc_score = 0.4
    else:
        loc_score = 0.1
    breakdown["location_fit"] = loc_score * 0.05

    # Behavioral flag
    last_active = signals.get("last_active_date")
    months_inactive = _months_since(last_active, ref) if last_active else None
    if months_inactive is None:
        activity_score = 0.3
    else:
        activity_score = max(0.0, 1 - months_inactive / 6)  # linear decay over 6 months
    response_rate = signals.get("recruiter_response_rate")
    response_score = response_rate if isinstance(response_rate, (int, float)) and response_rate >= 0 else 0.2
    notice = signals.get("notice_period_days")
    if isinstance(notice, (int, float)):
        notice_score = 1.0 if notice <= 30 else max(0.0, 1 - (notice - 30) / 60)
    else:
        notice_score = 0.5
    availability_blend = (activity_score * 0.5 + response_score * 0.3 + notice_score * 0.2)
    breakdown["behavioral_availability"] = availability_blend * 0.20

    soft_score = sum(breakdown.values())

    if months_inactive is not None and months_inactive >= 6:
        flags.append(f"Inactive on platform for ~{months_inactive:.0f} months — effectively unreachable; "
                      f"down-weighted regardless of profile quality.")
    if isinstance(response_rate, (int, float)) and response_rate < 0.15:
        flags.append("Very low recruiter response rate — low practical availability.")

    return ConstraintResult(
        candidate_id=cid,
        disqualified=disqualified,
        disqualifier_reasons=reasons,
        soft_score=round(soft_score, 4),
        soft_score_breakdown={k: round(v, 4) for k, v in breakdown.items()},
        flags=flags,
    )


if __name__ == "__main__":
    import json

    candidates = load_candidates("candidates.jsonl")
   