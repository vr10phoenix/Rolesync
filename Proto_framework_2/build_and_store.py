import json
import os
from datetime import datetime
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_synthetic_document(candidate: dict) -> str:
    """
    Transforms the structured JSON schema into a dense, semantically rich
    text document optimized for Bi-Encoder and Cross-Encoder ranking.
    considering trap-avoidance logic for the Redrob JD.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    doc_parts = []

    # Title Mismatch" 
    current_title = profile.get('current_title', 'Unknown').lower()
    if any(non_tech in current_title for non_tech in ['marketing', 'hr', 'sales', 'recruiter', 'analyst']):
        doc_parts.append(f"CURRENT REALITY WARNING: Candidate's current title is '{profile.get('current_title')}'. "
                         f"High risk of being a non-engineering profile despite listed technical skills.")
    else:
        doc_parts.append(f"Candidate Title: {profile.get('current_title')} at {profile.get('current_company')}")

    doc_parts.append(f"Headline: {profile.get('headline')}")
    doc_parts.append(f"Total Experience: {profile.get('years_of_experience')} years in the {profile.get('current_industry')} industry.")
    doc_parts.append(f"Summary: {profile.get('summary')}")

    # Risk factors 
    doc_parts.append("\nRISK FACTORS & DISQUALIFIERS : ")
    risks = []

    # A. The Ghost Profile Trap
    last_active = signals.get("last_active_date")
    response_rate = signals.get("recruiter_response_rate", 1.0)

    if last_active:
        try:
            last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (datetime.now() - last_active_dt).days
            if days_inactive > 180 or response_rate < 0.10:
                risks.append(f"WARNING: Candidate is a 'Ghost Profile'. Inactive for {days_inactive} days "
                             f"with a {response_rate*100}% response rate. Do not shortlist.")
        except ValueError:
            pass

    # title chaser 
    if history and len(history) >= 3:
        avg_tenure = sum(role.get("duration_months", 0) for role in history) / len(history)
        if avg_tenure < 18:
            risks.append("WARNING: Demonstrates chronic job-hopping and title-chasing behavior. "
                         f"Average tenure is only {avg_tenure:.1f} months across recent roles.")

    # pure Consulting or research 
    if history:
        consulting_flags = ["it services", "consulting", "tcs", "infosys", "wipro", "accenture", "cognizant"]
        research_flags = ["academic", "research", "university", "lab"]

        all_consulting = all(any(c in role.get("industry", "").lower() or c in role.get("company", "").lower() for c in consulting_flags) for role in history)
        all_research = all(any(r in role.get("industry", "").lower() or r in role.get("company", "").lower() for r in research_flags) for role in history)

        if all_consulting:
            risks.append("WARNING: Career is entirely based in IT Services/Consulting with no demonstrable product-company experience.")
        if all_research:
            risks.append("WARNING: Pure research background. Lacks evidence of deploying models to production users.")

    # architect
    if history:
        recent_role = history[0]
        recent_title = recent_role.get("title", "").lower()
        recent_desc = recent_role.get("description", "").lower()
        if any(kw in recent_title for kw in ["architect", "tech lead", "manager", "director", "vp"]):
            if not any(kw in recent_desc for kw in ["code", "shipped", "built", "implemented", "wrote", "hands-on"]):
                 risks.append("WARNING: Moved into pure architecture/management. No explicit evidence of writing production code in recent 18 months.")

    if risks:
        doc_parts.extend(risks)
    else:
        doc_parts.append("No immediate disqualifying risk factors detected in career trajectory.")

    ##career growth
    doc_parts.append("\nCAREER TRAJECTORY & EXPERIENCE :")
    if history:
        for role in history:
            title = role.get("title", "Unknown Role")
            company = role.get("company", "Unknown Company")
            duration = role.get("duration_months", 0)
            desc = role.get("description", "")

            doc_parts.append(f"Worked as {title} at {company} for {duration} months.")
            doc_parts.append(f"Impact & Responsibilities: {desc}")

            if any(kw in desc.lower() for kw in ["ranking", "retrieval", "search", "recommendation", "matching"]):
                doc_parts.append("HIDDEN MATCH SIGNAL: Career history explicitly shows building ranking, search, or recommendation systems at scale.")

    # 3. framework trap
    doc_parts.append("\nVERIFIED SKILLS : ")
    foundational_skills = []
    framework_skills = []
    ai_hype_frameworks = ["langchain", "openai", "chatgpt", "llamaindex", "huggingface", "pinecone"]

    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "unknown")
        months = skill.get("duration_months", 0)
        skill_str = f"{prof.capitalize()} in {name} ({months} months)"

        if any(hype in name.lower() for hype in ai_hype_frameworks):
            framework_skills.append(skill_str)
        else:
            foundational_skills.append(skill_str)

    if foundational_skills:
        doc_parts.append("Core Engineering Foundation: " + ", ".join(foundational_skills))
    if framework_skills:
        doc_parts.append("Recent AI Frameworks: " + ", ".join(framework_skills))

    # framework Check
    if framework_skills and not any(skill.get("duration_months", 0) >= 24 for skill in skills if skill.get("name", "").lower() in ["python", "sql", "machine learning", "backend"]):
        doc_parts.append("WARNING: Framework Enthusiast. AI experience consists primarily of recent API wrappers without deep foundational engineering or ML deployment.")


    # 4. Behavioural aspects 
    doc_parts.append("\nBEHAVIORAL INTELLIGENCE : ")

    # communication and responsiveness
    if response_rate > 0.8:
        doc_parts.append("Highly responsive and active communicator.")
    elif response_rate < 0.3:
        doc_parts.append("Passive candidate; extremely slow communication style.")

    # reliability
    interview_rate = signals.get("interview_completion_rate", 0)
    if interview_rate > 0.85:
        doc_parts.append("Exceptional follow-through and high reliability in professional commitments.")
    elif interview_rate < 0.50:
        doc_parts.append("WARNING: Severe flake risk. Poor history of attending scheduled interviews.")

    # passion
    github_score = signals.get("github_activity_score", -1)
    if github_score > 70:
        doc_parts.append("Deeply passionate about technology with a highly active GitHub footprint and open-source contributions.")

    # platform verified assessments
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        top_skills = [k for k, v in assessments.items() if v > 80]
        if top_skills:
            doc_parts.append(f"Platform-verified top performer in: {', '.join(top_skills)}.")

    # concatenate
    return "\n".join(doc_parts)

def build_faiss_index(jsonl_filepath: str):
    print("Loading candidate dataset and generating synthetic documents...")
    langchain_documents = []

    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue

                candidate_data = json.loads(line)
                candidate_id = candidate_data.get("candidate_id", "UNKNOWN")

                synthetic_text = build_synthetic_document(candidate_data)
                doc = Document(
                    page_content=synthetic_text,
                    metadata={"candidate_id": candidate_id}
                )
                langchain_documents.append(doc)
    except FileNotFoundError:
        print(f"Error: Could not find '{jsonl_filepath}'. Make sure the file exists in this directory.")
        return

    print(f"Successfully synthesized {len(langchain_documents)} candidate profiles.")
    print("Initializing Local HuggingFace Model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("embedding canidates into local FAISS index ...")
    vector_db = FAISS.from_documents(langchain_documents, embeddings)
    print("Saving FAISS index locally...")
    vector_db.save_local("faiss_candidate_index_modified")
    print("FAISS Index created successfully")

if __name__ == "__main__":
    #initiator
    build_faiss_index("candidates.jsonl")