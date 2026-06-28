import json
from datetime import datetime

def build_synthetic_document_1(candidate: dict) -> str:
    """
    updates synthetic document : to analze deeper prospects
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    education = candidate.get("education" , {})
    certifications = candidate.get("certifications" , {})

    doc_parts = []

    doc_parts.append("\nCANDIDATE EXECUTIVE SUMMARY : ")

    title = profile.get("current_title", "Unknown Role")
    yoe = profile.get("years_of_experience", 0)
    industry = profile.get("current_industry", "Unknown Industry")
    company = profile.get("current_company", "Unknown Company")
    company_size = profile.get("current_company_size", "Unknown Size")

    doc_parts.append(f"Primary Profile: {title} with {yoe} years of total experience.")
    doc_parts.append(f"Current Operational Environment: Operating in the {industry} sector at {company} (Company Scale: {company_size} employees).")

    # 2. market positioning
    headline = profile.get("headline", "")
    if headline:
        doc_parts.append(f"Market Positioning (Headline): {headline}")

    inflated_titles = ["director", "principal", "head of", "vp", "chief", "architect"]
    if yoe < 3.0 and any(senior_title in title.lower() for senior_title in inflated_titles):
        doc_parts.append(f"WARNING: Seniority Inflation Risk. Candidate holds a highly senior title ('{title}') with only {yoe} years of total industry experience.")

    summary = profile.get("summary", "")
    if summary:
        doc_parts.append(f"Self-Reported Narrative: {summary}")

    location = profile.get("location", "")
    country = profile.get("country", "")
    if location and country:
        doc_parts.append(f"Geographic Context: {location}, {country}")



    doc_parts.append("\n DEEP TECHNICAL TRAJECTORY : ")

    if not history:
        doc_parts.append("WARNING: No career history provided. Unable to verify real-world engineering experience.")
        return doc_parts

    # for i in enumerate(history):
    #   company = history.get("company" , "unemployed")
    #   title_ = history.get("title" , "unknown")
    #   firm = history.get("company" , "unknown")
    #   doc_parts.append(f"{title} for {company} for ")

    # Title chaser

    if len(history) >= 3:
        avg_tenure = sum(role.get("duration_months", 0) for role in history) / len(history)
        if avg_tenure < 18:
            doc_parts.append(f"WARNING: Demonstrates chronic job-hopping and title-chasing behavior. Average tenure is only {avg_tenure:.1f} months across recent roles.")

    # consulting and research traps 
    consulting_flags = ["it services", "consulting", "tcs", "infosys", "wipro", "accenture", "cognizant", "deloitte"]
    research_flags = ["academic", "research", "university", "lab", "phd", "postdoc"]

    all_consulting = all(any(c in role.get("industry", "").lower() or c in role.get("company", "").lower() for c in consulting_flags) for role in history)
    all_research = all(any(r in role.get("industry", "").lower() or r in role.get("company", "").lower() for r in research_flags) for role in history)

    if all_consulting:
        doc_parts.append("WARNING: Career is entirely based in IT Services/Consulting with no demonstrable internal product-company experience.")
    if all_research:
        doc_parts.append("WARNING: Pure research/academic background. Lacks evidence of deploying models or systems to production users.")

    # Architect
    recent_role = history[0]
    recent_title = recent_role.get("title", "").lower()
    recent_desc = recent_role.get("description", "").lower()

    senior_titles = ["architect", "tech lead", "manager", "director", "vp", "head", "principal"]
    action_verbs = ["code", "shipped", "built", "implemented", "wrote", "hands-on", "developed"]

    if any(kw in recent_title for kw in senior_titles):
        if not any(kw in recent_desc for kw in action_verbs):
            doc_parts.append("WARNING: Moved into pure architecture/management. No explicit evidence of writing production code in the most recent role.")

    for index, role in enumerate(history):
        title = role.get("title", "Unknown Title")
        company = role.get("company", "Unknown Company")
        months = role.get("duration_months", 0)
        company_size = role.get("company_size", "Unknown Size")
        industry = role.get("industry", "Unknown Industry")
        desc = role.get("description", "")
        doc_parts.append(f"[{index + 1}] Role: {title} at {company} ({months} months) | Environment: {industry}, Scale: {company_size} employees")

        if desc:
            doc_parts.append(f"    Technical Implementation & Impact: {desc}")


    doc_parts.append("\n--- CAREER TRAJECTORY & EXPERIENCE ---")
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




    # Education aspects :
    doc_parts.append("\nACADEMICS & FOUNDATION : ")

    has_tier_1 = False
    for index, edu in enumerate(education):
        institution = edu.get("institution", "Unknown Institution")
        degree = edu.get("degree", "Unknown Degree")
        field = edu.get("field_of_study", "Unknown Field")
        start = edu.get("start_year", "Unknown")
        end = edu.get("end_year", "Unknown")
        tier = edu.get("tier", "unknown")
        grade = edu.get("grade")

        edu_string = f"[{index + 1}] Degree: {degree} in {field} from {institution} ({start}-{end})"

        if tier == "tier_1":
            edu_string += " | SIGNAL: Elite Tier-1 Academic Institution"
            has_tier_1 = True
        elif tier == "tier_2":
            edu_string += " | Foundation: Tier-2 Institution"
        elif tier in ["tier_3", "tier_4"]:
            edu_string += f" | Foundation: {tier.replace('_', '-').title()} Institution"

        if grade:
            edu_string += f" | Academic Performance: {grade}"

        doc_parts.append(edu_string)

    if has_tier_1:
         doc_parts.append("HIGHLIGHT: Candidate possesses elite academic pedigree, indicating strong foundational potential.")

    # field mismatch
    if education:
        latest_field = education[0].get("field_of_study", "").lower()
        tech_keywords = ["computer", "software", "information", "data", "math", "physics", "electrical", "engineering"]

        if not any(kw in latest_field for kw in tech_keywords):
            doc_parts.append("Contextual Note: Candidate possesses a non-traditional/non-CS academic background. Technical competency must be evaluated heavily via career history and verified skills.")



   #technical aspects
    ai_hype_wrappers = ["langchain", "openai", "chatgpt", "llamaindex", "huggingface", "pinecone", "prompt engineering"]
    heavy_ml_data = ["machine learning", "deep learning", "pytorch", "tensorflow", "peft", "yolo", "reinforcement learning", "mlflow", "mlops", "data pipelines", "spark", "hadoop", "statistical modeling"]
    core_engineering = ["python", "java", "c++", "sql", "docker", "kubernetes", "aws", "gcp", "azure", "backend", "system design"]

    categorized_skills = {
        "Core Engineering (18+ Months)": [],
        "Heavy ML & Data Infra (18+ Months)": [],
        "Recent AI/LLM Wrappers": [], # We don't filter wrappers by time, since they are new technologies
        "Domain & Additional Skills (18+ Months)": [],
        "Shallow Exposure / Familiarity (< 18 Months)": [] # The Graveyard for resume-stuffing
    }

    max_core_months = 0
    high_level_ml_claims = []

    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "unknown").capitalize()
        months = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        skill_str = f"{name} ({prof}, {months} months, {endorsements} endorsements)"
        name_lower = name.lower()

        # Track max core engineering time for the Framework Trap
        if any(core in name_lower for core in core_engineering):
            max_core_months = max(max_core_months, months)

        # Track high-level claims for the Aspirational Trap
        if prof in ["Advanced", "Expert"] and any(ml in name_lower for ml in heavy_ml_data):
            high_level_ml_claims.append(name)

        # 18 month logic : as given in PS
        if any(hype in name_lower for hype in ai_hype_wrappers):
            categorized_skills["Recent AI/LLM Wrappers"].append(skill_str)
        elif months < 18:
            categorized_skills["Shallow Exposure / Familiarity (< 18 Months)"].append(skill_str)
        elif any(ml in name_lower for ml in heavy_ml_data):
            categorized_skills["Heavy ML & Data Infra (18+ Months)"].append(skill_str)
        elif any(core in name_lower for core in core_engineering):
            categorized_skills["Core Engineering (18+ Months)"].append(skill_str)
        else:
            categorized_skills["Domain & Additional Skills (18+ Months)"].append(skill_str)

    for category, skill_list in categorized_skills.items():
        if skill_list:
            doc_parts.append(f"{category}: " + " | ".join(skill_list))

    # the framework enthusiast check
    if categorized_skills["Recent AI/LLM Wrappers"] and max_core_months < 24:
        doc_parts.append("WARNING: Framework Enthusiast Profile. Candidate has indexed heavily into recent AI API wrappers (LangChain, OpenAI) but lacks deep foundational engineering/ML deployment experience (Under 2 years of core engineering).")

    if high_level_ml_claims and history:
        history_text = " ".join([role.get("description", "").lower() for role in history])
        unverified_claims = []

        for claim in high_level_ml_claims:
            if claim.lower() not in history_text:
                unverified_claims.append(claim)

        if unverified_claims:
            doc_parts.append(f"CRITICAL WARNING: Skill Mismatch. Candidate claims Advanced/Expert proficiency in [{', '.join(unverified_claims)}], but their professional career history shows NO evidence of utilizing these in production. High likelihood these are self-taught or side-project skills.")



    doc_parts.append("\nVERIFIED CERTIFICATIONS & CREDENTIALS : ")

    # credential tiering dictionaries
    enterprise_issuers = ["aws", "amazon web services", "google cloud", "gcp", "microsoft", "azure", "kubernetes", "cncf", "linux foundation", "databricks", "snowflake", "cisco", "offensive security", "isc2"]
    advanced_keywords = ["professional", "expert", "architect", "administrator", "cka", "ckad", "cissp"]
    current_year = 2026

    high_value_count = 0

    # Iterate and enrich each certification
    for cert in certifications:
        name = cert.get("name", "Unknown Certification")
        issuer = cert.get("issuer", "Unknown Issuer")
        year = cert.get("year", 0)

        cert_str = f"Credential: {name} (Issued by {issuer}, {year})"

        name_lower = name.lower()
        issuer_lower = issuer.lower()

        #Recency filter logic 
        if year > 0:
            age = current_year - year
            if age > 5:
                cert_str += " | Context: Legacy Credential (>5 years old, may not reflect modern cloud/ML paradigms)"

        # cloud & ebterprise verification)
        is_enterprise = any(ei in issuer_lower or ei in name_lower for ei in enterprise_issuers)
        is_advanced = any(adv in name_lower for adv in advanced_keywords)

        if is_enterprise and is_advanced:
            cert_str += " | SIGNAL: Elite Enterprise/Architecture Credential"
            high_value_count += 1
        elif is_enterprise:
            cert_str += " | SIGNAL: Verified Cloud/Platform Credential"
            high_value_count += 1

        doc_parts.append(cert_str)
    if high_value_count >= 2:
        doc_parts.append("HIGHLIGHT: Candidate holds multiple verified enterprise/cloud credentials, indicating formal architectural training beyond self-taught exposure.")



  # REDDROB SIGNALS :
    doc_parts.append("\n--- BEHAVIORAL INTELLIGENCE & MARKET SIGNALS ---")
    
    # today's date for reference 
    current_date = datetime(2026, 6, 20)

    # ghost profile
    last_active = signals.get("last_active_date")
    response_rate = signals.get("recruiter_response_rate", 1.0)

    if last_active:
        try:
            last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (current_date - last_active_dt).days

            if days_inactive > 180 or response_rate < 0.15:
                doc_parts.append(f"CRITICAL WARNING: Ghost Profile. Candidate has been inactive for {days_inactive} days and ignores {((1 - response_rate)*100):.0f}% of recruiter outreach. High risk of dead-end sourcing.")
        except ValueError:
            pass

    # reliability trap : 
    interview_rate = signals.get("interview_completion_rate", 1.0)

    if interview_rate < 0.50:
         doc_parts.append(f"CRITICAL WARNING: Severe Flake Risk. Candidate has a documented history of abandoning scheduled interviews (Completion Rate: {interview_rate*100:.0f}%).")
    elif interview_rate > 0.90:
         doc_parts.append("Reliability Signal: Exceptional follow-through. Documented history of attending scheduled interviews and communicating effectively.")

    views = signals.get("profile_views_received_30d", 0)
    saves = signals.get("saved_by_recruiters_30d", 0)

    if views > 150 and saves > 15:
        doc_parts.append("Market Signal: HIGH DEMAND. Candidate is actively being sourced, viewed, and saved by competing recruiters. Fast action required.")

    #  open-Source contributer
    github_score = signals.get("github_activity_score", -1)
    if github_score > 75:
        doc_parts.append("Technical Signal: Deeply passionate engineer with a highly active Open-Source/GitHub footprint. Indicates coding is a craft, not just a day job.")
    elif github_score != -1 and github_score < 20:
        doc_parts.append("Contextual Note: Minimal open-source/public code footprint. Technical validation relies entirely on platform assessments and past enterprise scale.")

    # Redrob verified Skills
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        # Extract only the elite scores (80+)
        elite_skills = [f"{skill} ({score}/100)" for skill, score in assessments.items() if score >= 80]
        if elite_skills:
            doc_parts.append(f"Platform-Verified Elite Competencies: {', '.join(elite_skills)}. Candidate has mathematically proven their expertise in these domains.")

    #losgistic aspects
    notice_period = signals.get("notice_period_days", -1)
    salary = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", "Unknown")
    sal_max = salary.get("max", "Unknown")

    logistics_str = "Logistics Context: "
    if notice_period != -1:
        logistics_str += f"Notice Period is {notice_period} days. "
    if sal_min != "Unknown":
        logistics_str += f"Expected Compensation is {sal_min} - {sal_max} LPA."

    if logistics_str != "Logistics Context: ":
        doc_parts.append(logistics_str)

    
    # Concatentae all the text extracted
    return "\n".join(doc_parts)



if __name__ == "__main__":

    local_filename = "sample_one.json"

    try:
        with open(local_filename, "r", encoding="utf-8") as file:
            candidate_data = json.load(file)
        doc = build_synthetic_document_1(candidate_data)
        # 3. Print the results
        print(doc)

    except FileNotFoundError:
        print(f"\nCould not find '{local_filename}'. Folder not found.")
    except json.JSONDecodeError as e:
        print(f"\n🚨 ERROR: '{local_filename}' is empty or invalid JSON. (Error details: {e})")
       