import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Ensure your Gemini API Key is set globally
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6LiIGACGtKb7FxY9pWv6G0BmILbemy9JqpmwwPtrzWIGw")

def generate_semantic_query(raw_job_description: str) -> str:
    """
    PHASE 0: JD INTELLIGENCE (THE TRAP EVADER)
    Translates a raw Job Description into a highly concentrated semantic query
    that perfectly aligns with the Redrob Candidate Schema, actively avoiding
    known recruiter traps (Ghosting, Title Chasers, Framework Wrappers).
    """
    print("\n--- INITIATING PHASE 0: JD INTELLIGENCE ---")
    print("Analyzing Job Description for Deep Context, Signals, and Trap Evasion...")

    # Upgraded to Pro for advanced reasoning, deduction, and trap-avoidance
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)

    # The "Anti-Trap Rosetta Stone" Prompt.
    # Forces the LLM to write a query that naturally repels our injected FAISS warnings.
    prompt = PromptTemplate.from_template(
        """
        You are an elite AI Executive Recruiter for Redrob, specializing in early-stage, AI-native startups.
        Your task is to translate a raw Job Description into a highly optimized, dense paragraph
        designed to perfectly query a FAISS Vector Database.

        The database contains candidate documents built from a strict JSON schema that includes
        explicit "WARNING" labels for bad fits and "HIDDEN MATCH SIGNAL" labels for hidden gems.

        You must synthesize the JD into a single, cohesive narrative paragraph that explicitly mandates:

        1. TRAP EVASION (CRITICAL):
           - Explicitly demand stable tenure (to avoid "title-chasers").
           - Explicitly demand hands-on production code deployment at a *product* company (to avoid "pure research", "pure consulting", or "pure architecture" traps).
           - Explicitly demand alignment between their current title and engineering (to avoid keyword-stuffing non-engineers).

        2. FOUNDATIONS OVER FRAMEWORKS:
           - Emphasize deep, multi-year foundational experience (e.g., Python, Vector Databases, Machine Learning, Ranking/Retrieval) over short-term AI hype frameworks (LangChain, OpenAI wrappers).
           - Format skills exactly like this: "[Proficiency] proficiency in [Skill Name] ([X] months experience)."

        3. HIDDEN MATCH HUNTING:
           - If the JD asks for search, AI, or ranking, explicitly include this exact phrase in your output to trigger our vector magnet: "HIDDEN MATCH SIGNAL: Career history explicitly shows building ranking, search, or recommendation systems at scale."

        4. BEHAVIORAL INTELLIGENCE & REDROB SIGNALS:
           - Map soft skills to these EXACT phrases to match our database:
             - "Highly responsive and active communicator." (Avoids Ghost profiles)
             - "Exceptional follow-through and high reliability in professional commitments."
             - "Deeply passionate about technology with a highly active GitHub footprint."

        INSTRUCTIONS:
        Write a concise, dense, 4-5 sentence narrative query.
        DO NOT use bullet points. DO NOT output JSON. DO NOT include boilerplate.
        Output ONLY the pure, optimized text string designed for mathematical vector matching.

        RAW JOB DESCRIPTION:
        {jd}
        """
    )

    # Execute the chain
    chain = prompt | llm
    # optimized_query = chain.invoke({"jd": raw_job_description}).content

    # print("Success! Schema-Aligned & Trap-Proof Query Generated.")
    # return optimized_query.strip()
    raw_content = chain.invoke({"jd" : raw_job_description}).content

    if isinstance(raw_content, list) and len(raw_content) >0 :
      optimized_query = raw_content[0].get("text","")

    else:
      optimized_query = str(raw_content)

    return optimized_query.strip()


if __name__ == "__main__":
    # The Official Redrob Trap JD
    redrob_raw_jd = """
    Job Description: Senior AI Engineer — Founding Team
    Company: Redrob AI
    Experience: 5–9 years
    We need deep technical depth in modern ML systems (embeddings, retrieval, ranking) AND a scrappy shipper attitude.
    Disqualifiers:
    - Pure research/academic backgrounds without production deployment.
    - AI experience that is only Langchain/OpenAI wrappers under 12 months.
    - Senior engineers who haven't written code in 18 months (pure architects).
    - Title chasers jumping every 1.5 years.
    - Pure consulting firm backgrounds (TCS, Infosys, etc.).
    We need strong Python, vector database experience in production, and evaluation framework knowledge.
    Notice period: Sub-30 days preferred.
    """

    M_query = generate_semantic_query(redrob_raw_jd)
    
    print("\n--- FINAL OPTIMIZED VECTOR QUERY ---")
    print(M_query)