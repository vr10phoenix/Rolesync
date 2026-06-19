import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def generate_semantic_query(raw_job_description: str) -> str:
    """
    Translates a raw Job Description into a highly concentrated semantic query
    """
    print("\n--- INITIATING PHASE 0: JD INTELLIGENCE ---")
    print("Analyzing Job Description for Deep Context, Signals, and Relevance...")

    # model : gemini 2.5 flash , temp : 0.1 
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    prompt = PromptTemplate.from_template(
        """
        You are an elite Executive Recruiter. Your task is to translate a raw Job Description 
        into a highly optimized, dense paragraph designed to perfectly query a FAISS Vector Database.

        The database contains candidate documents built from a strict JSON schema. 
        You must synthesize the JD into a single, cohesive narrative paragraph that explicitly includes:

        1. CORE IDENTITY & TRAJECTORY: 
           - Extract the required job title, industry, and exact years of experience.
           - Infer the required career trajectory (e.g., "Demonstrated rapid growth with internal promotions" or "Stable, long-term experience in senior roles").

        2. CONTEXTUAL SKILLS: 
           - Do not just list skills. You must assign them a schema-compliant proficiency (beginner, intermediate, advanced, expert) and infer the required duration in months.
           - Format them exactly like this: "[Proficiency] proficiency in [Skill Name] ([X] months experience)."

        3. BEHAVIORAL INTELLIGENCE & REDROB SIGNALS:
           Map soft skills and hidden requirements from the JD to these EXACT phrases (use only the ones that apply):
           - If the JD asks for fast, good communication: "Highly responsive and active communicator."
           - If the JD asks for reliability/accountability: "Exceptional follow-through and high reliability in professional commitments."
           - If the JD asks for passion/open-source activity: "Deeply passionate about technology with a highly active GitHub footprint."
           - If the JD asks for top-tier/verified technical talent: "Platform-verified top performer."

        INSTRUCTIONS:
        Write a concise, dense, 4-5 sentence narrative query. 
        DO NOT use bullet points. DO NOT output JSON. DO NOT include boilerplate. 
        Output ONLY the pure, optimized text string designed for mathematical vector matching.

        RAW JOB DESCRIPTION:
        {jd}
        """
    )

    # chain
    chain = prompt | llm
    optimized_query = chain.invoke({"jd": raw_job_description}).content

    print("structured query constructed")
    return optimized_query.strip()


if __name__ == "__main__":
    #sample case
    sample_raw_jd = """
    About Us: We are a fast-growing tech startup in IT Services looking to disrupt the industry. 
    We offer catered lunches and unlimited PTO. 
    The Role: Looking for a strong Backend Engineer with at least 4 years of experience. 
    Must be an absolute expert in Python, SQL, and data pipelines. Also need some exposure to AWS.
    Ideally someone interested in transitioning toward ML. 
    We need a fast-learner ready to step up into a leadership role. 
    Requirements: Must be a highly accountable team player and a reliable communicator.
    """
    
    M_query = generate_semantic_query(sample_raw_jd)
    
    print(" Modififed query:\n")
    print(M_query)
    