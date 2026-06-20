import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Gemini API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


# Analysis output format:
class CandidateReasoning(BaseModel):
    candidate_id: str = Field(description="The exact Candidate ID.")
    match_score: int = Field(description="Final confidence score out of 100 based on deep fit.")
    deep_reasoning: str = Field(description="A detailed, 2-3 sentence recruiter justification on WHY they are in the Top 3. Reference specific skills, career trajectory, or behavioral signals.")

class FinalShortlist(BaseModel):
    top_candidates: list[CandidateReasoning] = Field(description="The array containing exactly the top 3 chosen candidates.")

# LLM Analysis
def llm_deep_analysis(job_description: str, stage_2_results: list, top_k: int = 3):
    """
    Ingests the Cross-Encoder shortlist and uses LLM reasoning
      to find the Top top_k candidates.
    """
    if not stage_2_results:
        print("error : No Candidates detected")
        return None

    print(f"LLM Analysis starting : \n")
    print(f" Model for reasoning : Gemini 2.5 flash\n")
    
    # temp = 0.1 to prevent halluncination
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    structured_llm = llm.with_structured_output(FinalShortlist)

    candidates_context = ""
    for rank, item in enumerate(stage_2_results):
        doc = item["document"]
        cross_encoder_score = item["rerank_score"]
        cand_id = doc.metadata.get("candidate_id", "UNKNOWN")
        
        candidates_context += f"--- Candidate {rank+1} (ID: {cand_id}) | Cross-Encoder Base Score: {cross_encoder_score:.2f} ---\n"
        candidates_context += f"{doc.page_content}\n\n"

    # Strict prompt instructions
    prompt = PromptTemplate.from_template(
        """
        You are an elite Executive Recruiter AI. 
        You have a Job Description and a shortlist of highly qualified candidates that have already passed a rigorous semantic filter.
        Your task is to perform a deep semantic analysis and select the absolute Top {top_k} candidates for the role.

        JOB DESCRIPTION:
        {jd}

        SHORTLISTED CANDIDATES:
        {candidates}

        INSTRUCTIONS:
        1. Analyze their skills, career trajectory, and behavioral signals against the core needs of the job description.
        2. Identify the {top_k} candidates with the strongest overall profile, maturity, and direct experience.
        3. Output the exact candidate IDs, a final match score out of 100, and a deep, reasoned Recruiter Justification for why they fit perfectly.
        """
    )

    # Execute the pipeline
    print(f"Running Jobsync Deep analysis ....")
    chain = prompt | structured_llm
    final_json_result = chain.invoke({
        "jd": job_description, 
        "candidates": candidates_context, 
        "top_k": top_k
    })
    
    return final_json_result

#Integrating all components
if __name__ == "__main__":
    sample_jd_query = """
    Looking for a strong Backend Engineer with experience in Python, SQL, and data pipelines. 
    Must know Apache Spark and Airflow. Ideally someone interested in transitioning toward ML.
    We need a fast-learner ready to step up into a leadership role.
    """
    
    cross_encoder_output = [
        {
            "document": Document(page_content="Backend Engineer | SQL, Spark, Cloud. 6.9 years building data pipelines. Career Trajectory: Promoted to Lead Backend Engineer after 2 years. Highly responsive.", metadata={"candidate_id": "CAND_0000001"}),
            "rerank_score": 0.98
        },
        {
            "document": Document(page_content="Data Engineer | Python, Airflow. 4 years experience. Trajectory: Static role for 4 years.", metadata={"candidate_id": "CAND_0000010"}),
            "rerank_score": 0.85
        },
        {
            "document": Document(page_content="Data Scientist | Python, ML. 3 years experience. Wants to move into data engineering.", metadata={"candidate_id": "CAND_0000088"}),
            "rerank_score": 0.76
        },
        {
            "document": Document(page_content="Backend Developer | Python, SQL, Spark. 5 years experience. Built massive DAGs in Airflow. Trajectory: Two internal promotions.", metadata={"candidate_id": "CAND_0000042"}),
            "rerank_score": 0.94
        }
    ]

    finalists = llm_deep_analysis(sample_jd_query, cross_encoder_output, top_k=3)
    

    print("🎉 FINAL ENTERPRISE SHORTLIST GENERATED 🎉")
    
    for i, candidate in enumerate(finalists.top_candidates):
        print(f"\nRANK #{i+1} | Candidate: {candidate.candidate_id} | Score: {candidate.match_score}/100")
        print(f"Recruiter Justification:\n{candidate.deep_reasoning}")