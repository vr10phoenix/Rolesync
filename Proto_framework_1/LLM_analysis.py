import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


# strict output schema
class CandidateReasoning(BaseModel):
    candidate_id: str = Field(description="The exact Candidate ID.")
    match_score: int = Field(description="Final confidence score out of 100 based on deep fit.")
    deep_reasoning: str = Field(description="A detailed, 2-3 sentence recruiter justification on WHY they are in the Top 3. Reference specific skills, trajectory, or signals.")

class FinalShortlist(BaseModel):
    top_candidates: list[CandidateReasoning] = Field(description="The array containing the top chosen candidates.")


# LLM analysis
def llm_deep_analysis(job_description: str, encoder_result: list, top_k: int = 3):
    """
    Ingests the Cross-Encoder shortlist dicts and uses LLM reasoning to filter the Top K.
    """
    if not encoder_result:
        print("No candidates provided.")
        return None

    print(f"\n Starting LLM Analysis")
    print(f" Evaluating to the Top {len(encoder_result)} candidates...")

    # model = gemini 2.5 flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    structured_llm = llm.with_structured_output(FinalShortlist)

    candidates_context = ""
    for rank, item in enumerate(encoder_result):
        doc = item["document"]
        score = item["rerank_score"]
        cand_id = doc.metadata.get("candidate_id", "UNKNOWN")

        candidates_context += f"--- Candidate {rank+1} (ID: {cand_id}) | Stage 2 Score: {score:.2f} ---\n"
        candidates_context += f"{doc.page_content}\n\n"

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
        3. Output the exact candidate IDs, a final match score out of 100, and a deep, reasoned Recruiter Justification.
        """
    )

    chain = prompt | structured_llm
    final_result = chain.invoke({
        "jd": job_description,
        "candidates": candidates_context,
        "top_k": top_k
    })

    return final_result

if __name__ == "__main__":
    print("LLM Analysis : Functional")