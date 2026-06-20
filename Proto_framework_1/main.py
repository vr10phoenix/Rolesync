import os
from search_database import search_candidates_faiss
from cross_encoder import cross_encoder_rerank
from LLM_analysis import llm_deep_analysis

# api key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def run_recruiter_pipeline(job_description: str):
    print("Jobsync AI\n")
    
    # Semantic search
    top_1000_docs = search_candidates_faiss(job_description, top_k=1000)
    
    if not top_1000_docs:
        print("Pipeline aborted at Stage 1.")
        return
    
    #Cross -Encoder reranking
    top_100_dicts = cross_encoder_rerank(job_description, stage_1_results=top_1000_docs, top_k = 100)

    if not top_100_dicts:
        print("Pipeline aborted at Stage 2.")
        return
    
    #LLM analysis
    final_report = llm_deep_analysis(job_description, stage_2_results=top_100_dicts, top_k=10)

    
    #Final list
    print("FINAL SHORTLIST GENERATED")

    for i, candidate in enumerate(final_report.top_candidates):
        print(f"\nRANK #{i+1} | Candidate ID: {candidate.candidate_id} | Confidence Score: {candidate.match_score}/100")
        print(f"Recruiter Justification:\n{candidate.deep_reasoning}")

if __name__ == "__main__":
    sample_jd = M_query
    
    run_recruiter_pipeline(sample_jd)