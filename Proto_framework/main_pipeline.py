import time
from query_db import search_candidates
from cross_encoder import cross_encoder_rerank
from LLM_analysis import llm_deep_analysis


# rate limiting wrapper for gemini model LLm
def process_shortlist_safely(job_description, top_100_candidates, batch_size=5, sleep_time=15):
    """
    Safely processes candidates in chunks to avoid hitting Gemini Free Tier RPM limits.
    """
    final_results = []
    
    print(f"\n[STAGE 3] Processing {len(top_100_candidates)} candidates in batches of {batch_size}...")
    
    # smaller chunks input : each chunk = 5
    for i in range(0, len(top_100_candidates), batch_size):
        batch = top_100_candidates[i:i + batch_size]
        print(f"\n--- Processing Batch {i//batch_size + 1} (Candidates {i+1} to {i+len(batch)}) ---")
        
        try:
            batch_result = llm_deep_analysis(job_description, cross_encoder_output=batch, top_k=3)
            
            if batch_result and hasattr(batch_result, 'top_candidates'):
                final_results.extend(batch_result.top_candidates)
                
        except Exception as e:
            print(f"Error processing batch: {e}. Retrying in 30 seconds...")

            time.sleep(30)
            
        if i + batch_size < len(top_100_candidates):
            print(f"Batch complete. Sleeping for {sleep_time} seconds to respect API limits...")
            time.sleep(sleep_time)

    # sort to find the best n from the sorted n1 from each batch
    final_results.sort(key=lambda x: x.match_score, reverse=True)
    return final_results[:3]

# main pipeline
if __name__ == "__main__":
    sample_jd_query = """
    Looking for a strong Backend Engineer with experience in Python, SQL, and data pipelines. 
    Must know Apache Spark and Airflow. Ideally someone interested in transitioning toward ML.
    We need a fast-learner ready to step up into a leadership role.
    """
    
    print("JOBSYNC AI\n")

    print("\nRunning Bi-Encoder FAISS Retrieval...")
    stage_1_docs = search_candidates(sample_jd_query, top_k=20) 

    if not stage_1_docs:
        print("Pipeline stopped: No candidates retrieved from FAISS.")
        exit()

    print("\nRunning Cross-Encoder Re-ranking...")
    #Cross-Encoder
    cross_encoder_output = cross_encoder_rerank(
        query_text=sample_jd_query, 
        stage_1_results=stage_1_docs, 
        top_k=5  
    )

    if not cross_encoder_output:
        print("Pipeline stopped: Cross-Encoder returned no results.")
        exit()

    #LLM Analysis
    finalists = process_shortlist_safely(
        job_description=sample_jd_query,
        top_100_candidates=cross_encoder_output,
        batch_size=5,
        sleep_time=15
    )
    
    if finalists:
        print("\n" + "="*50)
        print("FINAL LIST OF CANDIDATES")
        print("="*50)
        
        for i, candidate in enumerate(finalists):
            print(f"\nRANK #{i+1} | Candidate: {candidate.candidate_id} | Score: {candidate.match_score}/100")
            print(f"Recruiter Justification:\n{candidate.deep_reasoning}")
    else:
        print("\nError: LLM failed to return a properly formatted shortlist.")