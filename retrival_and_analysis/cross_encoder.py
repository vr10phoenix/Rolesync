from sentence_transformers import CrossEncoder
from query_db import search_candidates 

def cross_encoder_rerank(query_text: str, stage_1_results: list, top_k: int = 5):
    """
    cross - encoder re-ranking
    """
    if not stage_1_results:
        print("error : candidates not loaded.")
        return []

    print("\n Cross Encoder loading ...")
    reranker = CrossEncoder("BAAI/bge-reranker-large")

    print(f"Formatting {len(stage_1_results)} candidates for semantic comparison...")
    pairs = [[query_text, doc.page_content] for doc in stage_1_results]

    print("Reranking process started : ")
    scores = reranker.predict(pairs)

    scored_candidates = list(zip(stage_1_results, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    final_shortlist = scored_candidates[:top_k]
    
    print(f"Re-ranking completed : {len(stage_1_results)} down to top {top_k}.")
    
    return final_shortlist

if __name__ == "__main__":
    sample_jd_query = """
    Looking for a strong Backend Engineer with experience in Python, SQL, and data pipelines. 
    Must know Apache Spark and Airflow. Ideally someone interested in transitioning toward ML.
    """
    
    print("\n Retriving candidates via semantic search")
    
    # searching candiadtes via query_db
    top_N_docs = search_candidates(sample_jd_query, top_k=20)
    
    #passing the resulting list to cross-encoder
    top_n_finalists = cross_encoder_rerank(sample_jd_query, stage_1_results=top_N_docs, top_k=5)
    
    # sample verification
    print("\n Re-ranked Shortlist :\n")
    for i, (doc, score) in enumerate(top_n_finalists):
        candidate_id = doc.metadata.get('candidate_id', 'UNKNOWN')
        print(f"Rank : {i + 1} | Score: {score:.4f} /100 | ID: {candidate_id}")
        print(f"Snippet: {doc.page_content[:100]}...\n")