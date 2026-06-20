from sentence_transformers import CrossEncoder

# Cross encoder re-ranking

def cross_encoder_rerank(query_text: str, faiss_data: list, top_k: int = 5) -> list:
    """
    Takes documents, pairs them with the JD, and scores them.
    """
    if not faiss_data:
        print("No candidates provided from Stage 1.")
        return []

    print("\nLoading Cross-Encoder")
    reranker = CrossEncoder("BAAI/bge-reranker-large")

    # Creating pair : [JD, Candidate_Text] 
    pairs = [[query_text, doc.page_content] for doc in faiss_data]

    print("Executing Re-ranking Matrix\n")
    scores = reranker.predict(pairs)

    scored_candidates = []
    for doc, score in zip(faiss_data, scores):
        scored_candidates.append({
            "document": doc,
            # Ensure correct format
            "rerank_score": float(score) 
        })

    # Sort decreasing by score
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    final_shortlist = scored_candidates[:100]
    print(f"[STAGE 2] Re-ranking Complete! Sliced down to the absolute Top {100}.")
    
    return final_shortlist

if __name__ == "__main__":
    print("Cross Encoder : functional")