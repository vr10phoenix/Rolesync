import os
import csv
import math
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from Prototying.Proto_framework_2.search import search_candidates_faiss
from Prototying.Proto_framework_1.JD_sorting import M_query

def cross_encoder_rerank_and_export(query_text: str, stage_1_results: list, top_k: int = 100):
    """
  Cross Encoder re-ranking sytem to refine 2nd - stage of the Candidates list 
    """
    if not stage_1_results:
        print("No candidates provided")
        return
    
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    pairs = [[query_text, doc.page_content] for doc in stage_1_results]
    raw_scores = reranker.predict(pairs ,batches = 10 , show_progress_bar = True)

    # # Attach scores normalization
    # scored_candidates = []
    # for doc, raw_score in zip(stage_1_results, raw_scores):
    #     raw_scores = raw_scores
    #     scored_candidates.append({
    #         "document": doc,
    #         "score": raw_scores
    #     })

    scored_candidates = []
    for doc, raw_score in zip(stage_1_results, raw_scores):
        # 1. Normalize the raw logit score to a 0.0 - 1.0 range using Sigmoid
        normalized_score = 1 / (1 + math.exp(-float(raw_score)))

        scored_candidates.append({
            "document": doc,
            "score": normalized_score
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    final_shortlist = scored_candidates[:top_k]

    print(f"[STAGE 2] Re-ranking Complete! Filtered down to Top {top_k}.")
    print("-" * 50)

    csv_filename = "result.csv"
    print(f"[EXPORT] Writing final shortlist to {csv_filename}...")

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, item in enumerate(final_shortlist, start=1):
            doc = item["document"]
            score = item["score"]

            # 1. Extract metadata 
            meta = doc.metadata
            candidate_id = meta.get('candidate_id', 'UNKNOWN')
            title = meta.get('current_title', meta.get('title', 'Unknown Title'))

            raw_exp = meta.get('years_of_experience', meta.get('experience', 0.0))
            try:
                exp_formatted = f"{float(raw_exp):.1f}"
            except (ValueError, TypeError):
                exp_formatted = str(raw_exp)


            raw_skills = meta.get('ai_core_skills', meta.get('skills', []))
            if isinstance(raw_skills, list):
                skill_count = len(raw_skills)
            elif isinstance(raw_skills, str) and raw_skills.strip():
                skill_count = len(raw_skills.split(','))
            else:
                skill_count = 0

            raw_rr = meta.get('response_rate', 0.0)
            try:
                rr_formatted = f"{float(raw_rr):.2f}"
            except (ValueError, TypeError):
                rr_formatted = str(raw_rr)
            dynamic_reasoning = f"{title} with {exp_formatted} yrs; {skill_count} AI core skills; response rate {rr_formatted}."

            # Write to CSV
            writer.writerow([candidate_id, rank, f"{score:.4f}", dynamic_reasoning])
            if rank <= 5:
                print(f"Rank #{rank} | ID: {candidate_id} | Score: {score:.4f} | {dynamic_reasoning}")


if __name__ == "__main__":
    query = M_query

    print("\nSTARTING TWO-STAGE RETRIEVAL PIPELINE")

    top_1000_docs = search_candidates_faiss(query, top_k=200)

    cross_encoder_rerank_and_export(query, stage_1_results=top_1000_docs, top_k=100)