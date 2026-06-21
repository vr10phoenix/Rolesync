import os
import csv
import math
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from search import search_candidates_faiss

def cross_encoder_rerank_and_export(query_text: str, stage_1_results: list, top_k: int = 100):
    """
    STAGE 2: CROSS-ENCODER RE-RANKING & EXPORT
    Scores the 1000 candidates, normalizes to 0-1, filters to Top 100, and writes to CSV.
    """
    if not stage_1_results:
        print("No candidates provided from Stage 1.")
        return

    print("\n[STAGE 2] Loading Cross-Encoder Model (BAAI/bge-reranker-large)...")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    print(f"[STAGE 2] Formatting {len(stage_1_results)} candidates for deep semantic comparison...")
    pairs = [[query_text, doc.page_content] for doc in stage_1_results]

    print("[STAGE 2] Executing Neural Re-ranking. This may take a moment...")
    raw_scores = reranker.predict(pairs , batch_size = 32 , show_progress_bar = True)

    # # Attach scores to documents and normalize
    # scored_candidates = []
    # for doc, raw_score in zip(stage_1_results, raw_scores):
    #     raw_scores = raw_scores
    #     scored_candidates.append({
    #         "document": doc,
    #         "score": raw_scores
    #     })

    # Attach scores to documents and normalize
    scored_candidates = []
    for doc, raw_score in zip(stage_1_results, raw_scores):
        # 1. Normalize the raw logit score to a 0.0 - 1.0 range using Sigmoid
        normalized_score = 1 / (1 + math.exp(-float(raw_score)))

        # 2. Append the INDIVIDUAL normalized score
        scored_candidates.append({
            "document": doc,
            "score": normalized_score
        })

    # Sort mathematically descending
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    final_shortlist = scored_candidates[:top_k]

    print(f"[STAGE 2] Re-ranking Complete! Filtered down to Top {top_k}.")
    print("-" * 50)

    # --- CSV EXPORT LOGIC ---
    csv_filename = "result.csv"
    print(f"[EXPORT] Writing final shortlist to {csv_filename}...")

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write exact required headers
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, item in enumerate(final_shortlist, start=1):
            doc = item["document"]
            score = item["score"]

            # 1. Extract metadata dictionary safely
            meta = doc.metadata
            candidate_id = meta.get('candidate_id', 'UNKNOWN')

            # 2. Extract and format Title
            # (Change 'current_title' to whatever key you used in your JSON schema)
            title = meta.get('current_title', meta.get('title', 'Unknown Title'))

            # 3. Extract and format Experience Years
            raw_exp = meta.get('years_of_experience', meta.get('experience', 0.0))
            try:
                exp_formatted = f"{float(raw_exp):.1f}"
            except (ValueError, TypeError):
                exp_formatted = str(raw_exp)

            # 4. Extract and count AI Core Skills
            # Handles both list types and comma-separated string types
            raw_skills = meta.get('ai_core_skills', meta.get('skills', []))
            if isinstance(raw_skills, list):
                skill_count = len(raw_skills)
            elif isinstance(raw_skills, str) and raw_skills.strip():
                skill_count = len(raw_skills.split(','))
            else:
                skill_count = 0

            # 5. Extract and format Response Rate
            raw_rr = meta.get('response_rate', 0.0)
            try:
                rr_formatted = f"{float(raw_rr):.2f}"
            except (ValueError, TypeError):
                rr_formatted = str(raw_rr)

            # 6. Construct the dynamic reasoning string EXACTLY as requested
            dynamic_reasoning = f"{title} with {exp_formatted} yrs; {skill_count} AI core skills; response rate {rr_formatted}."

            # Write to CSV
            writer.writerow([candidate_id, rank, f"{score:.4f}", dynamic_reasoning])

            # Print Top 5 to terminal for immediate visual verification
            if rank <= 5:
                print(f"Rank #{rank} | ID: {candidate_id} | Score: {score:.4f} | {dynamic_reasoning}")

    print("-" * 50)
    print(f"🎉 Success! Submission file '{csv_filename}' generated with {len(final_shortlist)} candidates.")

if __name__ == "__main__":
    # Ensure this variable matches the query you generated via JD Intelligence
    super_query = """
    Looking for a strong Backend Engineer with experience in Python, SQL, and data pipelines.
    Must know Apache Spark and Airflow. Ideally someone interested in transitioning toward ML.
    """

    print("\n=== STARTING TWO-STAGE RETRIEVAL PIPELINE ===")

    # 1. Retrieve Top 1000 via Vector DB
    top_1000_docs = search_candidates_faiss(super_query, top_k=200)

    # 2. Re-rank to Top 100 and Export to CSV
    cross_encoder_rerank_and_export(super_query, stage_1_results=top_1000_docs, top_k=100)