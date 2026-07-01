import csv
from typing import List, Any
from sentence_transformers import CrossEncoder
import time
from retriever import CandidateRetriever


def cross_encoder_rerank_and_export(
    query: str,
    bi_encoder_candidates: List[Any],
    model_name: str = "BAAI/bge-reranker-base",
    top_k: int = 100,
    output_csv: str = "result_1.csv"
):
    print(f"\n[STAGE 2] Loading Cross-Encoder Model: '{model_name}'...")
    model = CrossEncoder(model_name, max_length=512, device='cpu')
    cross_inp = [[query, candidate.text_preview] for candidate in bi_encoder_candidates]

    print("[STAGE 2] Running Cross-Encoder prediction...")
    start_time = time.time()
    raw_scores = model.predict(cross_inp)
    end_time = time.time()
    print(f"completed in {end_time - start_time:.2f} seconds\n")

    # Build list with raw scores and metadata
    scored = []
    for candidate, raw_score in zip(bi_encoder_candidates, raw_scores):
        meta = candidate.metadata
        exp = meta.get('years_of_experience', 'N/A')
        skills = meta.get('skill_names', [])
        num_skills = len(skills) if skills else 0
        preview = candidate.text_preview[:120].replace('\n', ' ')
        reason = f"Exp: {exp} yrs, Skills: {num_skills}, Preview: {preview}..."

        scored.append({
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "raw_score": float(raw_score),
            "reasoning": reason
        })

    # 1. Primary sort: raw score desc, then candidate_id asc (for consistent pool)
    scored.sort(key=lambda x: (-x["raw_score"], x["candidate_id"]))
    final_shortlist = scored[:top_k]

    # 2. Scale raw scores to [0.6, 1.0] within this top_k
    raw_vals = [c["raw_score"] for c in final_shortlist]
    min_raw, max_raw = min(raw_vals), max(raw_vals)
    range_raw = max_raw - min_raw
    if range_raw == 0:
        range_raw = 1.0

    DESIRED_LOW = 0.6
    DESIRED_HIGH = 1.0
    for c in final_shortlist:
        normalized = (c["raw_score"] - min_raw) / range_raw
        c["display_score"] = DESIRED_LOW + normalized * (DESIRED_HIGH - DESIRED_LOW)
        c["rounded_score"] = round(c["display_score"], 4)   # for tie‑break

    # 3. Re‑sort by rounded score desc, then candidate_id asc (visual tie‑break)
    final_shortlist.sort(key=lambda x: (-x["rounded_score"], x["candidate_id"]))

    # Print results
    print(f"\n[STAGE 2] --- TOP {len(final_shortlist)} SHORTLISTED CANDIDATES ---")
    for rank, candidate in enumerate(final_shortlist, start=1):
        print(f"Rank #{rank:<3} | ID: {candidate['candidate_id']:<10} | Name: {candidate['name']:<20} | Score: {candidate['rounded_score']:.4f}")
    print("-" * 70 + "\n")

    # Export CSV
    print(f"[STAGE 2] Exporting Top {len(final_shortlist)} candidates to '{output_csv}' ...")
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['candidate_id', 'rank', 'score', 'reasoning']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(final_shortlist, start=1):
            writer.writerow({
                "candidate_id": candidate["candidate_id"],
                "rank": rank,
                "score": candidate["rounded_score"],
                "reasoning": candidate["reasoning"]
            })

    print(f"[STAGE 2] Success! Pipeline complete.\n")
    return final_shortlist


if __name__ == "__main__":
    # Quick test (if you have the retriever available)
    retriever = CandidateRetriever("index_database")
    query = "We require an active software engineer with stable tenure..."
    results = retriever.search(query, top_k=200, recall_k=200, alpha=0.5)
    cross_encoder_rerank_and_export(query, results, top_k=100, output_csv="submission.csv")