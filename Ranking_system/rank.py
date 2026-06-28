import csv
import math
import re
from typing import List , Any
from sentence_transformers import CrossEncoder
from document_builder import build_candidate_document,load_candidates, process_candidates_pipeline
from Index_builder import build_index
from retrival import CandidateRetriever
from JD_sorting import generate_semantic_query
import time
import argparse

def sigmoid(x :float) -> float:
    """
    Normalizes Cross-Encoder logits to a 0-1 probability.
    """
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def generate_reasoning_from_preview(text_preview: str) -> str:
    """
    Parses the structured synthetic document (text_preview) to extract 
    """
    if not text_preview:
        return "No profile data available."

    #  Extract Experience
    exp_match = re.search(r"Total Experience:\s*(.*?)(?:\n|$)", text_preview)
    experience = exp_match.group(1) if exp_match else "Experience unknown"

    # Extract AI & Core Skills
    ai_skills_match = re.search(r"Recent AI Frameworks:\s*(.*?)(?:\n|$)", text_preview)
    core_skills_match = re.search(r"Core Engineering Foundation:\s*(.*?)(?:\n|$)", text_preview)
    
    skills_list = []
    if ai_skills_match:
        skills_list.append(f"AI: {ai_skills_match.group(1)}")
    if core_skills_match:
        skills_list.append(f"Core: {core_skills_match.group(1)}")
        
    skills_str = " | ".join(skills_list) if skills_list else "No verified skills"

    # Extract Hidden Signals / Risks
    signals = []
    if "HIDDEN MATCH SIGNAL" in text_preview:
        signals.append("Hidden Match Signal (Ranking/Search Exp)")
    if "WARNING:" in text_preview:
        signals.append("Contains Risk Flags")

    signals_str = " | ".join(signals) if signals else "Standard profile"

    # Combine into a dense, readable reasoning string
    reasoning = f"[{experience}] Skills: {skills_str}. Notes: {signals_str}"
    
    # concatenate reasoning
    return reasoning[:250] + "..." if len(reasoning) > 250 else reasoning


def cross_encoder_rerank_and_export(query:str , bi_encoder_candidates : List[Any] ,model_name : str = "BAAI/bge-reranker-base",top_k: int = 100 , output_csv: str = "result_1.csv"):

    print(f"\n[STAGE 2] Loading Cross-Encoder Model: '{model_name}'...")
    model = CrossEncoder(model_name , max_length = 512)
    cross_inp = [[query , candidate.text_preview] for candidate in bi_encoder_candidates]

    print("[STAGE 2] Running Cross-Encoder prediction...")
    start_time = time.time()
    raw_scores = model.predict(cross_inp)
    scored_candidates = []
    end_time = time.time()
    exe_time = end_time - start_time
    print(f"completed in {exe_time} seconds \n ")
    # Loop through and score all candidates first
    for candidate , raw_score in zip(bi_encoder_candidates , raw_scores):
        normalized_score = sigmoid(float(raw_score))
        
        # Parse the reasoning right here!
        raw_preview = getattr(candidate, 'text_preview', '')
        parsed_reasoning = generate_reasoning_from_preview(raw_preview)
        
        scored_candidates.append({
            "candidate_id" : candidate.candidate_id ,
            "name": candidate.name ,
            "score" : normalized_score,
            "reasoning" : parsed_reasoning # <--- Add the parsed reasoning to the dict
        })

    # Sort the list ONLY ONCE after the loop finishes
    scored_candidates.sort(key = lambda x : x["score"] , reverse = True)

    # Grab the top K
    final_shortlist = scored_candidates[:top_k]


    # To print the output 
    print(f"\n[STAGE 2] --- TOP {len(final_shortlist)} SHORTLISTED CANDIDATES ---")
    for rank, candidate in enumerate(final_shortlist, start=1):
        print(f"Rank #{rank:<3} | ID: {candidate['candidate_id']:<10} | Name: {candidate['name']:<20} | Score: {candidate['score']:.4f}")
    print("-" * 70 + "\n")


    print(f"[STAGE 2] Exporting Top {len(final_shortlist)} candidates to '{output_csv}' ...")

    # Export the file ONLY ONCE
    with open(output_csv , mode = 'w' , newline = '' , encoding = 'utf-8') as csv_file:
        # Make sure 'reasoning' is in your fieldnames!
        fieldnames = ['candidate_id' , 'rank' , 'score', 'reasoning']
        writer = csv.DictWriter(csv_file , fieldnames=fieldnames)

        writer.writeheader()
        for rank , candidate in enumerate(final_shortlist , start = 1):
            writer.writerow({
                "candidate_id" : candidate["candidate_id"],
                "rank" : rank,
                "score" : round(candidate["score"] , 4),
                "reasoning" : candidate["reasoning"] # <--- Write it to the CSV
            })

    print(f"[STAGE 2] Success! Pipeline complete.\n")
    return final_shortlist


# if __name__ == "__main__":
#     # Example usage:
#     retriever = CandidateRetriever("index_database__1")
#     M_query = M_query
#     start_time = time.time()
#     st_1_res = retriever.search(M_query , top_k = 200 , recall_k = 0.5)
#     end_time = time.time()

#     exe_time = end_time - start_time
#     print(f"\n search completed in {exe_time} seconds \n")
#     cross_encoder_rerank_and_export(query = M_query , bi_encoder_candidates = st_1_res ,
#                                     top_k = 100)
#     pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidates using Cross-Encoder reranking.")
    parser.add_argument("--candidates", type=str, required = True,
                        help="Path to the candidates JSONL file")
    parser.add_argument("--out", type=str, required=True,
                        help="Path to export the ranked shortlist CSV file.")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of top candidates to export (default=100).")
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-large",
                        help="Cross-Encoder model name (default=BAAI/bge-reranker-large).")

    args = parser.parse_args()
    
    # build_index(args.candidates , args.out)


    # Example usage with parsed args
    retriever = CandidateRetriever("index_database__1")
    redrob_raw_jd = """
    Job Description: Senior AI Engineer — Founding Team
    Company: Redrob AI
    Experience: 5–9 years
    We need deep technical depth in modern ML systems (embeddings, retrieval, ranking) AND a scrappy shipper attitude.
    Disqualifiers:
    - Pure research/academic backgrounds without production deployment.
    - AI experience that is only Langchain/OpenAI wrappers under 12 months.
    - Senior engineers who haven't written code in 18 months (pure architects).
    - Title chasers jumping every 1.5 years.
    - Pure consulting firm backgrounds (TCS, Infosys, etc.).
    We need strong Python, vector database experience in production, and evaluation framework knowledge.
    Notice period: Sub-30 days preferred.
    """
    # Query = generate_semantic_query(redrob_raw_jd)
    Query = "We require an active software engineer with stable tenure to avoid title-chasers, possessing a current title aligned with engineering and a proven track record of hands-on production code deployment at a product company rather than pure research, consulting, or architecture. The ideal candidate must demonstrate Expert proficiency in Python (60 months experience), Expert proficiency in Vector Databases (36 months experience), and Expert proficiency in Machine Learning (60 months experience), prioritizing deep foundational systems over short-term AI wrappers. HIDDEN MATCH SIGNAL: Career history explicitly shows building ranking, search, or recommendation systems at scale. To ensure cultural alignment, we require a candidate who is a Highly responsive and active communicator, exhibits Exceptional follow-through and high reliability in professional commitments, and is Deeply passionate about technology with a highly active GitHub footprint."

    start_time = time.time()
    st_1_res = retriever.search(Query, top_k=200, recall_k=0.5)
    end_time = time.time()
    print(f"\nSearch completed in {end_time - start_time:.2f} seconds\n")

    cross_encoder_rerank_and_export(
        query=Query,
        bi_encoder_candidates=st_1_res,
        model_name=args.model,
        top_k=args.top_k,
        output_csv=args.out
    )
    start_time_last = time.time()
    print(f"Complete search and Reranking done in {start_time_last - start_time} seconds")

# "python rank.py --candidates ./candidates.jsonl --out ./submission.csv"