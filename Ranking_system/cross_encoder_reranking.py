import csv
import math
from typing import List , Any
from sentence_transformers import CrossEncoder
from retrival import CandidateRetriever
from Proto_framework_1.JD_sorting import M_query

def sigmoid(x :float) -> float:
    """
    Normalizes cross-Encoder logits to a 0-1 probability.
    """
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def cross_encoder_rerank_and_export(query:str , bi_encoder_candidates : List[Any] ,
                                    model_name : str = "BAAI/bge-reranker-base",
                                    top_k: int = 100 , output_csv: str = "result_1.csv"):

    model = CrossEncoder(model_name , max_length = 512)
    cross_inp = [[query , candidate.text_preview] for candidate in bi_encoder_candidates]

    raw_scores = model.predict(cross_inp)
    scored_candidates = []

    # Loop through and score candidates 
    for candidate , raw_score in zip(bi_encoder_candidates , raw_scores):
        normalized_score = sigmoid(float(raw_score))
        scored_candidates.append({
            "candidate_id" : candidate.candidate_id ,
            "name": candidate.name ,
            "score" : normalized_score
        })

    scored_candidates.sort(key = lambda x : x["score"] , reverse = True)

    # top k csndidates
    final_shortlist = scored_candidates[:top_k]

    print(f" exporting Top {len(final_shortlist)} candidates to {output_csv} ...")

    # export file , restricted to once 
    with open(output_csv , mode = 'w' , newline = '' , encoding = 'utf-8') as csv_file:
        fieldnames = ['candidate_id' , 'rank' , 'score']
        writer = csv.DictWriter(csv_file , fieldnames=fieldnames)

        writer.writeheader()
        for rank , candidate in enumerate(final_shortlist , start = 1):
            writer.writerow({
                "candidate_id" : candidate["candidate_id"],
                "rank" : rank,
                "score" : round(candidate["score"] , 4)
            })

    return final_shortlist


if __name__ == "__main__":
     retriever = CandidateRetriever("index_database")
     Refined_query = M_query
     st_1_res = retriever.search(Refined_query , top_k = 200 , recall_k = 0.5)

     results = cross_encoder_rerank_and_export(query = Refined_query , bi_encoder_candidates = st_1_res ,
                                     top_k = 100)

     for rank, r in enumerate(results, 1):
        print(f"\n#{rank} {r.name} ({r.candidate_id}) — final={r.final_score} "
              f"(sim={r.similarity}, soft={r.soft_score})")
        print(f"   title: {r.metadata.get('current_title')} @ {r.metadata.get('current_company')}")
        if r.flags:
            print(f"   flags: {r.flags}")
        print(f"   preview : {r.text_preview} ...")

     pass