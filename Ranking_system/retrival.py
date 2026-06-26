import os
import argparse
import json
import faiss
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from __future__ import annotations
from Index_builder import embed_texts, _load_embedding_model, DEFAULT_MODEL
from Proto_framework_1.JD_sorting import M_query


class RankedCandidate:
    candidate_id: str
    name: str
    similarity: float
    soft_score: float
    final_score: float
    disqualified: bool
    disqualifier_reasons: list[str]
    flags: list[str]
    soft_score_breakdown: dict[str, float]
    metadata: dict[str, Any]
    text_preview: str = ""


class CandidateRetriever:
    def __init__(self, index_dir: str, model_name: str = DEFAULT_MODEL):

        self.index_dir = index_dir
        config_path = os.path.join(index_dir, "index_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            model_name = cfg.get("model_name", model_name)

        self.index = faiss.read_index(os.path.join(index_dir, "candidates.faiss"))
        self.records: list[dict] = []
        with open(os.path.join(index_dir, "candidates_meta.jsonl"), encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

        self.model_name = model_name
        self.model = _load_embedding_model(model_name)

    def search( self, query: str, top_k: int = 10, recall_k: int = 50, alpha: float = 0.5,
                exclude_disqualified: bool = True,min_similarity: float = 0.0,) -> list[RankedCandidate]:
        # Bi-encoder Search
        recall_k = max(recall_k, top_k)
        q_vec = embed_texts(self.model, [query], is_query=True)
        sims, idxs = self.index.search(q_vec, recall_k)
        sims, idxs = sims[0], idxs[0]

        candidates: list[RankedCandidate] = []
        for sim, idx in zip(sims, idxs):
            if idx == -1:
                continue
            record = self.records[idx]
            if sim < min_similarity:
                continue
            constraint = record["constraint"]
            if exclude_disqualified and constraint["disqualified"]:
                continue

            soft_score = constraint["soft_score"]
            final_score = alpha * float(sim) + (1 - alpha) * soft_score

            candidates.append(
                RankedCandidate(
                    candidate_id=record["candidate_id"],
                    name=record["metadata"].get("name", "Unknown"),
                    similarity=round(float(sim), 4),
                    soft_score=soft_score,
                    final_score=round(final_score, 4),
                    disqualified=constraint["disqualified"],
                    disqualifier_reasons=constraint["disqualifier_reasons"],
                    flags=constraint["flags"],
                    soft_score_breakdown=constraint["soft_score_breakdown"],
                    metadata=record["metadata"],
                    text_preview=record["text"][:300].replace("\n", " "),
                )
            )

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates[:top_k]


JD_QUERY = M_query


def main():
    parser = argparse.ArgumentParser(description="Query the candidate FAISS index.")
    parser.add_argument("--index-dir", default="index_database")
    parser.add_argument("--query", default=JD_QUERY)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)

    args = parser.parse_args()
    retriever = CandidateRetriever(args.index_dir)
    results = retriever.search(args.query, top_k=args.top_k, alpha=args.alpha)

    for rank, r in enumerate(results, 1):
        print(f"\n#{rank} {r.name} ({r.candidate_id}) — final={r.final_score} "
              f"(sim={r.similarity}, soft={r.soft_score})")
        print(f"   title: {r.metadata.get('current_title')} @ {r.metadata.get('current_company')}")
        if r.flags:
            print(f"   flags: {r.flags}")
        print(f"   preview: {r.text_preview}...")


if __name__ == "__main__":
    main()