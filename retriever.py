from __future__ import annotations
import os
import sys
os.environ["TRANSFORMERS_CACHE"] = "models"
os.environ["HF_HOME"] = "models"
import argparse
import json
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import faiss

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class _OfflineHashEmbedder:
    """
    offline embedder , not used in pipeline
    """

    DIM = 384

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               normalize_embeddings=True, convert_to_numpy=True):
        import hashlib
        vectors = np.zeros((len(texts), self.DIM), dtype="float32")
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vectors[i, h % self.DIM] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors


def _load_embedding_model(model_name: str):
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model '{model_name}': {e}")
        print("[ERROR] Make sure you have an internet connection for the first run to download the model.")
        print("[ERROR] Alternatively, place the model in the 'models/' cache directory.")
        sys.exit(1)


def embed_texts(model, texts: list[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
    """
    Encodes texts to L2-normalized float32 vectors (so inner product == cosine similarity).
    """
    if is_query:
        texts = [QUERY_INSTRUCTION + t for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings).astype("float32")

@dataclass
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
        self.model.to('cpu')

    def search(self,query: str,top_k: int = 10,recall_k: int = 50,
        alpha: float = 0.5,exclude_disqualified: bool = True,min_similarity: float = 0.0,
    ) -> list[RankedCandidate]:
        """
        query: refined job description
        top_k: how many final results to return.
        recall_k: how many candidates to pull from FAISS before filtering/reranking
                  (must be >= top_k; higher gives the rerank stage more to work with).
        alpha: weight on semantic similarity vs rule-based soft_score in final ranking.
        """
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



def main():
    parser = argparse.ArgumentParser(description="Query the candidate FAISS index.")
    parser.add_argument("--index-dir", default="index_database")
    parser.add_argument("--query", default=JD_QUERY)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)

    args, _ = parser.parse_known_args()

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

# python -u main.py --candidates ./candidates.jsonl --out submission.csv --download-index "https://drive.google.com/uc?export=download&id=1zpIrxjRwCNYw0Eo4CpCUcmqTWFXylzYU"

JD_QUERY ="We require a dedicated Senior AI Engineer with 5 to 9 years of stable tenure who actively ships hands-on production code at a product company, strictly maintaining alignment between their current engineering title and continuous technical execution. The candidate bypasses short-term AI hype in favor of deep foundational expertise, demonstrating Expert proficiency in Python (84 months experience), Advanced proficiency in Vector Databases (36 months experience), and Expert proficiency in Machine Learning Systems (84 months experience). HIDDEN MATCH SIGNAL: Career history explicitly shows building ranking, search, or recommendation systems at scale. As a scrappy shipper with a sub-30 day notice period, they must be a Highly responsive and active communicator. with Exceptional follow-through and high reliability in professional commitments. Furthermore, the ideal founding team member is Deeply passionate about technology with a highly active GitHub footprint."