from __future__ import annotations
import argparse
import json
import os
from datetime import date
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from document_builder import load_candidates, build_candidate_document
from constraints import evaluate_candidate
from Proto_framework_1.JD_sorting import M_query

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class _OfflineHashEmbedder:
    """
    backup embedder if sentence transformer cant secure an online connection 
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
        # Backup for emergency / internet error
        print(f"Waring : Could not load real embedding model '{model_name}' ({e}).\n"
              f"Falling back to a non-semantic offline hashing embedder. "
              f"Retrieval quality will be poor without internet access.")
        return _OfflineHashEmbedder()


def embed_texts(model, texts: list[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
    """
    text embedder
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


def build_index(data_path: str,out_dir: str,model_name: str = DEFAULT_MODEL,batch_size: int = 32,) -> None:

    os.makedirs(out_dir, exist_ok=True)
    candidates = load_candidates(data_path)
    print(f"Loaded {len(candidates)} candidates from {data_path}")

    docs = [build_candidate_document(c) for c in candidates]
    today = date.today()
    constraint_results = [evaluate_candidate(c, reference_date=today) for c in candidates]

    print(f"Loading embedding model: {model_name}")
    model = _load_embedding_model(model_name)

    texts = [d.text for d in docs]
    print(f"Embedding {len(texts)} documents...")
    vectors = embed_texts(model, texts, batch_size=batch_size, is_query=False)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"Built FAISS index: {index.ntotal} vectors, dim={dim}")

    faiss_path = os.path.join(out_dir, "candidates.faiss")
    faiss.write_index(index, faiss_path)
    print(f"Saved index : {faiss_path}")

    meta_path = os.path.join(out_dir, "candidates_meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for row_id, (doc, cres) in enumerate(zip(docs, constraint_results)):
            record = {
                "row_id": row_id,
                "candidate_id": doc.candidate_id,
                "text": doc.text,
                "metadata": doc.metadata,
                "constraint": {
                    "disqualified": cres.disqualified,
                    "disqualifier_reasons": cres.disqualifier_reasons,
                    "soft_score": cres.soft_score,
                    "soft_score_breakdown": cres.soft_score_breakdown,
                    "flags": cres.flags,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved metadata : {meta_path}")

    config_path = os.path.join(out_dir, "index_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, "dim": dim, "count": index.ntotal,
                   "metric": "inner_product_on_normalized_vectors"}, f, indent=2)
    print(f"Saved config :{config_path}")


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index for candidate JSON profiles.")

    parser.add_argument("--data", default="candidates.jsonl", help="Path to JSON/JSONL candidate file.")
    parser.add_argument("--out", default="index_database", help="Output directory for index + metadata.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name.")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    build_index(args.data, args.out, args.model, args.batch_size)


if __name__ == "__main__":
    main()