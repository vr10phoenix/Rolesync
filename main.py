import os
import sys
import argparse
import zipfile
import shutil     
import gdown

# symlink warnings disabled and directries
os.environ["TRANSFORMERS_CACHE"] = "models"
os.environ["HF_HOME"] = "models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from Index_builder import build_index
from retriever import CandidateRetriever, JD_QUERY
from cross_encoder import cross_encoder_rerank_and_export


def ensure_models_downloaded(bi_model_name: str, cross_model_name: str) -> None:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("[MAIN] Ensuring models are downloaded locally...", flush=True)
    try:
        _ = SentenceTransformer(bi_model_name)
        print("[MAIN] Bi-encoder model is available.", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to download/load bi-encoder model: {e}", flush=True)
        sys.exit(1)
    try:
        _ = CrossEncoder(cross_model_name)
        print("[MAIN] Cross-encoder model is available.", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to download/load cross-encoder model: {e}", flush=True)
        sys.exit(1)
    print("[MAIN] All models are ready.\n", flush=True)


def index_exists(index_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(index_dir, "candidates.faiss")) and
        os.path.exists(os.path.join(index_dir, "candidates_meta.jsonl"))
    )


def download_and_extract_index(url: str, out_dir: str = "index_database") -> bool:
    try:
        zip_path = "index_database_temp.zip"
        print(f"[MAIN] Downloading index from Google Drive...", flush=True)
        gdown.download(url, zip_path, quiet=False)

        if not zipfile.is_zipfile(zip_path):
            print(f"[ERROR] Downloaded file is not a valid zip archive.", flush=True)
            os.remove(zip_path)
            return False

        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_dir)
        os.remove(zip_path)

        # single top-level folder : move its contents up
        items = os.listdir(out_dir)
        if len(items) == 1 and os.path.isdir(os.path.join(out_dir, items[0])):
            subdir = os.path.join(out_dir, items[0])
            for f in os.listdir(subdir):
                src = os.path.join(subdir, f)
                dst = os.path.join(out_dir, f)
                shutil.move(src, dst)
            os.rmdir(subdir)

        # Verify required index files exist :
        if not (os.path.exists(os.path.join(out_dir, "candidates.faiss")) and
                os.path.exists(os.path.join(out_dir, "candidates_meta.jsonl"))):
            print("[ERROR] Index files not found after extraction.", flush=True)
            return False

        print(f"[MAIN] Index downloaded and extracted to {out_dir}", flush=True)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to download/extract index: {e}", flush=True)
        if os.path.exists("index_database_temp.zip"):
            os.remove("index_database_temp.zip")
        return False


def main():
    print("[MAIN] Starting pipeline...", flush=True)

    parser = argparse.ArgumentParser(
        description="Build FAISS index, retrieve candidates, rerank with cross‑encoder, and export CSV."
    )
    parser.add_argument("--candidates", required=True, help="Path to candidate data (JSON or JSONL).")
    parser.add_argument("--out", required=True, help="Output CSV file path.")
    parser.add_argument("--query", default=JD_QUERY, help="Job description query text.")
    parser.add_argument("--index-dir", default="index_database", help="Directory for index and metadata.")
    parser.add_argument("--download-index", default=None,
                        help="Google Drive share link or file ID to download a pre-built index ZIP.")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="Bi‑encoder model name.")
    parser.add_argument("--cross-model", default="BAAI/bge-reranker-base", help="Cross‑encoder model name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding.")
    parser.add_argument("--recall-k", type=int, default=200, help="Number of candidates from bi‑encoder.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of final candidates.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend weight for similarity.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index.")

    args = parser.parse_args()

    if not os.path.isfile(args.candidates):
        print(f"[ERROR] Candidates file not found: {args.candidates}", flush=True)
        sys.exit(1)

    ensure_models_downloaded(args.model, args.cross_model)

    # handle index
    if args.rebuild or not index_exists(args.index_dir):
        if args.download_index and not index_exists(args.index_dir):
            if download_and_extract_index(args.download_index, args.index_dir):
                print(f"[MAIN] Using downloaded index at {args.index_dir}", flush=True)
            else:
                print("[MAIN] Download failed; falling back to building index.", flush=True)
                build_index(
                    data_path=args.candidates,
                    out_dir=args.index_dir,
                    model_name=args.model,
                    batch_size=args.batch_size,
                )
        else:
            print(f"[MAIN] Building index from {args.candidates} ... (this may take a while)", flush=True)
            build_index(
                data_path=args.candidates,
                out_dir=args.index_dir,
                model_name=args.model,
                batch_size=args.batch_size,
            )
    else:
        print(f"[MAIN] Index already exists at {args.index_dir} – skipping build.", flush=True)

    print(f"[MAIN] Loading retriever from {args.index_dir} ...", flush=True)
    retriever = CandidateRetriever(args.index_dir, model_name=args.model)

    print(f"[MAIN] Retrieving top {args.recall_k} bi‑encoder candidates ...", flush=True)
    bi_candidates = retriever.search(
        query=args.query,
        top_k=args.recall_k,
        recall_k=args.recall_k,
        alpha=args.alpha,
        exclude_disqualified=True,
    )

    if not bi_candidates:
        print("[MAIN] No candidates passed the hard filters. Exiting.", flush=True)
        sys.exit(0)

    print(f"[MAIN] Reranking with cross‑encoder and exporting to {args.out} ...", flush=True)
    cross_encoder_rerank_and_export(
        query=args.query,
        bi_encoder_candidates=bi_candidates,
        model_name=args.cross_model,
        top_k=args.top_k,
        output_csv=args.out,
    )

    print("[MAIN] Pipeline finished successfully.", flush=True)


if __name__ == "__main__":
    main()