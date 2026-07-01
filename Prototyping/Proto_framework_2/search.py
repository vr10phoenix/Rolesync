import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Prototying.Proto_framework_1.JD_sorting import M_query

def search_candidates_faiss(query_text: str, top_k: int = 20) -> list:
    """
   Bi - encoder retriveal
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vector_db = FAISS.load_local("faiss_candidate_index_modified", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading database. Error: {e}")
        return []

    results = vector_db.similarity_search(query_text, k=top_k)
    return results

if __name__ == "__main__":
    # refined query input
    sample_jd = M_query
    docs = search_candidates_faiss(sample_jd, top_k=1000)
    for i, d in enumerate(docs):
        print(f"Match {i+1}: {d.metadata.get('candidate_id')}")