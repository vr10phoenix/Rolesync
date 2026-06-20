import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def search_candidates_faiss(query_text: str, top_k: int = 20) -> list:
    """
    Searches the local FAISS database and returns the top K broad matches.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Loading FAISS database ...")
    try:
        vector_db = FAISS.load_local("faiss_candidate_index_1", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading datsbsde. Error: {e}")
        return []
    
    print(f"Searching FAISS for Top {top_k} matches...\n")
    results = vector_db.similarity_search(query_text, k=top_k)
    
    print(f"Retrieved {len(results)} candidates.\n")
    return results

if __name__ == "__main__":
    sample_jd = M_query
    docs = search_candidates_faiss(sample_jd, top_k=1000)
    for i, d in enumerate(docs):
        print(f"Match {i+1}: {d.metadata.get('candidate_id')}")