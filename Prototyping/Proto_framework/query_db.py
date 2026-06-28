import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# gemini api
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def search_candidates(query_text: str, top_k: int = 2):
    print(f"Loading Google Embedding Model...")
    # same model is MUST
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    print("FAISS database starting .....")
    try:
        vector_db = FAISS.load_local("faiss_candidate_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    #Perform the Semantic Search
    print(f"\nSearching for: '{query_text}'")
    print("-" * 50)
    
    # return k target candidates
    results = vector_db.similarity_search(query_text, k=top_k)
    
    #Display the Results
    for i, doc in enumerate(results):
        candidate_id = doc.metadata.get("candidate_id", "UNKNOWN ID")
        print(f"Match #{i + 1} | Candidate ID: {candidate_id}")
        # Print the first 150 characters of their profile to verify it worked
        print(f"Profile Snippet: {doc.page_content[:150]}...\n")

if __name__ == "__main__":
    #test with a sample
    sample_jd_query = """
    Looking for a strong Backend Engineer with experience in Python, SQL, and data pipelines. 
    Must know Apache Spark and Airflow. Ideally someone interested in transitioning toward ML.
    """
    
    search_candidates(sample_jd_query, top_k=20)