import json
import os
import dotenv
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

#Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def create_candidate_document(candidate_json: dict) -> Document:
    """Stitches structured JSON into a dense text document for vector search."""
    profile = candidate_json.get("profile", {})
    history = candidate_json.get("career_history", [])
    skills = candidate_json.get("skills", [])
    
    text_parts = [
        f"Title: {profile.get('headline', '')}",
        f"Summary: {profile.get('summary', '')}",
        f"Total Experience: {profile.get('years_of_experience', 0)} years"
    ]
    
    # Career history
    for job in history:
        text_parts.append(f"Role: {job.get('title')} at {job.get('company')} ({job.get('industry')}). Impact: {job.get('description')}")
        
    # skills of candidate
    skill_names = [f"{s.get('name')} ({s.get('proficiency')})" for s in skills]
    text_parts.append(f"Skills: {', '.join(skill_names)}")
    
    # Combine the data
    page_content = "\n".join(text_parts)
    
    # candiate ID as metadata
    metadata = {"candidate_id": candidate_json.get("candidate_id")}
    
    return Document(page_content=page_content, metadata=metadata)

# def build_faiss_index(jsonl_file_path: str, index_save_path: str):
#     print("Loading Google Embedding Model...")
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
#     documents = []
#     print(f"Reading candidates from {jsonl_file_path}...")
    
#     # reading jsonl file
#     with open(jsonl_file_path, "r", encoding="utf-8") as file:
#         for line in file:
#             if line.strip():
#                 candidate_data = json.loads(line)
#                 doc = create_candidate_document(candidate_data)
#                 documents.append(doc)
    
#     print(f"Embedding {len(documents)} candidates into FAISS... (This will be fast!)")
    
#     #FAISS index
#     vector_db = FAISS.from_documents(documents, embeddings)
    
#     #local save
#     vector_db.save_local(index_save_path)
#     print(f"Success! FAISS index saved to {index_save_path}")

# for sample_candidates.json testing as it is .json and not .jsonl
def build_faiss_index(json_file_path: str, index_save_path: str):
    print("Loading Google Embedding Model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    documents = []
    print(f"Reading candidates from {json_file_path}...")
    
    with open(json_file_path, "r", encoding="utf-8") as file:
        candidates_list = json.load(file) 
        
        for candidate_data in candidates_list:
            doc = create_candidate_document(candidate_data)
            documents.append(doc)
    
    print(f"Embedding {len(documents)} candidates into FAISS... (This will be fast!)")
    
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(index_save_path)
    print(f"Success! FAISS index saved to {index_save_path}")

if __name__ == "__main__":
    build_faiss_index("sample_candidates.json", "faiss_candidate_index")