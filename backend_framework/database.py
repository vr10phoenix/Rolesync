import json
import os
from langchain_core.documents import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Gemini api key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def build_synthetic_document(candidate: dict) -> str:
    """
    Transforms the structured JSON schema into a dense, semantically rich 
    text document optimized for Bi-Encoder and Cross-Encoder ranking.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    doc_parts = []
    
    # Identity txt
    doc_parts.append(f"Candidate Title: {profile.get('current_title')} at {profile.get('current_company')}")
    doc_parts.append(f"Headline: {profile.get('headline')}")
    doc_parts.append(f"Total Experience: {profile.get('years_of_experience')} years in the {profile.get('current_industry')} industry.")
    doc_parts.append(f"Summary: {profile.get('summary')}")
    
    # Tracing Career growth
    doc_parts.append("\nCareer Trajectory and Experience")
    if history:
        for role in history:
            title = role.get("title", "Unknown Role")
            company = role.get("company", "Unknown Company")
            duration = role.get("duration_months", 0)
            desc = role.get("description", "")
            
            doc_parts.append(f"Worked as {title} at {company} for {duration} months.")
            doc_parts.append(f"Impact & Responsibilities: {desc}")
            
    # track relevent skills:
    doc_parts.append("\n Skillset of Candidate")
    skill_strings = []
    for skill in skills:
        name = skill.get("name")
        prof = skill.get("proficiency", "unknown")
        months = skill.get("duration_months", 0)
        skill_strings.append(f"{prof.capitalize()} proficiency in {name} ({months} months experience).")
    
    if skill_strings:
        doc_parts.append(" ".join(skill_strings))
        
    # Tracing Behaviour traits : 
    doc_parts.append("\n Behavioural context")
    
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate > 0.8:
        doc_parts.append("Highly responsive and active communicator.")
    elif response_rate < 0.3:
        doc_parts.append("Passive candidate; slower communication style.")
        
    interview_rate = signals.get("interview_completion_rate", 0)
    if interview_rate > 0.85:
        doc_parts.append("Exceptional follow-through and high reliability in professional commitments.")
        
    github_score = signals.get("github_activity_score", -1)
    if github_score > 70:
        doc_parts.append("Deeply passionate about technology with a highly active GitHub footprint.")
        
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        top_skills = [k for k, v in assessments.items() if v > 80]
        if top_skills:
            doc_parts.append(f"Platform-verified top performer in: {', '.join(top_skills)}.")

    # append all to single string/txt
    return "\n".join(doc_parts)

def build_faiss_index(jsonl_filepath: str):
    print("Loading candidate dataset and generating synthetic documents...")
    langchain_documents = []

    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                
                candidate_data = json.loads(line)
                candidate_id = candidate_data.get("candidate_id", "UNKNOWN")
                
                # getting the sting
                synthetic_text = build_synthetic_document(candidate_data)
                
                # Packing into Langchain document object
                doc = Document(
                    page_content=synthetic_text,
                    metadata={"candidate_id": candidate_id}
                )
                langchain_documents.append(doc)
    except FileNotFoundError:
        print(f"Error: Could not find '{jsonl_filepath}'. Make sure the file exists in this directory.")
        return

    print(f"Successfully synthesized {len(langchain_documents)} candidate profiles.")
    
    # Initialize Google's foundational embedding model
    # print("Initializing Google Embedding Model (gemini-embedding-001)...")  
    # embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001") 
    # dropped google embedings due to rate limit constrain

    # Local HuggingFace embedding model
    print("Initializing embedding ....")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # creating faiss index
    print("Embedding candidates into vector db")
    vector_db = FAISS.from_documents(langchain_documents, embeddings)
    
    # saving files locally
    print("Saving FAISS index locally...")
    vector_db.save_local("faiss_candidate_index_1")
    print("🎉 Success! Vector DB built and saved to disk.")

if __name__ == "__main__":
    #calling the funtion
    build_faiss_index("candidates.jsonl")