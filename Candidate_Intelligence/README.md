# Candidate Intelligence

## ```vector_db.py``` 
his script is for the data ingestion. It reads structured candidate profiles from a JSON\l file, converts them into dense, searchable text documents, generates vector embeddings using Google's gemini model, and builds a local FAISS index.


## ``` Query_db.py ```
This script performs a semantic similarity search on a local vector database of candidate profiles. It uses Google's gemini embeddings to match a job description (or search query) with the most relevant candidates stored in a FAISS index.


## result : 
The semantic search is ready and working flawlessly 

OUTPUT : 
Searching for: '
    Looking for a strong Backend Engineer with experience in Python, SQL, and data pipelines. 
    Must know Apache Spark and Airflow. Ideally someone interested in transitioning toward ML.
    '
--------------------------------------------------
Match #1 | Candidate ID: CAND_0000001
Profile Snippet: Title: Backend Engineer | SQL, Spark, Cloud
Summary: Software / data professional with 6.9 years of experience building data pipelines, backend system...

