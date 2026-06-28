# Intitial Architecture Design
![System Architecture](https://github.com/vr10phoenix/Rolesync/blob/main/Assets/initial_framework.png)

Initial Framework finalised : 
* Consists of 3 stage filtering
* Refinement of Job description and Cross encoder reranking to achieve better shortlising

## Design_01 : Candidate Retrieval System - FAISS pipeline

### Components

A retrieval system for ranking tens of thousands candidate JSON profiles against a job
First model was built to test the working of a typical retrival system:

```
query_db.py         -> search function , applies semantic serch to the faiss database.
cross_encoder.py      -> provides re-ranking the shortlist to refine the filtering. model : BAAI/bge-reranker-large.
LLM_analysis.py       -> Performs deep analysis and gives deep analysis for final selected list of candidates
Main_pipeline             -> Pipeline function performing the components sequentially.

```

### Why this design
Initially best Architecture for the retriveal system.  
Simple and yet robust design providing strong and sound results.

### FAISS database
embedding model used : ```gemini-embedding-001``` for generating embeddings and FAISS from ```from langchain_community.vectorstores import FAISS``` to store embeddinds as Vector database.

### Two-stage retrieval (bi-encoder recall -> cross-encoder rerank)

Bi-encoder embed the JD and each candidate **separately**. The model used : ```gemini-embedding-001```
comparison is just a fast vector similarity — scalable to 100K+ candidates,
but the model never reads JD and candidate text together.

Cross-encoders (Stage 2, `cross_encoder_rerank.py`) feed `(JD, candidate)`
as a single joint input and output one relevance score — much more
accurate, but ~100-1000x slower per pair, so it's only run over the
Stage-1 shortlist, not the full pool.

