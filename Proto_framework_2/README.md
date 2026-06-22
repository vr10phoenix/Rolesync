## Components 

### Document builder and storage mechanism 
``` build_and_store.py```  
- This script transforms complex candidate schemas into context-rich, narrative-driven Synthetic Documents. 
- exposes behavioral signals, surfaces hidden alignment vectors for bi-encoders and cross-encoders about deceptive profiles. 

#### Core Logic  
- The Title Mismatch & Non-Tech Trap 
 - The builder intercepts the current_title field. If it contains non-technical departments, gives a critical warning string (CURRENT REALITY WARNING) ensuring the vector distance increases.

- The Ghost Profile Trap
 - last_active_date and recruiter_response_rate. Inactivity exceeding 180 days or response rates below 10% append a strict WARNING: Candidate is a 'Ghost Profile' string directly to the text payload.

- The Framework Enthusiast vs. Foundational Trap
 - Separates skills into Core Engineering Foundation and Recent AI Frameworks. If a candidate lists modern frameworks but does not possess at least 24 months of core foundational engineering experience (Python, SQL, Backend, etc.), a dedicated flag is appended to penalize downstream vector similarity scores.
- The Job-Hopper & Architecture Chaser Trap
 - Calculates moving averages of employment durations (flagging tenures under 18 months). It also sweeps the most recent role descriptions for explicit verbs (built, shipped, coded, implemented) if the candidate holds an executive title.

- Exposing Hidden Match Signals
 - Automatically scans narrative experience bodies for deep-indexing keywords and injects a prominent string: HIDDEN MATCH SIGNAL: Career history explicitly shows building ranking, search, or recommendation systems at scale.

### Architecture Flow 
```
[ Raw candidates.jsonl Record ]
               │
               ▼
   [ build_synthetic_document() ] ──► Applies Anti-Trap Heuristics & Signals
               │
               ▼
    [ LangChain Document Object ]  ──► Injects metadata={"candidate_id": ID}
               │
               ▼
[ sentence-transformers/all-MiniLM-L6-v2 ] (Local Embeddings Engine)
               │
               ▼
[ faiss_candidate_index_modified ] ──► Saved locally for millisecond retrieval
```

#### Execution Ingestion : 
```
python build_and_store.py
```  

#### Prerequisities & system dependenciesa  
```
langchain-core
langchain-huggingface
langchain-community
faiss-cpu (or faiss-gpu for rapid local scaling)
sentence-transformers
```









### Bi-encoder Retrival 
#### Vector search ```search.py```
Uses an optimized symmetrical embedding representation model to map incoming search queries into a localized vector space, running an ultra-fast similarity comparison against pre-indexed synthetic documents in milliseconds.

#### Architecture 
```
        Refined Super Query (M_query) 
                      │
                      ▼
 sentence-transformers/all-MiniLM-L6-v2 Embedder 
                      │
                      ▼
    FAISS Database Symmetrical Scan  ──► (Loads 'faiss_candidate_index_modified')
                      │
                      ▼
     Local Similarity Search Space  ──► Executes L2 / Cosine Matrix Sweeps
                      │
                      ▼
  Top-K Shortlisted LangChain Documents ──► Extracts Top 1,000 Matches with Metadata
```

#### execution
````
# Execution block within the pipeline runtime
if __name__ == "__main__":
    # Ingest the context-clean query produced by the Phase 0 JD Intelligence Engine
    sample_jd = M_query
    
    # Run broad vector filtering down to the top matching candidate pool
    docs = search_candidates_faiss(sample_jd, top_k=1000)
    
    # Stream match validation indicators to stdout
    for i, d in enumerate(docs):
        print(f"Match {i+1}: {d.metadata.get('candidate_id')}")
````

#### Output expectation
[STAGE 1] Initializing Local Embeddings Registry...
[STAGE 1] Loading local FAISS database footprint...
[STAGE 1] Database successfully verified and bound with deserialization permission.
[STAGE 1] Scanning candidate index for Top 1000 candidates...

Match 1: C_94821
Match 2: C_10432
Match 3: C_43012
Match 4: C_88321
Match 5: C_00294










### Cross Encoder Re-reanking
#### Precision Filter & Re-Ranking :```cross_encoder.py```
It feeds the coarse candidates from the FAISS database into an Cross-Encoder network (```BAAI/bge-reranker-base```). By processing the refined query and the candidate’s rich synthetic document simultaneously, the model captures deep, non-linear contextual interactions and semantic alignments that standard vector search misses entirely.

#### Logit-to-Probability Transformation (Sigmoid Math)
Raw logits from Cross-Encoders are unbounded, frequently presenting as negative integers or numbers greater than 1.0 depending on the model's structural design. To establish a standardized score suitable for ATS tracking platforms, the code runs each raw output through a Sigmoid Activation Function:
 $$\text{Normalized Score} = \frac{1}{1 + e^{-\text{logit}}}$$ 
 This maps the abstract alignment value into a uniform, human-readable range between ```0.0``` (pure mismatch) and ```1.0``` (pure match).

#### Dynamic Reasoning Engine
Instead of outputting  raw math floats, the script dynamically unpacks structured LangChain document metadata to construct real-time humanized profile summaries (dynamic_reasoning) explicitly tracking:

- Candidate Title
- Absolute Years of Experience (years_of_experience)
- Quantifiable core skills density counts (ai_core_skills)
- Real-time behavioral interaction metric bounds (response_rate)

#### component workflow 
```
 Stage 1: Top 200 FAISS Candidates 
                 │
                 ▼
     Pair Compilation Loop  ──► Pair Form: [Query, Synthetic Page Content]
                 │
                 ▼
  BAAI/bge-reranker-base Model  ──►  Computes joint attention matrix (Batch Size = 10)
                 │
                 ▼
    Sigmoid Normalization Math  ──► Normalizes raw logits securely into 0.0 -> 1.0 Range
                 │
                 ▼
      In-Memory Priority Sort  ──► Sorts down strictly by Descending Score
                 │
                 ▼
     Structured CSV Export Engine  ──► Serializes data cleanly into `result.csv`
```

#### Execution & Integration Trace 
```
# Execution code snippet within the core architecture pipeline
query = M_query # comes from JD via LLM refining

print("\nSTARTING TWO-STAGE RETRIEVAL PIPELINE")

# Stage 1: Broad filtering (Top recall net limit set safely to 200)
top_1000_docs = search_candidates_faiss(query, top_k=200)

# Stage 2: In-depth Cross-Encoder analysis and local file compilation
cross_encoder_rerank_and_export(query, stage_1_results=top_1000_docs, top_k=100)
```

#### Output Expectation : 
```
STARTING TWO-STAGE RETRIEVAL PIPELINE

[STAGE 1] Loading Local HuggingFace Embedding Model...
[STAGE 1] Loading local FAISS database ('faiss_candidate_index_modified')...
[STAGE 1] Searching FAISS for Top 200 broad matches...
[STAGE 1] Success! Retrieved candidates from index.

[STAGE 2] Running Cross-Encoder Re-Ranking...
Reranking Pairs: 100%|███████████████████████████| 20/20 [00:04<00:00,  4.20it/s]
[STAGE 2] Re-ranking Complete! Filtered down to Top 100.
--------------------------------------------------
[EXPORT] Writing final shortlist to result.csv...

Rank #1 | ID: C_94821 | Score: 0.9412 | Lead AI Engineer with 6.5 yrs; 14 AI core skills; response rate 0.89.
Rank #2 | ID: C_10432 | Score: 0.9105 | MLOps Specialist with 5.0 yrs; 9 AI core skills; response rate 0.95.
Rank #3 | ID: C_43012 | Score: 0.8874 | Senior Backend Engineer with 8.2 yrs; 12 AI core skills; response rate 0.78.
```

#### Export Deliverable Sechma : ```result.csv```
COLUMN NAME | DATA TYPE | DESCRIPTION 
candidate_id | ```String/Integer``` | Unique platform tracking token 
rank | ```Integer``` | Absolute ranking position (1 to 100).
score | ```Float``` | Sigmoid-normalized matching score ($0.0 \rightarrow 1.0$).
reasoning | ```String``` | Human-readable profile summary


