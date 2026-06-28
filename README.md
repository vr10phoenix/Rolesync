# Jobsync : An Intelligent Multi-Stage Candidate Retrieval & Ranking Engine

> **Beyond keyword matching. Beyond embeddings.**
>
> A production-inspired, multi-stage retrieval architecture that understands candidate profiles, filters unsuitable applicants through deterministic reasoning, retrieves semantically relevant candidates using dense vector search, and finally performs deep relevance re-ranking using a Cross Encoder.

---

### Motivation

Modern recruitment systems still rely heavily on keyword overlap.

This causes several critical problems:

* Excellent candidates are ignored because they don't use the "right" keywords.
* Prompt-engineering or AI-hype resumes often rank above engineers with real production experience.
* Recruiters spend valuable time reviewing candidates that should have been filtered much earlier.

This project introduces an **enterprise-grade retrieval pipeline** inspired by modern Information Retrieval systems used at companies like Google, LinkedIn and OpenAI.

Instead of treating recruitment as a keyword search problem, we treat it as a **semantic retrieval + reasoning + ranking** problem.

---

## System Architecture

```
Candidate JSON Profiles
           │
           ▼
  Intelligent Document Builder
           │
           ▼
     Constraint Engine
     Rule Based Evaluation
           │
           ▼
   Dense Vector Index
   (FAISS + BGE Embeddings)
           │
  Recruiter Job Description
           │
           ▼
   Bi-Encoder Retrieval
   Fast Candidate Recall
           │
   Top-K Candidate Pool
           │
           ▼
     Cross Encoder
     Deep Re-ranking
           │
           ▼
 Final Ranked Candidate List

```

---

##  Pipeline Components

This project follows a **Two stage Retrival mechaninsm** :   
- Bi-encoder Retrival
- Cross encoder Rereanking

The pipeline was designed with these features to gurantee simple yet powerful and ultra - fast filtering and reranking even without use of LLMs.

---

### Intelligent Document Builder

Raw JSON profiles are poor retrieval documents.

The Document Builder converts every structured candidate profile into a rich natural-language document containing:

* Professional summary
* Career progression
* Skills ranked by evidence
* Education
* Certifications
* Languages
* Recruiter activity signals
* Availability metadata

Instead of embedding isolated fields, the system embeds a coherent human-readable career narrative.

This dramatically improves semantic retrieval quality.

---

### Constraint Engine

Before retrieval, every candidate passes through a deterministic reasoning engine.

Unlike LLMs, these rules are:

* Explainable
* Repeatable
* Auditable

The engine automatically detects patterns such as:

- Consulting-only career history
- Research-only profiles
- Prompt-engineering without production ML
- Job hopping
- Senior titles without coding evidence
- Relocation & visa constraints

Every decision produces recruiter-readable explanations instead of black-box scores.

---

### Vector Index Construction
Each enriched document is embedded using
 ```BAAI/bge-base-en-v1.5```
The embeddings are stored inside a FAISS Inner Product index.

During indexing the system stores

* Candidate metadata
* Constraint results
* Soft scores
* Ranking signals

This enables semantic retrieval without repeatedly evaluating business logic.

---

### Bi-Encoder Retrieval

Given a recruiter Job Description,

the system generates an embedding for the JD and performs approximate nearest-neighbor search over the FAISS index.

The retrieval score is combined with the Constraint Engine's soft score:

Final Score

= α × Semantic Similarity
* (1 − α) × Constraint Score

This allows the system to retrieve candidates that are not only semantically similar but also practically suitable.

---

### Cross Encoder Re-ranking

The Bi-Encoder optimizes **speed**.
The Cross Encoder optimizes **accuracy**.
The top retrieved candidates are re-scored using

> **BAAI/bge-reranker-base**

Instead of independently embedding the query and document,
the Cross Encoder jointly processes
(Job Description, Candidate Profile)
allowing token-level interactions between both texts.
This stage significantly improves ranking precision.

---

### Key Features

#### Intelligent Candidate Understanding

Converts structured profiles into semantically rich documents.

---

#### Explainable Constraint Filtering

Every rejection includes human-readable reasoning.

---

#### Hybrid Ranking

Combines

* semantic similarity
* business constraints
* recruiter signals

into one unified score.

---

#### Enterprise Retrieval Pipeline

Inspired by production Retrieval-Augmented Search systems.

---

#### Modular Design

Every stage is completely independent.

```
Document Builder

↓

Constraint Engine

↓

Index Builder

↓

Retriever

↓

Cross Encoder
```

Each module can be improved independently.

---

## Project Structure

```
project/

│
├── document_builder.py
├── constraints.py
├── Index_builder.py
├── retrival.py
├── cross_encoder_reranking.py
│
├── candidates.jsonl
├── index_database/
│   ├── candidates.faiss
│   ├── candidates_meta.jsonl
│   └── index_config.json
│
└── README.md
```

---

#  Technologies Used

| Component     | Technology                   |
| ------------- | ---------------------------- |
| Embeddings    | ```BAAI/bge-base-en-v1.5```               |
| Vector Search | FAISS                        |
| Re-ranking    | ```BAAI/bge-reranker-base\large```|
| Language      | Python                       |
| Retrieval     | Sentence Transformers        |
| Ranking       | Hybrid Semantic + Rule Based |

---

## Running the Pipeline

### Build Candidate Documents

```bash
python document_builder.py
```
---
### Evaluate Constraints & Build Index

```bash
python Index_builder.py
```
This generates

```
FAISS Index

Metadata

Constraint Scores

Configuration
```

---

### Retrieve Candidates
```bash
python retrival.py
```
Outputs the highest scoring semantic matches.

---

### Re-rank Results

```bash
python cross_encoder_reranking.py
```

Exports the final ranked shortlist.

---

## Results : 
### Cross encoder : ```BAAI/bge-reranker-base```
```
search completed in ```1.088270664215088``` seconds  
Loading Cross-Encoder Model: 'BAAI/bge-reranker-base'...  
Running Cross-Encoder prediction...  
completed in 199.18889999389648 seconds  

--- TOP 100 SHORTLISTED CANDIDATES ---  
Rank #1   | ID: CAND_0060072 | Name: Anil Mishra          | Score: 0.6691
Rank #2   | ID: CAND_0002025 | Name: Ira Dalal            | Score: 0.6388
Rank #3   | ID: CAND_0071974 | Name: Sai Verma            | Score: 0.6352
Rank #4   | ID: CAND_0006567 | Name: Aditya Subramanian   | Score: 0.6279
Rank #5   | ID: CAND_0088025 | Name: Amit Arora           | Score: 0.6150
Rank #6   | ID: CAND_0068351 | Name: Aadhya Iyer          | Score: 0.6073
Rank #7   | ID: CAND_0092278 | Name: Ananya Arora         | Score: 0.6039
Rank #8   | ID: CAND_0033861 | Name: Kabir Kapoor         | Score: 0.5971
Rank #9   | ID: CAND_0005538 | Name: Aryan Goyal          | Score: 0.5821
Rank #10  | ID: CAND_0081846 | Name: Arjun Khanna         | Score: 0.5795
Rank #11  | ID: CAND_0039754 | Name: Mira Banerjee        | Score: 0.5718
Rank #12  | ID: CAND_0094759 | Name: Aditya Pillai
....
```
### Cross encoder : ```BAAI/bge-reranker-large```
```
search completed in 0.9966781139373779 seconds  
Loading Cross-Encoder Model: 'BAAI/bge-reranker-large'...  
Running Cross-Encoder prediction...  
completed in 680.5309031009674 seconds  

--- TOP 100 SHORTLISTED CANDIDATES ---
Rank #1   | ID: CAND_0071974 | Name: Sai Verma            | Score: 0.7191
Rank #2   | ID: CAND_0002025 | Name: Ira Dalal            | Score: 0.7034
Rank #3   | ID: CAND_0081846 | Name: Arjun Khanna         | Score: 0.6863
Rank #4   | ID: CAND_0094759 | Name: Aditya Pillai        | Score: 0.6831
Rank #5   | ID: CAND_0006567 | Name: Aditya Subramanian   | Score: 0.6581
Rank #6   | ID: CAND_0005538 | Name: Aryan Goyal          | Score: 0.6553
Rank #7   | ID: CAND_0086022 | Name: Dhruv Naidu          | Score: 0.6453
Rank #8   | ID: CAND_0060072 | Name: Anil Mishra          | Score: 0.6429
Rank #9   | ID: CAND_0039754 | Name: Mira Banerjee        | Score: 0.6165
Rank #10  | ID: CAND_0088025 | Name: Amit Arora           | Score: 0.5900
Rank #11  | ID: CAND_0068351 | Name: Aadhya Iyer          | Score: 0.5887
Rank #12  | ID: CAND_0046064 | Name: Saanvi Naidu         | Score: 0.5641
Rank #13  | ID: CAND_0007411 | Name: Rahul Bansal         | Score: 0.5482
Rank #14  | ID: CAND_0092278 | Name: Ananya Arora
.......
```


###  Why Multi-Stage Retrieval?

Each model specializes in a different task.

| Stage             | Purpose               |
| ----------------- | --------------------- |
| Document Builder  | Better representation |
| Constraint Engine | Business reasoning    |
| Bi-Encoder        | High Recall           |
| Cross Encoder     | High Precision        |

Instead of asking one model to do everything,
the workload is distributed across specialized components.

This improves:

* Retrieval accuracy
* Explainability
* Speed 

---

### Dropping LLMs for Analysis
This was done as the output required is top 100 candidates and applying LLM analysis significantly loses the perfrmance in terms of speed.  
Cases where speed is not a vital parameter can be implemented by extending the components by adding a LLM analysis module.   
Considering constraints of not using GPU while fetching candidates and Hard ceiling of 300  seconds as provided by Redrob the current Architecture and mechanism seemed mathematically stable and powerful coming with a very high speed retrrival.

##  Future Improvements

* Hybrid BM25 + Dense Retrieval
* Learning-to-Rank (LTR)
* Multi-vector candidate representations
* Graph-based career reasoning
* Skill ontology expansion
* Recruiter feedback learning
* Personalized ranking models
* Multi-modal candidate understanding
* LLM-generated hiring explanations

---

## Design Philosophy

This project follows the same philosophy adopted in modern search systems:

> **Retrieve broadly. Filter intelligently. Rank precisely.**

Rather than replacing recruiter judgment, the system amplifies it by combining semantic understanding, deterministic business rules, and neural ranking into one explainable pipeline.

### what makes it differnet ?
#### Python logic
 used Python logic to read the JSON, find traps (job hoppers, pure consulting, low response rate), and wrote them into the document as text warnings.  
 **why different** : Using an LLM to figure out if someone is a job hopper takes 2 seconds and $0.01 per candidate. Using Python if/else logic takes 0.0001 seconds and $0.00. You dynamically translated structured database metrics (which semantic search normally ignores) into latent semantic text signals that the Cross-Encoder could actually "read."

#### True Two-Stage Retrieval
What you did: 0.9 seconds for FAISS (Bi-encoder), 199 seconds for Re-ranking (Cross-encoder).  
**why differnet** : 
- Stage 1 (FAISS): Casts a wide net. It calculates cosine similarity instantly (0.9s) but doesn't understand deep context.

- Stage 2 (Cross-Encoder): Highly computationally expensive. It reads the Query and the Document at the same time and performs self-attention between the words. It is incredibly accurate.

Finishing a 100-candidate Cross-Encoder evaluation on a purely local machine in 199 seconds without a GPU.

#### Asymmetric Penalty scoring
injecting warnings like "WARNING: Candidate is a 'Ghost Profile'" and "CURRENT REALITY WARNING"  

**Why differnet** : By putting those specific words into the text, you essentially "poisoned" the bad candidates mathematically. When the Cross-Encoder compared the Job Description against a profile containing the word "WARNING", it automatically drove the candidate's logit score down , Successfully bypassed the need for an LLM to act as a judge by forcing the embedding math to do the judging.

The result is a retrieval engine that is **fast, simple , Powerful ,scalable, interpretable, and production-ready**.
