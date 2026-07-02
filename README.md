# RoleSync : An Intelligent Multi-Stage Candidate Retrieval & Ranking Engine

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

![System Architecture](https://github.com/vr10phoenix/Rolesync/blob/main/Assets/system_architecture.png)

### Workflow
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
### Index Building Work-flow
![Index Building Woekflow](https://github.com/vr10phoenix/Rolesync/blob/main/Assets/index_architecture.png)
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

├── download_candidates.py
├── document_builder.py
├── constraints.py
├── Index_builder.py
├── retriever.py
├── cross_encoder_reranking.py
├── main.py
|
├── models/
│    ├── BAAI/bge-reranker-base
│    ├── BAAI/bge-base-en-v1.5
|
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

## Use cases 
Command‑line Arguments
| Argument        | Type | Default                     | Description                                      |
|-----------------|------|-----------------------------|--------------------------------------------------|
| --candidates    | str  | required                    | Path to the candidate JSONL file.                |
| --out           | str  | required                    | Output CSV file path.                            |
| --query         | str  | (predefined JD)             | Job description query text.                      |
| --index-dir     | str  | index_database              | Directory to store/read the FAISS index.         |
| --download-index| str  | None                        | Google Drive link or file ID to download a pre‑built index ZIP. |
| --model         | str  | BAAI/bge-base-en-v1.5       | Bi‑encoder model name.                           |
| --cross-model   | str  | BAAI/bge-reranker-base      | Cross‑encoder model.                             |
| --batch-size  | int  | 32      | Batch size for embedding generation.                          |
| --recall-k    | int  | 200     | Number of candidates retrieved by the bi‑encoder (pool for reranking). |
| --top-k       | int  | 100     | Number of final candidates to output after reranking.         |
| --alpha       | float| 0.5     | Blend weight: α * similarity + (1‑α) * soft_score.            |
| --rebuild     | flag | False   | Force rebuild of the index, even if it already exists.        |

## Running the Pipeline
Make sure you are in the Root Directory  

Install the dependencies : 
```
# install dependencies
pip install -r requirements.txt
```

Download the dataset:
```
<span style="color:green">python</span>  download_candidates.py
```

Once the dataset is downloaded -> 2 ways : 
FIRST >
```
# Downloads Pre-computed Candidate database , ready to use index to test it without
# making one which takes long to construct -> (RECOMMENDED APPROACH)
<span style="color:green">python</span> -u main.py --candidates ./candidates.jsonl --out submission.csv --download-index "https://drive.google.com/uc?export=download&id=1zpIrxjRwCNYw0Eo4CpCUcmqTWFXylzYU"
```
Use this command to run the Ranking system with a ready to use index , test it staright away

SECOND > 
(NOT RECOMMENDED) build Index from scratch and run the ranking , only when you want to explicity construct the index which is not needed , ready-to-use index already is the result of this command
```
# If explicitly need to build another index
python -u main.py --candidates ./candidates.jsonl --out submission.csv --download-index "https://drive.google.com/uc?export=download&id=1zpIrxjRwCNYw0Eo4CpCUcmqTWFXylzYU"
```

That's it !! 

**NOTE** : **The pipeline uses Internet on the first run to Download the Dataset , ranking and retrival models and Pre-built Candidate database.**
**The Ranking systems starts just after this and during the whole ranking process there are no external API calls , strictly tested and Verified.**
**After first run onwards , the entire system will run without making any API calls or simple without connecting to Internet.**

---

## Results : 
### Cross encoder : ```BAAI/bge-reranker-base```
```
[MAIN] Starting pipeline...
[MAIN] Ensuring models are downloaded locally...
modules.json: 100%|██████████████████████████████████████████████████████████████████████████████| 349/349 [00:00<?, ?B/s]
config_sentence_transformers.json: 100%|█████████████████████████████████████████████████████████| 124/124 [00:00<?, ?B/s]
README.md: 100%|█████████████████████████████████████████████████████████████████████| 94.6k/94.6k [00:00<00:00, 50.7MB/s]
sentence_bert_config.json: 100%|███████████████████████████████████████████████████████████████| 52.0/52.0 [00:00<?, ?B/s]
config.json: 100%|███████████████████████████████████████████████████████████████████████████████| 777/777 [00:00<?, ?B/s]
model.safetensors: 100%|███████████████████████████████████████████████████████████████| 438M/438M [00:48<00:00, 9.00MB/s]
Loading weights: 100%|████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 1776.57it/s]
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████| 366/366 [00:00<00:00, 358kB/s]
vocab.txt: 100%|███████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 3.20MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████| 711k/711k [00:00<00:00, 7.80MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████| 125/125 [00:00<00:00, 125kB/s]
config.json: 100%|███████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<?, ?B/s]
[MAIN] Bi-encoder model is available.
config.json: 100%|███████████████████████████████████████████████████████████████████████████████| 799/799 [00:00<?, ?B/s]
model.safetensors: 100%|█████████████████████████████████████████████████████████████| 1.11G/1.11G [03:01<00:00, 6.14MB/s]
Loading weights: 100%|████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 3772.09it/s]
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████| 443/443 [00:00<00:00, 445kB/s]
sentencepiece.bpe.model: 100%|███████████████████████████████████████████████████████| 5.07M/5.07M [00:03<00:00, 1.49MB/s]
tokenizer.json: 100%|████████████████████████████████████████████████████████████████| 17.1M/17.1M [00:03<00:00, 5.25MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████| 279/279 [00:00<00:00, 279kB/s]
[MAIN] Cross-encoder model is available.
[MAIN] All models are ready.

[MAIN] Downloading index from Google Drive...
Downloading...
From (original): https://drive.google.com/uc?id=1zpIrxjRwCNYw0Eo4CpCUcmqTWFXylzYU
From (redirected): https://drive.google.com/uc?id=1zpIrxjRwCNYw0Eo4CpCUcmqTWFXylzYU&confirm=t&uuid=215aad84-acb5-4367-8c67-1605ab333903
To: C:\Users\P Vardhan Reddy\Projects\RedDrob\index_database_temp.zip
100%|██████████████████████████████████████████████████████████████████████████████████| 349M/349M [01:05<00:00, 5.33MB/s]
[MAIN] Index downloaded and extracted to index_database
[MAIN] Using downloaded index at index_database
```
**Ranking starts and whole system operates without Internet from here onwards**
```
[MAIN] Loading retriever from index_database ...
Loading weights: 100%|████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 1745.63it/s]
[MAIN] Retrieving top 200 bi‑encoder candidates ...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.83s/it]
[MAIN] Reranking with cross‑encoder and exporting to submission.csv ...

[STAGE 2] Loading Cross-Encoder Model: 'BAAI/bge-reranker-base'...
Loading weights: 100%|████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 1531.94it/s]
[STAGE 2] Running Cross-Encoder prediction...
completed in 36.27 seconds


[STAGE 2] --- TOP 100 SHORTLISTED CANDIDATES ---
Rank #1   | ID: CAND_0060072 | Name: Anil Mishra          | Score: 1.0000
Rank #2   | ID: CAND_0071974 | Name: Sai Verma            | Score: 0.8956
Rank #3   | ID: CAND_0002025 | Name: Ira Dalal            | Score: 0.8944
Rank #4   | ID: CAND_0006567 | Name: Aditya Subramanian   | Score: 0.8792
Rank #5   | ID: CAND_0061257 | Name: Advaith Pillai       | Score: 0.7536
Rank #6   | ID: CAND_0094759 | Name: Aditya Pillai        | Score: 0.7415
Rank #7   | ID: CAND_0081846 | Name: Arjun Khanna         | Score: 0.7093
Rank #8   | ID: CAND_0077337 | Name: Aarav Agarwal        | Score: 0.7056
Rank #9   | ID: CAND_0068351 | Name: Aadhya Iyer          | Score: 0.6886
Rank #10  | ID: CAND_0039754 | Name: Mira Banerjee        | Score: 0.6724
Rank #11  | ID: CAND_0037980 | Name: Reyansh Chowdary     | Score: 0.6683
Rank #12  | ID: CAND_0092278 | Name: Ananya Arora         | Score: 0.6564
Rank #13  | ID: CAND_0086022 | Name: Dhruv Naidu          | Score: 0.6528
Rank #14  | ID: CAND_0005538 | Name: Aryan Goyal          | Score: 0.6488
Rank #15  | ID: CAND_0008425 | Name: Myra Krishnan        | Score: 0.6350
Rank #16  | ID: CAND_0046064 | Name: Saanvi Naidu         | Score: 0.6273
Rank #17  | ID: CAND_0080766 | Name: Kiara Mittal         | Score: 0.6252
Rank #18  | ID: CAND_0007411 | Name: Rahul Bansal         | Score: 0.6216
Rank #19  | ID: CAND_0093547 | Name: Arnav Ghosh          | Score: 0.6152
Rank #20  | ID: CAND_0093193 | Name: Aarohi Bose          | Score: 0.6140
......
Rank #99  | ID: CAND_0050504 | Name: Rajesh Ghosh         | Score: 0.6000
Rank #100 | ID: CAND_0087630 | Name: Aisha Rao            | Score: 0.6000
----------------------------------------------------------------------

[STAGE 2] Exporting Top 100 candidates to 'submission.csv' ...
[STAGE 2] Success! Pipeline complete.
```
**OUTPUT** : A CSV containing list of candidates after reranking.

### Cross encoder : ```BAAI/bge-reranker-large```
#### **(EXTRA : NOT FOR HACKATHON)**
Was done during the development of system , yeilded better results but re-reanking time exceed limit : 300s (completed in 680.5 seconds)  
bge-ranker-large can be used by making necessary changes in ```--cross-model```
For even better results I shall recommend using this re-ranker 
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
