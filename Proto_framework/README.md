# Intitial Architecture Design
![System Architecture](Assets/initial_framework.png)

Initial Framework finalised : 
* Consists of 3 stage filtering
* Refinement of Job description and Cross encoder reranking to achieve better shortlising

## Components
### ```query_db.py```
searchs for the sample description on the faiss index.
Embedding model : ```gemini-embedding-001```  
Faster and efficient for intial dataset embeddings.


### ```cross_encoder.py```
re-ranks the result list from the semantic search to shortlist further.  

encoder used : ```BAAI/bge-reranker-large```  
Result : user defined ```K``` shortlisted candidates.

### ```LLM Analysis.py```
takes the output from encoder and  peform final deep analysis for final selection of candidates.
Candidates list sent in batches to be under the rate limit for LLM and sorts out n required candidates out of the results.  
Reasoning Model : ```Gemini 2.5 Flash``` for deeper dive into analysis to find best fit.


### ```Main pipeline.py```
A Function that integrates all the components:


### Results
```python main_pipeline.py```  

output :   

RANK #1 | Candidate: CAND_0000042 | Score: 97/100  
Recruiter Justification:
This candidate is a standout, possessing all the core technical skills: Python, SQL, and Spark, with proven experience building "massive DAGs in Airflow" for data pipelines. Their 5 years of experience and two internal promotions highlight a strong career trajectory, fast learning ability, and read...

RANK #2 | Candidate: CAND_0000001 | Score: 94/100  
Recruiter Justification:  
This candidate is an exceptional fit, having 6.9 years as a Backend Engineer with extensive experience in SQL, Spark, and building data pipelines. Their promotion to Lead Backend Engineer demonstrates strong leadership potential and a fast-learner mentality, aligning perfectly with the role's growth expectations...

RANK #3 | Candidate: CAND_0000010 | Score: 75/100  
Recruiter Justification:  
As a Data Engineer with 4 years of experience in Python and Airflow, this candidate meets several key technical requirements for data pipelines. While they lack explicit Spark experience and their career trajectory shows a static role,....

## Further Development 
Semantic search and Hybrid Retrival system have been implemented and tested with no flaws.

### behavioural intellegence: 
* Impelement behavioural intellegence in smeantic search makind use of candidates schema
* Career Trajectory Intellegence for tracking and weighing the Career growth in selection
* Polishing main pipeline and it components 
* Establish semantic search for whole dataset (was sample dataset till now).
