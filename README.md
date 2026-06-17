# Rolesync

## Project Overview
Traditional ATS systems rely on rigid keyword matching, causing recruiters to miss out on top talent who don't fit a standard mold. The Rolesync AI solves this by introducing a Two-Pass Pipeline that combines semantic vector search with deep LLM reasoning. It doesn't just match words - it understands the true intent of the job description and mathematically proves why a candidate fits.

## Components 

### JD (Job description) Intellegence 
Job descriptions often contain implied needs that aren't explicitly written as keywords.

To solve this : An LLM layer powered by Gemini 2.5 Flash and Pydantic that reads the job description and extracts the true essence of the role.

The LLM to output a strict JSON schema containing mandatory skills, preferred skills, and hidden requirements. Furthermore, it analyzes the tone of the JD to generate a dynamic Weight Vector across 5 core categories : 

* Technical
* Experience
* Leadership
* Learning
* Communication

Overall sum of weights equals 1.

### Broad Filtering via local VectorDB
We cannot afford to use heavy LLM reasoning on a dataset of hundreds of thousandsvof candidates simultaneously due to high costs and latency constraints.

This was tackled by A lightweight, local vector database ingestion pipeline using ```FAISS``` and Google's ```gemini-embedding-001``` model.


### Further Development
Hybrid Retreival system is under devlopment , the above sytems have been tested with good results as outcomes

