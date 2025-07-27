# Round 1B: Persona-Driven Document Intelligence

## Overview
Extracts and prioritizes relevant sections from PDFs based on persona and job-to-be-done, using Round 1A outlines and lightweight NLP models.

## Approach
Uses semantic analysis with all-MiniLM-L6-v2 (~80MB) for relevance scoring, supplemented by TF-IDF keyword matching. Leverages Round 1A outlines for efficient section extraction. Processes documents in parallel to meet 60-second constraint.

## Libraries Used
- PyMuPDF: PDF text extraction
- sentence-transformers: Semantic similarity
- scikit-learn: TF-IDF and cosine similarity
- numpy: Numerical computations
- concurrent.futures: Parallel processing

## Build and Run Instructions
### Build
```bash
docker build --platform linux/amd64 -t persona-doc-analyzer:latest .

Run
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none persona-doc-analyzer:latest

Input Structure

PDFs (e.g., doc1.pdf)
Round 1A outline JSONs (e.g., doc1.json)
config.json with:

{
  "persona": "PhD Researcher in Computational Biology",
  "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks",
  "documents": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
}

Output
Generates challenge1b_output.json with metadata, top 15 sections, and top 15 subsections.```