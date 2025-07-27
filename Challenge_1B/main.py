```python
import fitz
import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIntelligenceAnalyzer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        
    def extract_text_from_pdf_section(self, pdf_path: str, page_num: int, section_title: str, max_chars: int = 2000) -> str:
        try:
            doc = fitz.open(pdf_path)
            if page_num > len(doc):
                doc.close()
                return ""
            page = doc[page_num - 1]
            text = page.get_text()
            doc.close()
            lines = text.split('\n')
            section_text = ""
            found_section = False
            for i, line in enumerate(lines):
                if section_title.lower() in line.lower() and line.strip():
                    found_section = True
                    for j in range(i, min(i + 50, len(lines))):
                        current_line = lines[j].strip()
                        if current_line:
                            if j > i + 3 and len(current_line) < 50 and current_line.isupper() and not current_line.isdigit():
                                break
                            section_text += current_line + " "
                    break
            if not found_section:
                section_text = text[:max_chars]
            return section_text[:max_chars].strip()
        except Exception:
            logger.error(f"Error extracting text from {pdf_path}, page {page_num}")
            return ""
    
    def compute_relevance_score(self, text: str, persona: str, job_to_be_done: str) -> float:
        if not text.strip():
            return 0.0
        try:
            query = f"{persona} {job_to_be_done}"
            text_embedding = self.model.encode([text])
            query_embedding = self.model.encode([query])
            similarity = cosine_similarity(text_embedding, query_embedding)[0][0]
            keyword_score = self.compute_keyword_relevance(text, persona, job_to_be_done)
            final_score = 0.7 * similarity + 0.3 * keyword_score
            return max(0.0, min(1.0, final_score))
        except Exception:
            logger.error(f"Error computing relevance score")
            return self.compute_keyword_relevance(text, persona, job_to_be_done)
    
    def compute_keyword_relevance(self, text: str, persona: str, job_to_be_done: str) -> float:
        try:
            query = f"{persona} {job_to_be_done}".lower()
            text_lower = text.lower()
            key_terms = []
            if 'researcher' in query or 'research' in query:
                key_terms.extend(['methodology', 'method', 'approach', 'algorithm', 'dataset', 'data', 'experiment', 'result', 'analysis'])
            if 'analyst' in query or 'business' in query or 'investment' in query:
                key_terms.extend(['revenue', 'profit', 'financial', 'market', 'strategy', 'performance', 'growth', 'investment', 'return'])
            if 'student' in query or 'education' in query or 'exam' in query:
                key_terms.extend(['concept', 'principle', 'example', 'definition', 'theory', 'practice', 'exercise', 'problem'])
            job_words = re.findall(r'\b\w+\b', job_to_be_done.lower())
            key_terms.extend([word for word in job_words if len(word) > 3])
            matches = sum(1 for term in set(key_terms) if term in text_lower)
            max_possible_matches = len(set(key_terms))
            return matches / max_possible_matches if max_possible_matches > 0 else 0.0
        except Exception:
            logger.error(f"Error in keyword relevance computation")
            return 0.0
    
    def extract_subsections(self, text: str, max_subsections: int = 5) -> List[str]:
        if not text.strip():
            return []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 50]
            paragraphs = sentences
        subsections = paragraphs[:max_subsections]
        refined_subsections = []
        for subsection in subsections:
            refined = subsection[:80].strip() + "..." if len(subsection) > 80 else subsection.strip()
            refined_subsections.append(refined)
        return refined_subsections
    
    def process_single_document(self, args) -> Tuple[List[Dict], List[Dict]]:
        pdf_path, outline_path, persona, job_to_be_done = args
        try:
            with open(outline_path, 'r', encoding='utf-8') as f:
                outline_data = json.load(f)
            sections = []
            subsections = []
            for item in outline_data.get('outline', []):
                level = item.get('level', '')
                text_title = item.get('text', '')
                page_num = item.get('page', 1)
                section_text = self.extract_text_from_pdf_section(pdf_path, page_num, text_title)
                if section_text:
                    relevance_score = self.compute_relevance_score(section_text, persona, job_to_be_done)
                    sections.append({
                        'document': os.path.basename(pdf_path),
                        'page_number': page_num,
                        'section_title': text_title,
                        'relevance_score': relevance_score,
                        'text': section_text
                    })
                    sub_texts = self.extract_subsections(section_text)
                    for sub_text in sub_texts:
                        if sub_text:
                            sub_relevance = self.compute_relevance_score(sub_text, persona, job_to_be_done)
                            subsections.append({
                                'document': os.path.basename(pdf_path),
                                'page_number': page_num,
                                'refined_text': sub_text,
                                'relevance_score': sub_relevance
                            })
            return sections, subsections
        except Exception:
            logger.error(f"Error processing document {pdf_path}")
            return [], []
    
    def analyze_documents(self, input_dir: str, output_dir: str):
        try:
            config_path = os.path.join(input_dir, 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            persona = config['persona']
            job_to_be_done = config['job_to_be_done']
            document_names = config['documents']
            if len(document_names) < 3 or len(document_names) > 10:
                logger.error("Document collection must contain 3-10 PDFs")
                return
            process_args = []
            for doc_name in document_names:
                pdf_path = os.path.join(input_dir, doc_name)
                outline_name = doc_name.replace('.pdf', '.json')
                outline_path = os.path.join(input_dir, outline_name)
                if os.path.exists(pdf_path) and os.path.exists(outline_path):
                    process_args.append((pdf_path, outline_path, persona, job_to_be_done))
                else:
                    logger.warning(f"Missing files for {doc_name}")
            all_sections = []
            all_subsections = []
            max_workers = min(4, mp.cpu_count())
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.process_single_document, args) for args in process_args]
                for future in futures:
                    sections, subsections = future.result(timeout=30)
                    all_sections.extend(sections)
                    all_subsections.extend(subsections)
            all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
            all_subsections.sort(key=lambda x: x['relevance_score'], reverse=True)
            top_sections = all_sections[:15]
            top_subsections = all_subsections[:15]
            output = {
                "metadata": {
                    "input_documents": document_names,
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                },
                "extracted_sections": [
                    {
                        "document": section['document'],
                        "page_number": section['page_number'],
                        "section_title": section['section_title'],
                        "importance_rank": i + 1
                    }
                    for i, section in enumerate(top_sections)
                ],
                "subsection_analysis": [
                    {
                        "document": subsection['document'],
                        "page_number": subsection['page_number'],
                        "refined_text": subsection['refined_text']
                    }
                    for subsection in top_subsections
                ]
            }
            output_path = os.path.join(output_dir, 'challenge1b_output.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Output saved to {output_path}")
        except Exception:
            logger.error("Error in main analysis")
            raise

def main():
    input_dir = '/app/input'
    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    analyzer = DocumentIntelligenceAnalyzer()
    analyzer.analyze_documents(input_dir, output_dir)

if __name__ == "__main__":
    main()
```