import os
import re
import time
import json
import hashlib
import platform
import string
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

# OCR-related imports (conditional)
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    import cv2
    import numpy as np
    
    if platform.system() == "Windows":
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Tesseract-OCR\tesseract.exe',
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                OCR_AVAILABLE = True
                break
    else:
        OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Excel-related imports
EXCEL_AVAILABLE = False
try:
    import pandas as pd
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# PyMuPDF for PDF analysis
FITZ_AVAILABLE = False
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# BM25 import
BM25_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# ----------------------------------------------------
# 🔧 PAGE CONFIG
# ----------------------------------------------------
PAGE_ICON = "📚"
PAGE_TITLE = "SmartDoc Analyzer Pro 4.0"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
load_dotenv()

# ----------------------------------------------------
# 🔧 CONFIG - EXPERT TUNED FOR ACCURACY
# ----------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5.2"
CHUNK_SIZE = 800  # Smaller chunks for better granularity
CHUNK_OVERLAP = 200  # More overlap to not split lists
MAX_CHUNKS = 100000
BATCH_SIZE = 50
QUOTA_LIMIT = 150
MAX_WORKERS = 8
KB_FOLDER = "knowledge_bases"

# RETRIEVAL CONFIG - EXPERT TUNED
RETRIEVAL_K = 100  # Retrieve MORE candidates
FINAL_K = 15  # Use more context
BM25_WEIGHT = 0.4  # Increased BM25 weight for keyword matching
SEMANTIC_WEIGHT = 0.6

# OCR Configuration
OCR_DPI = 300
OCR_LANG = "eng"
MIN_TEXT_THRESHOLD = 50
IMAGE_COVERAGE_THRESHOLD = 0.85

# ----------------------------------------------------
# 📝 EXPERT PROMPTS - GROUNDED & ACCURATE (WITH CITATIONS)
# ----------------------------------------------------

SYSTEM_TEMPLATE = """You are SmartDoc Analyzer Pro 4.0, a document-grounded QA system specialized in construction project documents.

CRITICAL RULES:
1) Answer based ONLY on the provided context. Extract and compile ALL relevant information found.
2) For list/compilation questions: Aggregate information from ALL sources provided. Do not stop at partial data.
3) If information is PARTIALLY available, provide what you found and clearly state what's missing.
4) NEVER say "Not found" if ANY relevant information exists in context - instead provide partial answer.
5) **CITATION REQUIREMENT**: You MUST cite sources using [1], [2], [3] etc. for EVERY factual claim.
   - Each number corresponds to the source number in the context
   - Example: "John Smith attended the meeting [1]" or "The deadline is March 15 [2][4]"
   - For lists, cite each item: "- Ahmed Hassan (QA Engineer) [1]"
6) **BE COMPREHENSIVE AND DETAILED**: Provide thorough, detailed answers. Do NOT summarize or shorten.
   - Include ALL relevant details from the context
   - For lists: include ALL items found, with their full details (names, roles, dates, etc.)
   - For explanations: provide complete information, not abbreviated versions
   - Longer, more detailed answers are preferred over brief summaries

DOMAIN KNOWLEDGE (use for interpretation):
- IFC = Issued For Construction
- MOM = Minutes of Meeting
- RFI = Request for Information
- NCR = Non-Conformance Report
- BOQ = Bill of Quantities
- EOT = Extension of Time
- VO = Variation Order
- QAQC = Quality Assurance / Quality Control

OUTPUT FORMAT:
For LIST questions (attendees, items, etc.):
- Compile ALL found items into a COMPLETE, DETAILED consolidated list
- Include full names, titles, organizations, and any other details available
- Remove duplicates but note frequency if relevant
- Include [source_number] citation for each item or group
- Do NOT truncate or summarize - list EVERYTHING found

For SPECIFIC questions:
- Direct answer first with citations [1], [2], etc.
- Provide ALL supporting evidence with citations
- Include relevant context, dates, details
- Note if information is incomplete

For NOT FOUND (only if truly zero relevant info):
- State clearly no relevant information found
- Suggest which document types might contain it
"""

HUMAN_TEMPLATE = """Question: {question}

Retrieved Context (from {num_sources} document chunks) - USE [1], [2], etc. to cite:
{context}

Instructions:
1. Carefully read ALL provided context chunks
2. Extract and compile ALL relevant information - DO NOT SUMMARIZE OR SHORTEN
3. **IMPORTANT: Add [source_number] citation after EVERY claim** (e.g., "The meeting was held on Jan 5 [1]")
4. For list questions: aggregate across all sources, deduplicate, cite each item with [source_number]
5. Provide COMPREHENSIVE, DETAILED answer - include ALL details found (names, dates, roles, specifics)
6. Only say "not found" if truly zero relevant information exists
7. IMPORTANT: Do not truncate or abbreviate your answer - longer detailed answers are preferred
"""

# Multi-query generation for better retrieval
MULTI_QUERY_TEMPLATE = """You are an expert at generating search queries for document retrieval.

Original question: {question}

Generate 5 different search queries that would help find relevant information.
Focus on:
1. Original intent with key terms
2. Synonyms and alternative phrasings  
3. Specific entity names or codes mentioned
4. Broader category if specific search fails
5. Related concepts that might appear near the answer

Return ONLY a JSON array of 5 query strings, no explanation:
["query1", "query2", "query3", "query4", "query5"]
"""

# Relaxed answerability - only reject if completely irrelevant
ANSWERABILITY_TEMPLATE = """Assess if the context contains ANY relevant information for the question.

Question: {question}

Context preview (first 2000 chars): {context}

Rules:
- Answer "YES" if context contains ANY information related to the question topic
- Answer "YES" if context has partial/incomplete information (partial is better than nothing)
- Answer "NO" ONLY if context is completely unrelated to the question topic

Respond with ONLY "YES" or "NO":"""


# ----------------------------------------------------
# 🎨 CITATION HTML/CSS STYLING
# ----------------------------------------------------
CITATION_CSS = """
<style>
.citation-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.8;
}

.answer-box {
    background: #ffffff;
    padding: 20px 25px;
    border-radius: 12px;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.answer-content {
    font-size: 1rem;
    color: #333;
    line-height: 1.9;
}

.citation-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.75em;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    margin: 0 1px;
    display: inline-block;
    transition: all 0.2s ease;
    vertical-align: middle;
}

.citation-badge:hover {
    transform: scale(1.15);
    box-shadow: 0 3px 8px rgba(102, 126, 234, 0.5);
}

.sources-section {
    margin-top: 25px;
}

.sources-title {
    font-size: 1.1em;
    font-weight: 600;
    color: #444;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 2px solid #667eea;
}

.source-card {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    margin-bottom: 12px;
    overflow: hidden;
    transition: all 0.2s ease;
}

.source-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.source-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 10px 15px;
    font-weight: 600;
    font-size: 0.9em;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.source-number {
    background: rgba(255,255,255,0.25);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85em;
}

.source-meta {
    background: #f8f9fa;
    padding: 8px 15px;
    font-size: 0.8em;
    color: #666;
    border-bottom: 1px solid #eee;
}

.source-meta span {
    margin-right: 15px;
}

.source-content {
    padding: 15px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.85em;
    line-height: 1.6;
    max-height: 250px;
    overflow-y: auto;
    background: #fafbfc;
    white-space: pre-wrap;
    word-wrap: break-word;
    color: #444;
}

.source-content::-webkit-scrollbar {
    width: 6px;
}

.source-content::-webkit-scrollbar-track {
    background: #f1f1f1;
}

.source-content::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

.no-citations {
    color: #888;
    font-style: italic;
    padding: 10px;
    background: #f9f9f9;
    border-radius: 8px;
}
</style>
"""


def format_answer_with_citations_html(answer: str, sources: List[Document]) -> str:
    """
    Convert answer with [1], [2] citations to interactive HTML.
    Creates clickable citation badges and source cards below.
    """
    
    # Build source lookup
    source_map = {}
    for i, doc in enumerate(sources):
        source_map[i + 1] = {
            "content": doc.page_content,
            "file_name": doc.metadata.get("file_name", "Unknown"),
            "page_number": doc.metadata.get("page_number", "N/A"),
            "doc_type": doc.metadata.get("doc_type", ""),
            "meeting_ref": doc.metadata.get("meeting_ref", ""),
        }
    
    # Find all citations used in the answer
    cited_nums = set(map(int, re.findall(r'\[(\d+)\]', answer)))
    
    # Replace [1], [2], etc. with styled badges
    def replace_citation(match):
        num = int(match.group(1))
        if num in source_map:
            return f'<span class="citation-badge" onclick="document.getElementById(\'src-{num}\').scrollIntoView({{behavior:\'smooth\',block:\'center\'}})" title="Click to view source">[{num}]</span>'
        return match.group(0)
    
    cited_answer = re.sub(r'\[(\d+)\]', replace_citation, answer)
    
    # Convert markdown-style formatting to HTML
    cited_answer = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', cited_answer)
    cited_answer = re.sub(r'\n', r'<br>', cited_answer)
    
    # Build HTML output
    html_parts = [CITATION_CSS]
    
    html_parts.append('<div class="citation-container">')
    
    # Answer section
    html_parts.append(f'''
    <div class="answer-box">
        <div class="answer-content">{cited_answer}</div>
    </div>
    ''')
    
    # Sources section (only show cited sources)
    if cited_nums:
        html_parts.append('<div class="sources-section">')
        html_parts.append(f'<div class="sources-title">📚 Cited Sources ({len(cited_nums)} referenced)</div>')
        
        for num in sorted(cited_nums):
            if num in source_map:
                src = source_map[num]
                # Escape HTML in content
                content_preview = src["content"][:600]
                if len(src["content"]) > 600:
                    content_preview += "..."
                content_escaped = content_preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                
                html_parts.append(f'''
                <div class="source-card" id="src-{num}">
                    <div class="source-header">
                        <span>📄 {src["file_name"]}</span>
                        <span class="source-number">Source [{num}]</span>
                    </div>
                    <div class="source-meta">
                        <span>📃 Page: {src["page_number"]}</span>
                        <span>📁 Type: {src["doc_type"] or "General"}</span>
                        {f'<span>🔖 Ref: {src["meeting_ref"]}</span>' if src["meeting_ref"] else ''}
                    </div>
                    <div class="source-content">{content_escaped}</div>
                </div>
                ''')
        
        html_parts.append('</div>')
    else:
        html_parts.append('<div class="no-citations">No specific sources were cited in this response.</div>')
    
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)


# ----------------------------------------------------
# 🔍 EXPERT BM25 TOKENIZATION
# ----------------------------------------------------
def expert_tokenize(text: str) -> List[str]:
    """
    Expert tokenization for BM25:
    - Lowercase
    - Keep alphanumeric and key punctuation
    - Handle construction codes (P2B-XXX, RFI-001, etc.)
    - N-grams for better matching
    """
    text = text.lower()
    
    # Preserve important codes/references
    codes = re.findall(r'[a-z0-9]+-[a-z0-9-]+', text)
    
    # Remove punctuation except hyphens in codes
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Split into tokens
    tokens = text.split()
    
    # Add preserved codes
    tokens.extend(codes)
    
    # Remove very short tokens except known abbreviations
    known_short = {'qa', 'qc', 'rfi', 'ncr', 'ifc', 'mom', 'boq', 'eot', 'vo'}
    tokens = [t for t in tokens if len(t) > 2 or t in known_short]
    
    # Add bigrams for better phrase matching
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    
    return tokens + bigrams


# ----------------------------------------------------
# 🔍 PDF TYPE DETECTION
# ----------------------------------------------------
def analyze_pdf_page(page) -> Dict[str, Any]:
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    
    text_blocks = page.get_text("blocks")
    text_area = 0
    text_content = ""
    
    for block in text_blocks:
        if block[6] == 0:
            x0, y0, x1, y1 = block[:4]
            text_area += (x1 - x0) * (y1 - y0)
            text_content += block[4]
    
    image_list = page.get_images()
    image_area = 0
    
    for img in image_list:
        try:
            xref = img[0]
            img_rect = page.get_image_rects(xref)
            if img_rect:
                for rect in img_rect:
                    image_area += rect.width * rect.height
        except:
            pass
    
    text_ratio = text_area / page_area if page_area > 0 else 0
    image_ratio = image_area / page_area if page_area > 0 else 0
    
    clean_text = text_content.strip()
    has_meaningful_text = len(clean_text) > MIN_TEXT_THRESHOLD
    is_image_dominant = image_ratio > IMAGE_COVERAGE_THRESHOLD
    
    return {
        "text_ratio": text_ratio,
        "image_ratio": image_ratio,
        "text_length": len(clean_text),
        "has_meaningful_text": has_meaningful_text,
        "is_image_dominant": is_image_dominant,
        "is_scanned": is_image_dominant and not has_meaningful_text,
        "text_content": clean_text
    }


def detect_pdf_type(pdf_path: str) -> Dict[str, Any]:
    if not FITZ_AVAILABLE:
        return {"type": "unknown", "needs_ocr": False, "pages": []}
    
    try:
        doc = fitz.open(pdf_path)
        pages_analysis = []
        scanned_pages = []
        text_pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            analysis = analyze_pdf_page(page)
            analysis["page_number"] = page_num + 1
            pages_analysis.append(analysis)
            
            if analysis["is_scanned"]:
                scanned_pages.append(page_num + 1)
            else:
                text_pages.append(page_num + 1)
        
        doc.close()
        
        total_pages = len(pages_analysis)
        scanned_ratio = len(scanned_pages) / total_pages if total_pages > 0 else 0
        
        if scanned_ratio > 0.8:
            pdf_type = "scanned"
            needs_ocr = True
        elif scanned_ratio > 0.2:
            pdf_type = "mixed"
            needs_ocr = True
        else:
            pdf_type = "text"
            needs_ocr = False
        
        return {
            "type": pdf_type,
            "needs_ocr": needs_ocr,
            "total_pages": total_pages,
            "scanned_pages": scanned_pages,
            "text_pages": text_pages,
            "scanned_ratio": scanned_ratio,
            "pages": pages_analysis
        }
        
    except Exception as e:
        return {"type": "error", "needs_ocr": False, "error": str(e), "pages": []}


# ----------------------------------------------------
# 🔤 OCR PROCESSING
# ----------------------------------------------------
def preprocess_image_for_ocr(image) -> Image.Image:
    if not OCR_AVAILABLE:
        return image
    
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return Image.fromarray(denoised)


def ocr_pdf_page(pdf_path: str, page_num: int) -> str:
    if not OCR_AVAILABLE:
        return ""
    
    try:
        images = convert_from_path(
            pdf_path,
            dpi=OCR_DPI,
            first_page=page_num,
            last_page=page_num
        )
        
        if not images:
            return ""
        
        image = images[0]
        processed_image = preprocess_image_for_ocr(image)
        
        config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(
            processed_image,
            lang=OCR_LANG,
            config=config
        )
        
        return text.strip()
        
    except Exception:
        return ""


# ----------------------------------------------------
# 📊 EXCEL PROCESSING
# ----------------------------------------------------
def process_excel_file(file_path: str) -> List[Document]:
    if not EXCEL_AVAILABLE:
        return []
    
    documents = []
    file_name = os.path.basename(file_path)
    
    try:
        xlsx = pd.ExcelFile(file_path)
        
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            
            text_content = f"Sheet: {sheet_name}\n\n"
            text_content += "Columns: " + ", ".join(df.columns.astype(str)) + "\n\n"
            
            max_rows = 500
            for idx, row in df.head(max_rows).iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                text_content += f"Row {idx + 1}: {row_text}\n"
            
            if len(df) > max_rows:
                text_content += f"\n... (truncated, {len(df) - max_rows} more rows)\n"
            
            doc = Document(
                page_content=text_content,
                metadata={
                    "file_name": file_name,
                    "file_path": file_path,
                    "sheet_name": sheet_name,
                    "file_type": "excel",
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "page_number": xlsx.sheet_names.index(sheet_name) + 1,
                    "category": "spreadsheet",
                    "doc_type": "Excel"
                }
            )
            documents.append(doc)
        
        return documents
        
    except Exception:
        return []


# ----------------------------------------------------
# 📁 KNOWLEDGE BASE MANAGEMENT
# ----------------------------------------------------
def ensure_kb_folder():
    if not os.path.exists(KB_FOLDER):
        os.makedirs(KB_FOLDER)
    return KB_FOLDER


def get_existing_knowledge_bases():
    ensure_kb_folder()
    kbs = []
    for item in os.listdir(KB_FOLDER):
        kb_path = os.path.join(KB_FOLDER, item)
        if os.path.isdir(kb_path):
            metadata_path = os.path.join(kb_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    kbs.append({
                        "name": item,
                        "path": kb_path,
                        "metadata": metadata
                    })
    return kbs


def compute_folder_hash(folder_path):
    hash_md5 = hashlib.md5()
    supported_extensions = (".pdf", ".xlsx", ".xls", ".csv")
    
    files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported_extensions)
    ])
    
    for file in files:
        hash_md5.update(file.encode())
        hash_md5.update(str(os.path.getmtime(file)).encode())
        hash_md5.update(str(os.path.getsize(file)).encode())
    
    return hash_md5.hexdigest()


def save_knowledge_base(kb_name, vector_store, documents, metadata):
    ensure_kb_folder()
    kb_path = os.path.join(KB_FOLDER, kb_name)
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
    
    vector_store.save_local(kb_path)
    
    docs_data = []
    for doc in documents:
        docs_data.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    
    docs_path = os.path.join(kb_path, "documents.json")
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, ensure_ascii=False, indent=2)
    
    metadata_path = os.path.join(kb_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return kb_path


def load_knowledge_base(kb_name):
    kb_path = os.path.join(KB_FOLDER, kb_name)
    if not os.path.exists(kb_path):
        raise ValueError(f"Knowledge base '{kb_name}' not found")
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        kb_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    docs_path = os.path.join(kb_path, "documents.json")
    documents = []
    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)
            for doc_dict in docs_data:
                documents.append(Document(
                    page_content=doc_dict["page_content"],
                    metadata=doc_dict["metadata"]
                ))
    
    metadata_path = os.path.join(kb_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return vector_store, documents, metadata


def delete_knowledge_base(kb_name):
    import shutil
    kb_path = os.path.join(KB_FOLDER, kb_name)
    if os.path.exists(kb_path):
        shutil.rmtree(kb_path)
        return True
    return False


def check_cache_valid(kb_name, folder_path):
    kb_path = os.path.join(KB_FOLDER, kb_name)
    metadata_path = os.path.join(kb_path, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return False
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    current_hash = compute_folder_hash(folder_path)
    return metadata.get("folder_hash") == current_hash


# ----------------------------------------------------
# 📥 DOCUMENT LOADING
# ----------------------------------------------------
def extract_meeting_reference(file_name):
    patterns = [
        r'(P2B[/-]QLY[/-]\d+)',
        r'(P2B[/-]PGM[/-]M[/-]\d+)',
        r'(MOM[/-]\d+)',
        r'(QAQC[/-]MOM[/-]\d+)',
        r'(RFI[/-]\d+)',
        r'(NCR[/-]\d+)',
        r'(\d{4}[/-]\d{2}[/-]\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def detect_document_type(file_name):
    file_lower = file_name.lower()
    
    type_patterns = {
        "QAQCMOM": ["qaqc", "qa/qc", "quality"],
        "PGM": ["pgm", "progress", "programme"],
        "MOM": ["mom", "minutes", "meeting"],
        "RFI": ["rfi", "request for information"],
        "NCR": ["ncr", "non-conformance"],
        "Letter": ["letter", "correspondence", "ltr"],
        "Submittal": ["submittal", "submission"],
        "Drawing": ["dwg", "drawing", "drg"],
        "Specification": ["spec", "specification"],
        "Contract": ["contract", "agreement"],
        "Report": ["report", "rpt"],
        "Excel": ["xlsx", "xls", "csv"],
    }
    
    for doc_type, keywords in type_patterns.items():
        for keyword in keywords:
            if keyword in file_lower:
                return doc_type
    
    return "General"


def extract_section_headers(content):
    headers = []
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) < 100:
            if line.isupper() and len(line) > 3:
                headers.append(line)
            elif re.match(r'^[\d.]+\s+[A-Z]', line):
                headers.append(line)
            elif re.match(r'^(ARTICLE|SECTION|CHAPTER|PART|ITEM|AGENDA)\s+', line, re.IGNORECASE):
                headers.append(line)
    return headers[:10]


def detect_document_category(file_name, content):
    file_lower = file_name.lower()
    content_lower = content[:2000].lower()
    
    categories = {
        "contract": ["contract", "agreement", "terms", "conditions", "party", "parties", "clause"],
        "invoice": ["invoice", "bill", "payment", "amount due", "total"],
        "report": ["report", "analysis", "summary", "findings", "conclusion"],
        "manual": ["manual", "guide", "instructions", "how to", "procedure"],
        "specification": ["specification", "spec", "requirements", "technical"],
        "legal": ["legal", "law", "court", "plaintiff", "defendant", "jurisdiction"],
        "financial": ["financial", "budget", "revenue", "expense", "profit", "loss"],
        "hr": ["employee", "hiring", "salary", "benefits", "leave", "policy"],
        "meeting": ["meeting", "minutes", "agenda", "attendees", "action items", "mom"],
        "construction": ["construction", "site", "contractor", "subcontractor", "ifc", "rfi"],
        "engineering": ["engineering", "design", "calculation", "drawing", "structural"],
        "spreadsheet": ["xlsx", "xls", "csv", "excel"],
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in file_lower or keyword in content_lower:
                return category
    
    return "general"


def load_single_pdf_with_ocr(pdf_path: str) -> List[Document]:
    documents = []
    file_name = os.path.basename(pdf_path)
    meeting_ref = extract_meeting_reference(file_name)
    doc_type = detect_document_type(file_name)
    
    try:
        pdf_analysis = detect_pdf_type(pdf_path)
        
        if pdf_analysis["type"] == "error":
            loader = PyMuPDFLoader(pdf_path)
            return loader.load()
        
        if pdf_analysis["type"] == "text":
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            
            for doc in docs:
                content = doc.page_content
                headers = extract_section_headers(content)
                category = detect_document_category(file_name, content)
                
                doc.metadata.update({
                    "file_name": file_name,
                    "file_path": pdf_path,
                    "page_number": doc.metadata.get("page", 0) + 1,
                    "section_headers": headers,
                    "category": category,
                    "doc_type": doc_type,
                    "meeting_ref": meeting_ref,
                    "ocr_applied": False,
                    "pdf_type": "text"
                })
                documents.append(doc)
        
        elif pdf_analysis["type"] in ["scanned", "mixed"]:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_analysis = pdf_analysis["pages"][page_num]
                
                if page_analysis["is_scanned"]:
                    if OCR_AVAILABLE:
                        text = ocr_pdf_page(pdf_path, page_num + 1)
                        ocr_applied = True
                    else:
                        text = ""
                        ocr_applied = False
                else:
                    text = page.get_text()
                    ocr_applied = False
                
                if text.strip():
                    headers = extract_section_headers(text)
                    category = detect_document_category(file_name, text)
                    
                    doc_obj = Document(
                        page_content=text,
                        metadata={
                            "file_name": file_name,
                            "file_path": pdf_path,
                            "page_number": page_num + 1,
                            "section_headers": headers,
                            "category": category,
                            "doc_type": doc_type,
                            "meeting_ref": meeting_ref,
                            "ocr_applied": ocr_applied,
                            "pdf_type": pdf_analysis["type"],
                            "char_count": len(text)
                        }
                    )
                    documents.append(doc_obj)
            
            doc.close()
        
        return documents
        
    except Exception as e:
        return [{"error": str(e), "file": pdf_path}]


def load_single_file(file_path: str) -> List[Document]:
    file_lower = file_path.lower()
    
    if file_lower.endswith(".pdf"):
        result = load_single_pdf_with_ocr(file_path)
        if result and isinstance(result[0], dict) and "error" in result[0]:
            return result
        return result
    elif file_lower.endswith((".xlsx", ".xls")):
        return process_excel_file(file_path)
    elif file_lower.endswith(".csv"):
        if EXCEL_AVAILABLE:
            try:
                df = pd.read_csv(file_path)
                file_name = os.path.basename(file_path)
                
                text_content = f"CSV File: {file_name}\n\n"
                text_content += "Columns: " + ", ".join(df.columns.astype(str)) + "\n\n"
                
                max_rows = 500
                for idx, row in df.head(max_rows).iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    text_content += f"Row {idx + 1}: {row_text}\n"
                
                return [Document(
                    page_content=text_content,
                    metadata={
                        "file_name": file_name,
                        "file_path": file_path,
                        "file_type": "csv",
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "page_number": 1,
                        "category": "spreadsheet",
                        "doc_type": "CSV"
                    }
                )]
            except Exception as e:
                return [{"error": str(e), "file": file_path}]
        else:
            return [{"error": "Pandas not available", "file": file_path}]
    else:
        return [{"error": f"Unsupported file type", "file": file_path}]


def process_files_in_folder(folder_path: str) -> Tuple[List[Document], Dict[str, Any]]:
    documents = []
    stats = {
        "total_files": 0,
        "pdf_text": 0,
        "pdf_scanned": 0,
        "pdf_mixed": 0,
        "excel": 0,
        "csv": 0,
        "errors": [],
        "ocr_pages": 0
    }
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid directory: {folder_path}")
    
    supported_extensions = (".pdf", ".xlsx", ".xls", ".csv")
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported_extensions)
    ]
    
    if not files:
        raise ValueError("No supported files found.")
    
    stats["total_files"] = len(files)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    completed = 0
    
    with st.spinner(f"📂 Processing {len(files)} files..."):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(load_single_file, f): f for f in files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                file_name = os.path.basename(file_path)
                completed += 1
                
                status_text.markdown(f"📄 Processing {completed}/{len(files)} → {file_name}")
                
                try:
                    result = future.result()
                    
                    if result and isinstance(result[0], dict) and "error" in result[0]:
                        stats["errors"].append(result[0])
                    else:
                        documents.extend(result)
                        
                        file_lower = file_path.lower()
                        if file_lower.endswith(".pdf") and result:
                            pdf_type = result[0].metadata.get("pdf_type", "text")
                            if pdf_type == "text":
                                stats["pdf_text"] += 1
                            elif pdf_type == "scanned":
                                stats["pdf_scanned"] += 1
                            else:
                                stats["pdf_mixed"] += 1
                            
                            for doc in result:
                                if doc.metadata.get("ocr_applied"):
                                    stats["ocr_pages"] += 1
                        elif file_lower.endswith((".xlsx", ".xls")):
                            stats["excel"] += 1
                        elif file_lower.endswith(".csv"):
                            stats["csv"] += 1
                            
                except Exception as e:
                    stats["errors"].append({"file": file_path, "error": str(e)})
                
                progress_bar.progress(completed / len(files))
    
    progress_bar.empty()
    status_text.empty()
    
    return documents, stats


# ----------------------------------------------------
# ✂️ TEXT SPLITTING
# ----------------------------------------------------
def process_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " ", ""]
    )
    
    chunks_with_metadata = []
    
    for doc in documents:
        if not doc.page_content:
            continue
        
        splits = splitter.split_text(doc.page_content)
        
        for i, split in enumerate(splits):
            chunk_doc = Document(
                page_content=split,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(splits)
                }
            )
            chunks_with_metadata.append(chunk_doc)
    
    if len(chunks_with_metadata) > MAX_CHUNKS:
        chunks_with_metadata = chunks_with_metadata[:MAX_CHUNKS]
    
    return chunks_with_metadata


# ----------------------------------------------------
# 🧠 EMBEDDING
# ----------------------------------------------------
def embed_documents_in_batches(chunks_with_metadata):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = None
    delay = 60 / (QUOTA_LIMIT / BATCH_SIZE)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    texts = [doc.page_content for doc in chunks_with_metadata]
    metadatas = [doc.metadata for doc in chunks_with_metadata]
    
    with st.spinner("🧠 Generating embeddings..."):
        total_batches = (len(texts) // BATCH_SIZE) + 1
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_metadatas = metadatas[i:i + BATCH_SIZE]
            
            status_text.markdown(f"⚙️ Embedding batch {i // BATCH_SIZE + 1}/{total_batches}")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if vector_store is None:
                        vector_store = FAISS.from_texts(
                            batch_texts, embedding=embeddings, metadatas=batch_metadatas
                        )
                    else:
                        temp = FAISS.from_texts(
                            batch_texts, embedding=embeddings, metadatas=batch_metadatas
                        )
                        vector_store.merge_from(temp)
                    break
                except Exception as e:
                    if "rate" in str(e).lower() and attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))
                    else:
                        raise e
            
            time.sleep(delay)
            progress_bar.progress(min((i + BATCH_SIZE) / len(texts), 1.0))
    
    progress_bar.empty()
    status_text.empty()
    
    return vector_store


# ----------------------------------------------------
# 🔍 EXPERT HYBRID RETRIEVAL WITH MULTI-QUERY
# ----------------------------------------------------
def generate_multi_queries(question: str, llm) -> List[str]:
    """Generate multiple search queries for better retrieval coverage"""
    try:
        prompt = PromptTemplate(
            input_variables=["question"],
            template=MULTI_QUERY_TEMPLATE
        )
        
        chain = prompt | llm
        result = chain.invoke({"question": question})
        response = result.content.strip()
        
        # Clean and parse JSON
        response = response.replace("```json", "").replace("```", "").strip()
        queries = json.loads(response)
        
        # Always include original question
        if question not in queries:
            queries.insert(0, question)
        
        return queries[:6]  # Max 6 queries
        
    except Exception:
        # Fallback: generate simple variations
        return [
            question,
            question.lower(),
            " ".join(question.split()[:10]),  # First 10 words
        ]


def expert_bm25_search(
    documents: List[Document],
    query: str,
    k: int = 20,
    filter_dict: Optional[Dict] = None
) -> List[Tuple[Document, float]]:
    """Expert BM25 search with proper tokenization and scoring"""
    
    if not BM25_AVAILABLE or not documents:
        return []
    
    # Filter documents first
    filtered_docs = documents
    if filter_dict:
        filtered_docs = [
            doc for doc in documents
            if all(doc.metadata.get(key) == val for key, val in filter_dict.items())
        ]
    
    if not filtered_docs:
        return []
    
    try:
        # Tokenize with expert tokenizer
        tokenized_docs = [expert_tokenize(doc.page_content) for doc in filtered_docs]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Tokenize query
        tokenized_query = expert_tokenize(query)
        
        # Get scores
        scores = bm25.get_scores(tokenized_query)
        
        # Pair documents with scores
        doc_scores = list(zip(filtered_docs, scores))
        
        # Sort by score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k with non-zero scores
        return [(doc, score) for doc, score in doc_scores[:k] if score > 0]
        
    except Exception:
        return []


def expert_semantic_search(
    vector_store,
    query: str,
    k: int = 20,
    filter_dict: Optional[Dict] = None
) -> List[Tuple[Document, float]]:
    """Semantic search with scores"""
    
    try:
        # Get results with scores
        results = vector_store.similarity_search_with_score(query, k=k * 2)
        
        # Filter if needed
        if filter_dict:
            results = [
                (doc, score) for doc, score in results
                if all(doc.metadata.get(key) == val for key, val in filter_dict.items())
            ]
        
        return results[:k]
        
    except Exception:
        return []


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Document, float]]],
    k: int = 60
) -> List[Document]:
    """
    Reciprocal Rank Fusion to combine multiple result lists.
    Better than simple merging - gives higher weight to docs appearing in multiple lists.
    """
    
    # Track scores by document content hash
    fusion_scores = defaultdict(float)
    doc_map = {}  # content_hash -> document
    
    for result_list in result_lists:
        for rank, (doc, _) in enumerate(result_list):
            content_hash = hash(doc.page_content[:500])
            
            # RRF formula: 1 / (k + rank)
            fusion_scores[content_hash] += 1.0 / (k + rank + 1)
            
            if content_hash not in doc_map:
                doc_map[content_hash] = doc
    
    # Sort by fusion score
    sorted_hashes = sorted(fusion_scores.keys(), key=lambda h: fusion_scores[h], reverse=True)
    
    return [doc_map[h] for h in sorted_hashes]


def expert_hybrid_search(
    vector_store,
    documents: List[Document],
    queries: List[str],
    filter_dict: Optional[Dict] = None,
    k: int = FINAL_K
) -> List[Document]:
    """
    Expert hybrid search combining:
    1. Multi-query retrieval
    2. BM25 (keyword) search
    3. Semantic (embedding) search
    4. Reciprocal Rank Fusion for combining results
    """
    
    all_results = []
    
    for query in queries:
        # BM25 search
        bm25_results = expert_bm25_search(
            documents, query, k=RETRIEVAL_K // 2, filter_dict=filter_dict
        )
        if bm25_results:
            all_results.append(bm25_results)
        
        # Semantic search
        semantic_results = expert_semantic_search(
            vector_store, query, k=RETRIEVAL_K // 2, filter_dict=filter_dict
        )
        if semantic_results:
            all_results.append(semantic_results)
    
    if not all_results:
        return []
    
    # Combine using RRF
    fused_results = reciprocal_rank_fusion(all_results)
    
    return fused_results[:k]


# ----------------------------------------------------
# 🤖 RAG CHAIN
# ----------------------------------------------------
def get_conversational_chain(filter_metadata=None):
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.1,  # Lower for more factual
        api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs={"max_completion_tokens": 8000}  # Pass directly to API for GPT-5.2
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    
    st.session_state.current_filter = filter_metadata
    st.session_state.llm = llm
    
    return memory, llm


def format_docs_for_context(docs: List[Document]) -> str:
    formatted_parts = []
    
    for i, doc in enumerate(docs):
        meta = doc.metadata
        file_name = meta.get("file_name", "Unknown")
        page_num = meta.get("page_number", "N/A")
        doc_type = meta.get("doc_type", "")
        meeting_ref = meta.get("meeting_ref", "")
        
        header = f"[Source {i+1}: {file_name} | Page {page_num}"
        if doc_type:
            header += f" | {doc_type}"
        if meeting_ref:
            header += f" | Ref: {meeting_ref}"
        header += "]"
        
        formatted_parts.append(f"{header}\n{doc.page_content}\n")
    
    return "\n---\n".join(formatted_parts)


def check_answerability_relaxed(question: str, context: str, llm) -> bool:
    """Relaxed answerability check - only reject if completely irrelevant"""
    try:
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=ANSWERABILITY_TEMPLATE
        )
        
        chain = prompt | llm
        result = chain.invoke({"question": question, "context": context[:2000]})
        response = result.content.strip().upper()
        
        return response.startswith("YES")
        
    except Exception:
        return True  # Default to answerable if check fails


def run_expert_rag_query(question: str, memory, llm) -> Dict[str, Any]:
    """Expert RAG query with multi-query retrieval and comprehensive answering"""
    
    # Step 1: Generate multiple search queries
    queries = generate_multi_queries(question, llm)
    
    # Step 2: Expert hybrid search
    source_documents = expert_hybrid_search(
        vector_store=st.session_state.vector_store,
        documents=st.session_state.documents,
        queries=queries,
        filter_dict=st.session_state.get("current_filter"),
        k=FINAL_K
    )
    
    if not source_documents:
        return {
            "answer": "**No relevant documents found.** Please check if the knowledge base contains documents related to your query.",
            "source_documents": [],
            "queries": queries,
            "answerable": False,
            "html_answer": None
        }
    
    # Step 3: Format context
    context = format_docs_for_context(source_documents)
    
    # Step 4: Relaxed answerability check
    is_answerable = check_answerability_relaxed(question, context, llm)
    
    # Step 5: Generate answer (even if partial info available)
    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ]
    
    final_prompt = ChatPromptTemplate.from_messages(messages)
    chain = final_prompt | llm
    
    result = chain.invoke({
        "question": question,
        "context": context,
        "num_sources": len(source_documents)
    })
    
    answer = result.content
    
    # Step 6: Generate HTML with citations
    html_answer = format_answer_with_citations_html(answer, source_documents)
    
    memory.save_context({"input": question}, {"answer": answer})
    
    return {
        "answer": answer,
        "source_documents": source_documents,
        "queries": queries,
        "answerable": is_answerable,
        "html_answer": html_answer
    }


# ----------------------------------------------------
# 🧭 SESSION STATE
# ----------------------------------------------------
def initialize_session_state():
    defaults = {
        "vector_store": None,
        "documents": [],
        "messages": [],
        "memory": None,
        "llm": None,
        "current_kb": None,
        "kb_metadata": None,
        "current_filter": None,
        "processing_stats": None,
        "show_citations": True  # NEW: Toggle for citation display
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_chat_history():
    """Display chat history with HTML citations support"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("html_answer") and st.session_state.get("show_citations", True):
                # Display HTML version with citations
                st.components.v1.html(msg["html_answer"], height=600, scrolling=True)
            else:
                # Display plain markdown
                st.markdown(msg["content"])


# ----------------------------------------------------
# 🚀 MAIN APP
# ----------------------------------------------------
def main():
    st.title("📚 SmartDoc Analyzer Pro 4.0")
    
    capabilities = []
    capabilities.append("✅ OCR" if OCR_AVAILABLE else "❌ OCR")
    capabilities.append("✅ Excel" if EXCEL_AVAILABLE else "❌ Excel")
    capabilities.append("✅ PDF" if FITZ_AVAILABLE else "❌ PDF")
    capabilities.append("✅ BM25" if BM25_AVAILABLE else "❌ BM25")
    
    st.caption(f"Expert Hybrid Retrieval | Multi-Query | RRF Fusion | 🎯 Inline Citations | {' | '.join(capabilities)}")
    st.markdown("---")
    
    initialize_session_state()
    
    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Citation Toggle (NEW)
        st.session_state.show_citations = st.checkbox(
            "🎯 Show Inline Citations", 
            value=st.session_state.get("show_citations", True),
            help="Toggle interactive citation display"
        )
        
        st.markdown("---")
        
        # Knowledge Bases
        st.subheader("📚 Knowledge Bases")
        
        existing_kbs = get_existing_knowledge_bases()
        
        if existing_kbs:
            kb_names = ["-- Select --"] + [kb["name"] for kb in existing_kbs]
            selected_kb = st.selectbox("Load existing KB:", kb_names)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if selected_kb != "-- Select --" and st.button("📂 Load"):
                    with st.spinner(f"Loading {selected_kb}..."):
                        try:
                            vector_store, documents, metadata = load_knowledge_base(selected_kb)
                            st.session_state.vector_store = vector_store
                            st.session_state.documents = documents
                            st.session_state.kb_metadata = metadata
                            st.session_state.current_kb = selected_kb
                            st.session_state.memory, st.session_state.llm = get_conversational_chain()
                            st.success(f"✅ Loaded: {selected_kb}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {str(e)}")
            
            with col2:
                if selected_kb != "-- Select --":
                    if st.button("🗑️ Delete"):
                        if delete_knowledge_base(selected_kb):
                            st.success(f"Deleted: {selected_kb}")
                            st.rerun()
        else:
            st.info("No existing knowledge bases")
        
        st.markdown("---")
        
        # Create new KB
        st.subheader("➕ Create New KB")
        kb_name = st.text_input("KB Name:", placeholder="e.g., project_docs")
        folder_path = st.text_input("📁 Folder Path:")
        
        if st.button("🚀 Process & Save", type="primary"):
            if not folder_path or not kb_name:
                st.warning("Please enter both KB name and folder path.")
            else:
                if check_cache_valid(kb_name, folder_path):
                    try:
                        vector_store, documents, metadata = load_knowledge_base(kb_name)
                        st.session_state.vector_store = vector_store
                        st.session_state.documents = documents
                        st.session_state.kb_metadata = metadata
                        st.session_state.current_kb = kb_name
                        st.session_state.memory, st.session_state.llm = get_conversational_chain()
                        st.success("✅ Loaded from cache!")
                        st.rerun()
                    except:
                        pass
                
                with st.status("🧠 Processing...", expanded=True) as status:
                    try:
                        docs, stats = process_files_in_folder(folder_path)
                        st.write(f"📊 {stats['total_files']} files processed")
                        
                        chunks = process_text_chunks(docs)
                        st.write(f"📄 {len(chunks)} chunks created")
                        
                        vector_store = embed_documents_in_batches(chunks)
                        
                        metadata = {
                            "folder_path": folder_path,
                            "folder_hash": compute_folder_hash(folder_path),
                            "total_documents": len(docs),
                            "total_chunks": len(chunks),
                            "files": list(set(d.metadata.get("file_name", "") for d in chunks)),
                            "categories": list(set(d.metadata.get("category", "") for d in chunks)),
                            "doc_types": list(set(d.metadata.get("doc_type", "") for d in chunks)),
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        
                        save_knowledge_base(kb_name, vector_store, chunks, metadata)
                        
                        st.session_state.vector_store = vector_store
                        st.session_state.documents = chunks
                        st.session_state.kb_metadata = metadata
                        st.session_state.current_kb = kb_name
                        st.session_state.memory, st.session_state.llm = get_conversational_chain()
                        
                        status.update(label="✅ Complete!", state="complete")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed: {str(e)}")
        
        # Filters
        st.markdown("---")
        st.header("🔍 Filters")
        
        if st.session_state.kb_metadata:
            metadata = st.session_state.kb_metadata
            
            files = ["All Files"] + sorted(metadata.get("files", []))
            selected_file = st.selectbox("Filter by File:", files)
            
            doc_types = ["All Types"] + sorted(metadata.get("doc_types", []))
            selected_type = st.selectbox("Filter by Type:", doc_types)
            
            if st.button("Apply Filters"):
                filter_dict = {}
                if selected_file != "All Files":
                    filter_dict["file_name"] = selected_file
                if selected_type != "All Types":
                    filter_dict["doc_type"] = selected_type
                
                st.session_state.current_filter = filter_dict if filter_dict else None
                st.success("Filters applied" if filter_dict else "Filters cleared")
            
            if st.button("Clear Filters"):
                st.session_state.current_filter = None
                st.info("Filters cleared")
        
        # Metrics
        st.markdown("---")
        st.header("📊 Metrics")
        
        if st.session_state.vector_store:
            st.metric("Total Chunks", st.session_state.vector_store.index.ntotal)
            if st.session_state.kb_metadata:
                st.metric("Files", len(st.session_state.kb_metadata.get("files", [])))
        
        if st.session_state.current_kb:
            st.info(f"📚 KB: {st.session_state.current_kb}")
        
        if st.session_state.current_filter:
            st.warning(f"🔍 Filter active")
        
        st.markdown("---")
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.rerun()
    
    # MAIN CHAT
    display_chat_history()
    
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if not st.session_state.vector_store:
            st.warning("⚠️ Please load a knowledge base first.")
            return
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching with expert retrieval..."):
                try:
                    result = run_expert_rag_query(
                        prompt,
                        st.session_state.memory,
                        st.session_state.llm
                    )
                    
                    # Display based on citation toggle
                    if st.session_state.get("show_citations", True) and result.get("html_answer"):
                        # Show interactive HTML with citations
                        st.components.v1.html(result["html_answer"], height=600, scrolling=True)
                    else:
                        # Show plain markdown
                        st.markdown(result["answer"])
                    
                    # Store message with both formats
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "html_answer": result.get("html_answer")
                    })
                    
                    # Show search queries used
                    with st.expander("🔍 Search Queries Used"):
                        for i, q in enumerate(result.get("queries", []), 1):
                            st.markdown(f"{i}. {q}")
                    
                    # Show sources (collapsed by default since they're in HTML)
                    if result["source_documents"]:
                        with st.expander(f"📄 All Retrieved Sources ({len(result['source_documents'])} chunks)"):
                            for i, doc in enumerate(result["source_documents"]):
                                meta = doc.metadata
                                st.markdown(f"**Source {i+1}:** {meta.get('file_name', 'Unknown')} | Page {meta.get('page_number', 'N/A')} | {meta.get('doc_type', '')}")
                                st.code(doc.page_content[:500], language=None)
                                st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    with st.expander("Debug"):
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
