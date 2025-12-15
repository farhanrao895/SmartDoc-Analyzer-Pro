📚 SmartDoc Analyzer Pro 4.0

SmartDoc Analyzer Pro 4.0 is an enterprise-grade Retrieval-Augmented Generation (RAG) application for analyzing large collections of construction and engineering documents such as Minutes of Meetings (MOMs), QA/QC records, progress reports, RFIs, NCRs, specifications, and spreadsheets.

The system is optimized for accuracy, traceability, and scalability, with OCR support, hybrid retrieval, and inline source citations.

🚀 Key Features
🔍 Advanced Hybrid Retrieval

Semantic search (FAISS + embeddings)

Keyword search (BM25)

Multi-query expansion

Reciprocal Rank Fusion (RRF) for best-of-both-worlds retrieval

📄 Document Intelligence

Native PDF parsing (text-based PDFs)

OCR support for scanned and mixed PDFs

Excel / CSV ingestion with structured row extraction

Automatic document classification:

QA/QC MOM

Progress Meeting (PGM)

RFI / NCR

Reports, Specifications, Letters, Drawings

🧠 Expert-Level RAG

Strict document-grounded answers

Partial-answer handling (never fails silently)

Multi-document aggregation for list-type questions

Domain-aware prompts (construction & contracts)

🎯 Inline Citations (Source-Aware Answers)

Clickable citation badges [1], [2], [3]

Source cards with:

File name

Page number

Meeting reference

Extracted chunk content

Toggleable citation UI

⚡ One-Time Knowledge Base Processing

Pre-process documents once

Persist FAISS indexes locally

Instant runtime querying

Cache validation using folder hashing

🖥️ Tech Stack

Python 3.10+

Streamlit – UI & chat experience

LangChain – RAG orchestration

FAISS – Vector search

BM25 – Keyword retrieval

OpenAI Embeddings – text-embedding-3-large

GPT-5.2 – Answer generation

PyMuPDF – PDF parsing

Tesseract OCR – Scanned PDF support

Pandas / OpenPyXL – Excel & CSV processing

📂 Project Structure
.
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── knowledge_bases/        # Generated locally (NOT committed)


⚠️ Important
knowledge_bases/ contains indexed document data and must never be pushed to GitHub.

🔐 Security & Data Safety

No API keys are hard-coded

API key is read from environment variables

All document data stays local

No uploads to external servers

Required .gitignore
.env
.env.*
.streamlit/secrets.toml
knowledge_bases/
*.faiss
*.pkl
documents.json

⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/your-org/smartdoc-analyzer-pro.git
cd smartdoc-analyzer-pro

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set environment variables

Create a .env file (do not commit it):

OPENAI_API_KEY=your_api_key_here

▶️ Run the Application
streamlit run main.py

📚 Using the App
Create a Knowledge Base

Provide a folder path containing PDFs / Excel files

Click Process & Save

Documents are:

OCR-processed if needed

Chunked and embedded

Stored locally for reuse

Ask Questions

“Who attended Quality Department meetings?”

“List all contractor delays discussed”

“Is there any EOT entitlement recorded?”

“Summarize IFC Addendum 3 notes across meetings”

Each answer includes fully traceable sources.

🧪 Designed For

Construction project teams

QA/QC departments

Claims & contract analysis

Engineering management

RAG research & experimentation

🧠 Architectural Highlights

Multi-query retrieval to reduce recall loss

Hybrid scoring (semantic + lexical)

RRF fusion instead of naïve merging

Relaxed answerability logic (partial > empty)

Chunk-level metadata preservation

📌 Limitations

Knowledge bases are local to the machine

Large datasets require sufficient RAM

OCR accuracy depends on scan quality

📈 Roadmap (Optional)

Role-based authentication

Cloud deployment (GCP / Azure)

Incremental KB updates

Knowledge-graph-assisted retrieval

Multi-KB routing

📄 License

This project is provided for internal / educational use.
