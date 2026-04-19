# SpecterScan

**AI-powered legal contract risk analysis.** Upload a PDF or plain-text contract and receive a structured, severity-tagged risk report with plain-English explanations and concrete mitigation recommendations — in seconds.

🔗 **Live Demo:** [specter-scan-7opa.vercel.app](https://specter-scan-7opa.vercel.app)
🔗 **Backend API:** [ironwallxr5-specterscan.hf.space](https://ironwallxr5-specterscan.hf.space)
🔗 **GitHub:** [github.com/IronwallxR5/SpecterScan](https://github.com/IronwallxR5/SpecterScan)

---

## What it does

SpecterScan runs every uploaded contract through a **LangGraph agentic workflow** with three stages:

1. **Clause Extraction & Classification** — spaCy segments the contract into individual sentences; a scikit-learn classifier (backed by `sentence-transformers/all-MiniLM-L6-v2` embeddings) flags each clause as risky or safe.
2. **RAG Retrieval** — Flagged clauses are embedded with `BAAI/bge-large-en-v1.5` and queried against a Pinecone vector index containing 7,700+ labelled legal clauses for contextual grounding.
3. **LLM Synthesis** — Groq's `llama-3.1-8b-instant` model produces a structured JSON report with per-clause severity (`High` / `Medium` / `Low`), plain-English explanation, and actionable mitigation — enforced by a Pydantic output schema.

The frontend renders the full document with **in-line colour highlights** and a clause-by-clause risk panel. Every clause card includes an **"Explain to Me"** button that fires a zero-shot Groq LLM call for an on-demand plain-English explanation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                │
│   Upload → results split-pane (doc viewer + clause list) │
└──────────────────────────┬──────────────────────────────┘
                           │ POST /analyze  (file upload)
                           │ POST /explain_clause  (JSON)
┌──────────────────────────▼──────────────────────────────┐
│               FastAPI Backend  (Python 3.10)            │
│                                                         │
│   Lifespan Startup:                                     │
│     all-MiniLM-L6-v2 ──► sklearn classifier (.pkl)     │
│     BAAI/bge-large-en-v1.5 ──► RAG embedder            │
│     Pinecone client ──► specterscan index               │
│     GroqWithFallback ──► 4-key pool                     │
│                                                         │
│   POST /analyze  ──►  LangGraph StateGraph              │
│                          │                              │
│                 ┌────────▼─────────┐                    │
│                 │  extract_risks   │ spaCy + sklearn    │
│                 └────────┬─────────┘                    │
│              flagged?    │    none flagged              │
│              ┌───────────┤    ┌────────────────┐        │
│              │           │    │ no_risks_found │─► END  │
│    ┌─────────▼──────┐    └────┴────────────────┘        │
│    │retrieve_context│ Pinecone top-k=3                  │
│    └─────────┬──────┘                                   │
│    ┌─────────▼──────────┐                               │
│    │ synthesize_report  │ Groq structured output        │
│    └─────────┬──────────┘                               │
│              └──► { filename, original_text, report }  │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React + TypeScript + Vite + CSS Modules | UI with drag-and-drop upload, split-pane viewer |
| Backend | FastAPI + Python 3.10 | REST API, file parsing, request orchestration |
| Agentic Workflow | LangGraph | Stateful 3-node graph with conditional routing |
| NLP Segmentation | spaCy `en_core_web_sm` | Sentence boundary detection |
| Risk Classification | scikit-learn + `all-MiniLM-L6-v2` | Binary risk labelling per clause |
| RAG Retrieval | Pinecone + `BAAI/bge-large-en-v1.5` | Semantic retrieval of similar legal clauses |
| LLM Synthesis | Groq `llama-3.1-8b-instant` | Structured JSON report generation |
| Structured Output | Pydantic v2 | Schema-enforced LLM responses |
| PDF Extraction | PyPDF2 | Page-by-page text extraction |
| Deployment (BE) | Hugging Face Spaces (Docker) | Containerised FastAPI on port 7860 |
| Deployment (FE) | Vercel | Static React build |

---

## Repository Structure

```
SpecterScan/
├── backend/
│   ├── agent.py                  # LangGraph state machine (3 nodes + conditional routing)
│   ├── main.py                   # FastAPI app, GroqWithFallback, lifespan model loading
│   ├── legal_risk_classifier.pkl # Trained sklearn binary classifier
│   ├── build_vector_db.py        # Offline script to populate Pinecone index
│   ├── requirements.txt
│   ├── Dockerfile                # For Hugging Face Spaces (port 7860)
│   └── .env.example              # Environment variable template
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # Root component, type definitions, fetch logic
│   │   └── components/
│   │       ├── UploadView/       # Drag-and-drop upload, file preview, demo mode
│   │       ├── ResultsView/      # Split-pane layout, stats header, summary banner
│   │       ├── DocumentViewer/   # In-document severity highlighting
│   │       └── ClausesList/      # Clause cards with severity badges + Explain button
│   ├── .env.example              # Frontend env variable template
│   └── vite.config.ts
├── main.tex                      # IEEE double-column project report
└── .gitignore
```

---

## Model Performance

The binary risk classifier was trained on a dataset of **21,144 labelled legal clauses** (12,816 risky, 8,328 normal) using `sentence-transformers/all-MiniLM-L6-v2` embeddings with `class_weight='balanced'` Logistic Regression.

```
              precision    recall  f1-score   support

           0       0.84      0.89      0.86      1666
           1       0.92      0.89      0.90      2563

    accuracy                           0.89      4229
   macro avg       0.88      0.89      0.88      4229
```

**89% accuracy** on held-out data. The classifier deliberately prioritises recall on risky clauses (F1 = 0.90) — it's better to flag a safe clause than to miss a dangerous one.

---

## API Reference

Base URL (local): `http://localhost:8000`
Base URL (production): `https://ironwallxr5-specterscan.hf.space`

### `GET /health`
Liveness probe.
```json
{ "status": "healthy", "version": "2.0.0" }
```

### `POST /analyze`
Upload a contract file and receive a full structured risk report.

**Request:** `multipart/form-data`, field `file`, accepts `.pdf` or `.txt`

**Response:**
```json
{
  "filename": "contract.pdf",
  "original_text": "...",
  "report": {
    "summary": "The contract includes three clauses with elevated risk...",
    "risks": [
      {
        "clause_index": 3,
        "clause_text": "The Client shall indemnify Provider from all claims...",
        "severity": "High",
        "explanation": "This clause places uncapped liability on the client...",
        "mitigation": "Limit indemnity to third-party claims caused by client negligence..."
      }
    ],
    "disclaimer": "This report is AI-generated and does not constitute legal advice."
  }
}
```

### `POST /explain_clause`
On-demand zero-shot LLM explanation for a single clause.

**Request:**
```json
{ "clause_text": "The contractor hereby waives all rights to legal counsel." }
```

**Response:**
```json
{ "explanation": "This clause is risky because it strips the contractor of their legal right to representation..." }
```

---

## Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- A [Pinecone](https://pinecone.io) account with the `specterscan` index populated
- A [Groq](https://console.groq.com) API key

### Backend

```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (one-time)
python -m spacy download en_core_web_sm

# Configure environment
cp .env.example .env
# Edit .env and fill in PINECONE_API_KEY and GROQ_API_KEY (+ fallback keys)

# Start the server
python -m uvicorn main:app --reload
```

> **Important:** Always use `python -m uvicorn` (not bare `uvicorn`) so the subprocess spawned by `--reload` inherits your virtualenv's packages correctly on macOS.

The API will be live at `http://localhost:8000`.

### Frontend

```bash
cd frontend

npm install

# Configure environment
cp .env.example .env
# Edit .env — set API_URL=http://localhost:8000 for local dev

npm run dev
```

The app will be live at `http://localhost:5173`.

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Description |
|---|---|---|
| `PINECONE_API_KEY` | ✅ | Pinecone API key for the `specterscan` index |
| `GROQ_API_KEY` | ✅ | Primary Groq API key |
| `GROQ_API_KEY_2` | Optional | Fallback key #2 (auto-rotated on rate-limit) |
| `GROQ_API_KEY_3` | Optional | Fallback key #3 |
| `GROQ_API_KEY_4` | Optional | Fallback key #4 |

### Frontend (`frontend/.env`)

| Variable | Description |
|---|---|
| `API_URL` | Backend base URL (e.g. `https://ironwallxr5-specterscan.hf.space`) |

---

## Deployment

### Hugging Face Spaces (Backend)

The backend runs as a Docker container on Hugging Face Spaces. The `Dockerfile` installs all dependencies, downloads the spaCy model, and starts FastAPI on port 7860.

Add **Repository Secrets** in your Space settings for all environment variables listed above. The `.env` file is gitignored and never committed.

### Vercel (Frontend)

Deploy the `frontend/` directory. Add `API_URL` as an environment variable in the Vercel project settings pointing to your Hugging Face Space URL.

---

## Known Limitations

- **Scanned PDFs** — PyPDF2 only extracts digital text. Image-based PDFs require OCR (not yet implemented).
- **Context-free classification** — Each clause is classified independently. Exceptions or cross-references between clauses are not modelled.
- **Single-document scope** — The system analyses one contract at a time. Batch processing is not supported.
- **LLM latency** — Cold-started Groq calls may take 3–8 seconds for structured synthesis on long contracts.

---

## License

MIT
