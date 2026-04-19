import os
import io
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Any

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything reads os.environ

import joblib
import spacy
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone as PineconeClient
from langchain_groq import ChatGroq
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import build_graph


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH            = os.path.join(os.path.dirname(__file__), "legal_risk_classifier.pkl")
CLASSIFY_EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RAG_EMBED_MODEL       = "BAAI/bge-large-en-v1.5"   # Must match the Pinecone index dimensions (1024)
PINECONE_INDEX_NAME   = "specterscan"

# Groq API key pool — loaded exclusively from environment variables.
# Keys are tried in order; the next is used when one hits a rate-limit or error.
GROQ_API_KEYS: List[str] = [
    k for k in [
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
        os.environ.get("GROQ_API_KEY_4"),
    ]
    if k  # filter out None / unset vars
]
# Deduplicate while preserving order
seen: set = set()
GROQ_API_KEYS = [k for k in GROQ_API_KEYS if not (k in seen or seen.add(k))]  # type: ignore[func-returns-value]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("specterscan")


# ── Groq Fallback Wrapper ─────────────────────────────────────────────────────

class GroqWithFallback:
    """
    Wraps multiple ChatGroq instances (one per API key) and automatically
    retries with the next key when a request fails due to rate-limiting (429)
    or any other API error. Exposes the same interface as ChatGroq so it
    can be passed directly to the LangGraph agent.
    """

    def __init__(self, api_keys: List[str], model: str = "llama3-8b-8192", temperature: float = 0.1):
        if not api_keys:
            raise RuntimeError("No Groq API keys available. Set GROQ_API_KEY or add fallback keys.")
        self._clients: List[ChatGroq] = [
            ChatGroq(model=model, temperature=temperature, groq_api_key=key)
            for key in api_keys
        ]
        self._current_index = 0
        logger.info(f"GroqWithFallback initialised with {len(self._clients)} API key(s).")

    def _next_client(self) -> ChatGroq:
        """Round-robin advance to the next available key."""
        self._current_index = (self._current_index + 1) % len(self._clients)
        return self._clients[self._current_index]

    def invoke(self, input: Any, **kwargs) -> Any:
        """Call invoke, retrying with each key on failure."""
        last_error = None
        for attempt in range(len(self._clients)):
            client = self._clients[(self._current_index + attempt) % len(self._clients)]
            try:
                result = client.invoke(input, **kwargs)
                # Advance current index on success so we distribute load
                self._current_index = (self._current_index + attempt) % len(self._clients)
                return result
            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str or "limit" in err_str or "auth" in err_str or "401" in err_str:
                    logger.warning(f"[Groq key #{attempt + 1}] Rate-limited or auth error — trying next key. Error: {e}")
                    time.sleep(0.3)
                    last_error = e
                else:
                    # Non-rate-limit error — re-raise immediately
                    raise e
        raise RuntimeError(f"All {len(self._clients)} Groq API keys exhausted. Last error: {last_error}")

    def with_structured_output(self, schema):
        """
        Return a wrapper that calls with_structured_output on each client in sequence
        until one succeeds, mirroring the ChatGroq interface used by agent.py.
        """
        return _StructuredOutputFallback(self._clients, self._current_index, schema)


class _StructuredOutputFallback:
    """
    Wraps multiple `client.with_structured_output(schema)` instances
    and retries across keys on failure.
    """

    def __init__(self, clients: List[ChatGroq], start_index: int, schema):
        self._structured = [c.with_structured_output(schema) for c in clients]
        self._start_index = start_index

    def invoke(self, input: Any, **kwargs) -> Any:
        last_error = None
        n = len(self._structured)
        for attempt in range(n):
            idx = (self._start_index + attempt) % n
            try:
                result = self._structured[idx].invoke(input, **kwargs)
                return result
            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str or "limit" in err_str or "auth" in err_str or "401" in err_str:
                    logger.warning(f"[Groq structured key #{idx + 1}] Rate-limited — trying next key. Error: {e}")
                    time.sleep(0.3)
                    last_error = e
                else:
                    raise e
        raise RuntimeError(f"All Groq API keys exhausted for structured output. Last error: {last_error}")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all heavy resources once at startup, store them on app.state,
    then release on shutdown. Nothing is re-loaded per-request.
    """
    logger.info("Starting up — loading models and connecting to services...")

    # 1. Classification embedder
    app.state.embedder = SentenceTransformer(CLASSIFY_EMBED_MODEL)
    logger.info(f"Classification embedder '{CLASSIFY_EMBED_MODEL}' loaded.")

    # 2. sklearn classifier
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find model at '{MODEL_PATH}'. "
            "Ensure 'legal_risk_classifier.pkl' is in the backend/ folder."
        )
    app.state.classifier = joblib.load(MODEL_PATH)
    logger.info("Classifier (.pkl) loaded.")

    # 3. spaCy
    try:
        app.state.nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )
        raise
    logger.info("spaCy 'en_core_web_sm' loaded.")

    # 4. RAG embedder (1024-dim to match Pinecone index)
    app.state.rag_embedder = SentenceTransformer(RAG_EMBED_MODEL)
    logger.info(f"RAG embedder '{RAG_EMBED_MODEL}' loaded.")

    # 5. Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY environment variable is not set. "
            "Add it to your Hugging Face Space secrets."
        )
    pc = PineconeClient(api_key=pinecone_api_key)
    app.state.pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' connected.")

    # 6. Groq LLM (multi-key fallback)
    app.state.groq_llm = GroqWithFallback(
        api_keys=GROQ_API_KEYS,
        model="llama3-8b-8192",
        temperature=0.1,
    )
    logger.info(f"Groq LLM initialised with {len(GROQ_API_KEYS)} fallback key(s).")

    # 7. LangGraph
    app.state.graph = build_graph(
        embedder=app.state.embedder,
        classifier=app.state.classifier,
        nlp=app.state.nlp,
        rag_embedder=app.state.rag_embedder,
        pinecone_index=app.state.pinecone_index,
        groq_llm=app.state.groq_llm,
    )
    logger.info("LangGraph compiled — server is ready!")

    yield

    logger.info("Shutting down — goodbye!")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SpecterScan API",
    description=(
        "Intelligent Contract Risk Analysis — "
        "LangGraph Agentic Workflow with Pinecone RAG + Groq LLM Synthesis"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://specter-scan-7opa.vercel.app",
        "https://specter-scan.vercel.app",
        "https://ironwallxr5-specterscan.hf.space",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Utility Helpers ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all readable text from a PDF via PyPDF2."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                logger.warning(f"Page {page_num} returned no text (may be scanned/image).")
        return "\n".join(pages_text)
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the uploaded PDF. It may be corrupted or encrypted. Error: {str(e)}",
        )


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode a plain-text file, trying UTF-8 then Latin-1."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed, falling back to Latin-1.")
        return file_bytes.decode("latin-1")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Server liveness probe for Docker / HF Spaces health checks."""
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/analyze")
def analyze_contract(file: UploadFile = File(...)):
    """
    Upload a contract (.pdf or .txt) and receive a structured agentic risk report.

    Response schema:
    {
        "filename":      str,
        "original_text": str,
        "report": {
            "summary":    str,
            "risks":      [{ "clause_index", "clause_text", "severity",
                             "explanation", "mitigation" }],
            "disclaimer": str
        }
    }
    """
    filename  = file.filename or "unknown"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in (".pdf", ".txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{extension}'. Please upload a .pdf or .txt file.",
        )

    try:
        file_bytes = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read the uploaded file: {str(e)}",
        )

    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty (0 bytes).")

    raw_text = (
        extract_text_from_pdf(file_bytes)
        if extension == ".pdf"
        else extract_text_from_txt(file_bytes)
    )

    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text could be extracted from the file. "
                "Scanned PDFs require OCR (not yet supported)."
            ),
        )

    logger.info(f"Extracted {len(raw_text)} characters from '{filename}'. Invoking LangGraph...")

    initial_state = {
        "contract_text":     raw_text,
        "all_clauses":       [],
        "flagged_clauses":   [],
        "retrieved_context": "",
        "structured_report": {},
        "errors":            [],
    }

    try:
        final_state = app.state.graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"LangGraph invocation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {str(e)}")

    if final_state.get("errors"):
        logger.warning(f"Agent completed with non-fatal warnings: {final_state['errors']}")

    return {
        "filename":      filename,
        "original_text": raw_text,
        "report":        final_state["structured_report"],
    }


class ExplainRequest(BaseModel):
    clause_text: str


@app.post("/explain_clause")
def explain_clause(request: ExplainRequest):
    """
    Accept a single clause string and return a concise plain-English explanation.

    Request:  { "clause_text": "The contractor hereby waives..." }
    Response: { "explanation": "This clause is risky because..." }
    """
    if not request.clause_text.strip():
        raise HTTPException(status_code=400, detail="clause_text cannot be empty.")

    prompt = (
        "You are a legal expert explaining contract clauses to non-lawyers. "
        "Explain why the following legal clause is risky in exactly 2 clear, "
        "jargon-free sentences. Be direct and specific.\n\n"
        f"Clause: {request.clause_text}"
    )

    try:
        response = app.state.groq_llm.invoke(prompt)
        return {"explanation": response.content}
    except Exception as e:
        logger.error(f"/explain_clause LLM call failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM explanation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
