import os
import io
import logging
from contextlib import asynccontextmanager

import joblib                     
import spacy                   
from PyPDF2 import PdfReader       
from sentence_transformers import SentenceTransformer  
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


MODEL_PATH = os.path.join(os.path.dirname(__file__), "legal_risk_classifier.pkl")


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


RISK_LABELS = {
    0: "Normal/Compliant",
    1: "Risky/Potential Issue",
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("specterscan")




@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs BEFORE the server starts accepting requests
    (the code above 'yield') and AFTER the server shuts down
    (the code below 'yield').
    """


    logger.info("Starting up — loading ML models...")


    app.state.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("SentenceTransformer loaded.")


    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at: {MODEL_PATH}")
        raise FileNotFoundError(
            f"Cannot find the model file at '{MODEL_PATH}'. "
            "Make sure 'legal_risk_classifier.pkl' is in the backend/ folder."
        )
    app.state.classifier = joblib.load(MODEL_PATH)
    logger.info("Classifier (.pkl) loaded.")

    try:
        app.state.nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )
        raise
    logger.info("spaCy model loaded.")

    logger.info("All models loaded successfully — server is ready!")


    yield


    logger.info("Shutting down — goodbye!")



app = FastAPI(
    title="SpecterScan API",
    description="Intelligent Contract Risk Analysis — clause-level risk classifier",
    version="1.0.0",
    lifespan=lifespan,
)




app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  
        "http://localhost:5173",    
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://specter-scan-7opa.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],            
)


# =====================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Given raw bytes of a PDF file, use PyPDF2 to extract all text.

    How it works:
      - PdfReader reads the binary content.
      - We loop through every page and call .extract_text() on each.
      - All page texts are joined together with newlines.

    Returns the combined text string (may be empty if the PDF
    contains only images or scanned content).
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                logger.warning(f"Page {page_num} returned no text (might be scanned/image).")
        return "\n".join(pages_text)
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the uploaded PDF. It may be corrupted or encrypted. Error: {str(e)}",
        )


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decode a plain-text file from bytes to a Python string.
    We try UTF-8 first (the most common encoding), then fall back
    to Latin-1 which never fails but might produce garbled characters.
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed, falling back to Latin-1.")
        return file_bytes.decode("latin-1")


def segment_into_clauses(text: str, nlp) -> list[str]:
    """
    Use spaCy's sentence boundary detection to split a contract
    into individual clauses/sentences.

    Why spaCy instead of simple regex splitting (e.g., splitting on '.')?
      - Legal text often contains abbreviations like "U.S.A." or "No."
        which would cause false splits with naive regex.
      - spaCy's model is trained on real English text and handles these
        edge cases much better.

    We also filter out any sentences that are just whitespace or
    shorter than 5 characters (those are usually noise).
    """
    doc = nlp(text)
    clauses = []
    for sent in doc.sents:
        clause = sent.text.strip()
    
        if len(clause) >= 5:
            clauses.append(clause)
    return clauses


# =================================================

@app.get("/health")
async def health_check():
    """
    A simple endpoint to verify the server is running.
    Useful for: Docker health checks, CI/CD, or just a quick sanity test.
    """
    return {"status": "healthy"}


@app.post("/analyze")
def analyze_contract(file: UploadFile = File(...)):
    """
    MAIN ENDPOINT — Upload a contract and get clause-level risk analysis.

    Pipeline:
      1. Validate the file extension (.pdf or .txt only).
      2. Read the file bytes.
      3. Extract raw text from the file.
      4. Segment the text into individual clauses using spaCy.
      5. Encode each clause into a 384-dim vector using SentenceTransformer.
      6. Predict risk (0 or 1) for each clause with the loaded classifier.
      7. Return a JSON response with all clauses and their predictions.

    Request:
      - Content-Type: multipart/form-data
      - Field name: "file"
      - Accepted types: .pdf, .txt

    Response (200 OK):
      {
        "filename": "contract.pdf",
        "total_clauses": 12,
        "results": [
          {
            "clause_index": 1,
            "clause_text": "The contractor shall ...",
            "risk_label": 0,
            "risk_category": "Normal/Compliant"
          },
          ...
        ]
      }
    """

    filename = file.filename or "unknown"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in (".pdf", ".txt"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{extension}'. "
                "Please upload a .pdf or .txt file."
            ),
        )

    try:
        # Since this is no longer an async function, we can just use file.file.read()
        file_bytes = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read the uploaded file: {str(e)}",
        )

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty (0 bytes).",
        )


    if extension == ".pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = extract_text_from_txt(file_bytes)

 
    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text could be extracted from the file. "
                "If this is a scanned PDF, it may require OCR (not supported yet)."
            ),
        )

    logger.info(f"Extracted {len(raw_text)} characters from '{filename}'.")

    clauses = segment_into_clauses(raw_text, app.state.nlp)

    if not clauses:
        raise HTTPException(
            status_code=400,
            detail="The file was readable but no meaningful clauses could be extracted.",
        )

    logger.info(f"Segmented into {len(clauses)} clauses.")


    try:

        embeddings = app.state.embedder.encode(clauses, show_progress_bar=False)

 
        predictions = app.state.classifier.predict(embeddings)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during analysis: {str(e)}",
        )

 
    results = []
    for i, (clause_text, prediction) in enumerate(zip(clauses, predictions)):
        label = int(prediction)    
        results.append({
            "clause_index": i + 1, 
            "clause_text": clause_text,
            "risk_label": label,
            "risk_category": RISK_LABELS.get(label, "Unknown"),
        })

    logger.info(
        f"Analysis complete for '{filename}': "
        f"{sum(1 for r in results if r['risk_label'] == 1)} risky / {len(results)} total clauses."
    )

    return {
        "filename": filename,
        "total_clauses": len(results),
        "results": results,
    }





if __name__ == "__main__":
    import uvicorn
    # When deployed on Render, it injects a PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    # reload=True is normally not recommended for production, 
    # but Render handles restarts anyway. Safer to turn it off.
    uvicorn.run("main:app", host="0.0.0.0", port=port)
