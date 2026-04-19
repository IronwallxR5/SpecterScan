"""
build_vector_db.py
==================
Run this script LOCALLY (not in Docker) to build the ChromaDB vector index
from legal_docs_cleaned.csv.

Usage:
    cd backend
    python build_vector_db.py

The resulting `chroma_db/` folder should be committed to the repository.
The FastAPI server will load this pre-built index at startup — no re-embedding
needed at runtime or Docker build time.
"""

import os
import sys
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
# This script lives in backend/; the CSV lives one level up in the project root.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(SCRIPT_DIR, "..", "legal_docs_cleaned.csv")
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma_db")

# ── Settings ──────────────────────────────────────────────────────────────────
MAX_ROWS    = 1000   # Only index the first N rows to keep the DB lightweight
BATCH_SIZE  = 100    # How many documents to add per ChromaDB call
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION  = "legal-risks"

def detect_text_column(df: pd.DataFrame) -> str:
    """
    Try common column name candidates for the raw text field.
    Falls back to the first column if none are found.
    """
    candidates = ["text", "clause_text", "clause", "content", "sentence", "document"]
    for c in candidates:
        if c in df.columns:
            print(f"[build_vector_db] Using text column: '{c}'")
            return c
    fallback = df.columns[0]
    print(f"[build_vector_db] No known text column found. Falling back to first column: '{fallback}'")
    return fallback


def main():
    print(f"[build_vector_db] Reading up to {MAX_ROWS} rows from: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at '{CSV_PATH}'. Make sure you run this from the backend/ directory.")
        sys.exit(1)

    # ── Load & clean ──────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH, nrows=MAX_ROWS)
    print(f"[build_vector_db] Loaded {len(df)} rows. Columns: {list(df.columns)}")

    text_col = detect_text_column(df)
    texts = df[text_col].dropna().astype(str).str.strip().tolist()
    # Remove trivially short entries that would pollute retrieval
    texts = [t for t in texts if len(t) >= 20]
    print(f"[build_vector_db] {len(texts)} valid documents after filtering.")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print(f"[build_vector_db] Loading embedding model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    print("[build_vector_db] Encoding documents (this may take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print(f"[build_vector_db] Encoded {len(embeddings)} embeddings of dim {embeddings.shape[1]}.")

    # ── Persist to ChromaDB ───────────────────────────────────────────────────
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Wipe and recreate so re-runs are idempotent
    try:
        client.delete_collection(COLLECTION)
        print(f"[build_vector_db] Deleted existing collection '{COLLECTION}'.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION)
    print(f"[build_vector_db] Created collection '{COLLECTION}'.")

    # Add in batches to avoid memory spikes
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        batch_texts      = texts[start:end]
        batch_embeddings = embeddings[start:end].tolist()
        batch_ids        = [f"doc_{start + j}" for j in range(len(batch_texts))]

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
        )
        print(f"[build_vector_db] Added batch {start}–{end}")

    print(f"\n✅  ChromaDB index built successfully.")
    print(f"    Collection : '{COLLECTION}'")
    print(f"    Documents  : {len(texts)}")
    print(f"    Location   : {CHROMA_PATH}")
    print(f"\nNext step: commit the 'chroma_db/' folder to the repository.")


if __name__ == "__main__":
    main()
