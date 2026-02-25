"""
SpecterScan — Model Training Script
====================================
Trains the legal risk classifier from the CSV dataset.
Run once:  python train_model.py

This produces `legal_risk_classifier.pkl` in the same folder.
"""

import os
import re
import ssl
import warnings
import numpy as np
import pandas as pd
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, "legal_docs_cleaned.csv")
MODEL_OUTPUT = os.path.join(SCRIPT_DIR, "legal_risk_classifier.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def download_nltk_data():
    """Download required NLTK data silently."""
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        nltk.download(pkg, quiet=True)


def preprocess_legal_text(text: str, stop_words: set) -> str:
    """Clean text while preserving legally significant words."""
    words_to_keep = {
        "no", "not", "nor", "except", "against", "without", "only", "any", "but",
        "more", "than", "less", "least", "greater", "equal", "over", "under",
        "above", "below", "if", "until", "while", "all", "both", "each",
        "other", "some", "such", "prior", "after", "before", "during", "once",
        "can", "will", "should", "a",
    }
    legal_stop_words = stop_words - words_to_keep

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s%$]", " ", text)
    words = word_tokenize(text)
    filtered = [w for w in words if w not in legal_stop_words]
    return re.sub(r"\s+", " ", " ".join(filtered)).strip()


def train():
    """Full training pipeline — load data, embed, train, save."""
    print("=" * 50)
    print("  SpecterScan — Model Training")
    print("=" * 50)

    # 1. NLTK setup
    print("\n[1/6] Downloading NLTK data...")
    download_nltk_data()
    stop_words = set(stopwords.words("english"))

    # 2. Load CSV
    print("[2/6] Loading dataset...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {CSV_PATH}\n"
            "Make sure 'legal_docs_cleaned.csv' is in the project root."
        )
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["clause_text"])
    df["totalwords"] = df["totalwords"].fillna(
        df["clause_text"].apply(lambda x: len(str(x).split()))
    )
    df["totalletters"] = df["totalletters"].fillna(
        df["clause_text"].apply(lambda x: sum(c.isalpha() for c in str(x)))
    )
    print(f"   Loaded {len(df)} rows  |  Label distribution:")
    print(f"   {df['clause_status'].value_counts().to_dict()}")

    # 3. Preprocess
    print("[3/6] Preprocessing text...")
    df["cleaned_clause"] = df["clause_text"].apply(
        lambda t: preprocess_legal_text(t, stop_words)
    )

    # 4. Embed
    print("[4/6] Generating sentence embeddings (this may take a minute)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    X = embedder.encode(df["clause_text"].astype(str).tolist(), show_progress_bar=True)
    y = df["clause_status"]

    # 5. Train
    print("[5/6] Training Logistic Regression classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("\n   Classification Report:")
    print("   " + "-" * 44)
    report = classification_report(y_test, y_pred)
    for line in report.strip().split("\n"):
        print(f"   {line}")

    # 6. Save
    print(f"\n[6/6] Saving model to: {MODEL_OUTPUT}")
    joblib.dump(classifier, MODEL_OUTPUT)
    print("\nTraining complete! Model saved successfully.")
    print("=" * 50)


if __name__ == "__main__":
    train()
