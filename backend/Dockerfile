# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set work directory
WORKDIR /app

# Install system-level build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy model (small, fast — this is fine at build time)
RUN python -m spacy download en_core_web_sm

# Copy the full backend source
# NOTE: chroma_db/ is no longer used — RAG is served by Pinecone.
# PINECONE_API_KEY and GROQ_API_KEY must be set as Hugging Face Space Secrets,
# NOT hardcoded here.
COPY . .

# Expose the port HF Spaces expects
EXPOSE 7860

# Start the FastAPI server
CMD ["python", "main.py"]
