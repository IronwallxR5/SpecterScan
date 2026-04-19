"""
agent.py
========
LangGraph state machine for SpecterScan Milestone 2.

Graph:
  extract_risks  →  [conditional]
      ├── no flagged clauses  →  no_risks_found  →  END
      └── has flagged clauses →  retrieve_context  →  synthesize_report  →  END
"""

import logging
from typing import TypedDict, List

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("specterscan.agent")


# ── Pydantic Output Schema ────────────────────────────────────────────────────

class RiskItem(BaseModel):
    clause_index: int = Field(
        description="The 1-based index of the clause in the document."
    )
    clause_text: str = Field(
        description="Verbatim text of the risky clause."
    )
    severity: str = Field(
        description="Risk severity level. Must be exactly one of: 'High', 'Medium', or 'Low'."
    )
    explanation: str = Field(
        description="Plain-English explanation of why this clause is risky, written for a non-lawyer."
    )
    mitigation: str = Field(
        description="Concrete, actionable step the user should take to address this risk."
    )


class StructuredReport(BaseModel):
    summary: str = Field(
        description="A 2–3 sentence executive summary of the contract's overall risk profile."
    )
    risks: List[RiskItem] = Field(
        description="A list of risk objects, one per flagged clause."
    )
    disclaimer: str = Field(
        description="Standard disclaimer that this report is AI-generated and not legal advice."
    )


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    contract_text: str       # Raw extracted text from the uploaded document
    all_clauses: list        # All clauses: [{clause_index, clause_text, risk_label}]
    flagged_clauses: list    # Only the clauses where risk_label == 1
    retrieved_context: str   # Top-k legal guidelines retrieved from Pinecone
    structured_report: dict  # Final JSON-serialisable report (StructuredReport.model_dump())
    errors: list             # Non-fatal warnings / errors accumulate here


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(embedder, classifier, nlp, rag_embedder, pinecone_index, groq_llm):
    """
    Build and compile the LangGraph graph.

    Parameters
    ----------
    embedder        : SentenceTransformer (all-MiniLM-L6-v2) — for spaCy clause classification
    classifier      : scikit-learn model loaded from legal_risk_classifier.pkl
    nlp             : spaCy Language model (en_core_web_sm)
    rag_embedder    : SentenceTransformer (BAAI/bge-large-en-v1.5) — MUST match Pinecone index dims
    pinecone_index  : Pinecone Index object for the 'specterscan' index
    groq_llm        : ChatGroq instance (llama3-8b-8192, configured without structured output)
    """

    # ── Node 1: Extract Risks ─────────────────────────────────────────────────
    def extract_risks(state: AgentState) -> dict:
        """
        Segment the raw contract text into clauses via spaCy, embed them with
        all-MiniLM-L6-v2, and classify each clause with the sklearn model.
        Populates both all_clauses and flagged_clauses in the state.
        """
        logger.info("[Node 1 — Extractor] Starting clause segmentation and classification...")
        errors = list(state.get("errors", []))

        try:
            doc = nlp(state["contract_text"])
            clauses = [s.text.strip() for s in doc.sents if len(s.text.strip()) >= 5]

            if not clauses:
                errors.append("spaCy produced no clauses from the document text.")
                return {"all_clauses": [], "flagged_clauses": [], "errors": errors}

            embeddings  = embedder.encode(clauses, show_progress_bar=False)
            predictions = classifier.predict(embeddings)

            all_clauses:     list = []
            flagged_clauses: list = []

            for idx, (text, pred) in enumerate(zip(clauses, predictions), start=1):
                entry = {
                    "clause_index": idx,
                    "clause_text":  text,
                    "risk_label":   int(pred),
                }
                all_clauses.append(entry)
                if int(pred) == 1:
                    flagged_clauses.append(entry)

            logger.info(
                f"[Extractor] Done — {len(flagged_clauses)} risky / {len(all_clauses)} total clauses."
            )
            return {"all_clauses": all_clauses, "flagged_clauses": flagged_clauses, "errors": errors}

        except Exception as e:
            logger.error(f"[Extractor] Failed: {e}")
            errors.append(f"Extraction error: {str(e)}")
            return {"all_clauses": [], "flagged_clauses": [], "errors": errors}


    # ── Conditional Router ────────────────────────────────────────────────────
    def route_after_extraction(state: AgentState) -> str:
        if state.get("flagged_clauses"):
            return "retrieve_context"
        return "no_risks_found"


    # ── Node 2: Retrieve Context (RAG) ────────────────────────────────────────
    def retrieve_context(state: AgentState) -> dict:
        """
        Embed the flagged clauses using BAAI/bge-large-en-v1.5 (matching the
        Pinecone index dimensions of 1024), query for the top 3 most similar
        legal guidelines, and store them as a formatted string.

        Fails gracefully — a Pinecone outage will not crash the pipeline.
        """
        logger.info("[Node 2 — Retriever] Querying Pinecone for similar legal clauses...")
        errors = list(state.get("errors", []))

        try:
            # Combine up to 5 flagged clauses into one query string
            query_text = " ".join(
                c["clause_text"] for c in state["flagged_clauses"][:5]
            )
            query_embedding = rag_embedder.encode(query_text).tolist()

            results = pinecone_index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
            )

            docs = [
                match["metadata"].get("clause_text", "")
                for match in results.get("matches", [])
                if match.get("metadata", {}).get("clause_text")
            ]
            retrieved_context = "\n---\n".join(docs) if docs else ""
            logger.info(f"[Retriever] Retrieved {len(docs)} context documents from Pinecone.")

        except Exception as e:
            logger.warning(f"[Retriever] Pinecone query failed (non-fatal): {e}")
            errors.append(f"Retrieval warning: {str(e)}")
            retrieved_context = ""

        return {"retrieved_context": retrieved_context, "errors": errors}


    # ── Node 3: Synthesize Report ─────────────────────────────────────────────
    def synthesize_report(state: AgentState) -> dict:
        """
        Call Groq (llama3-8b-8192) with with_structured_output(StructuredReport)
        to produce strict JSON output. Falls back to a classifier-based report
        if the LLM call fails, so the API always returns something useful.
        """
        logger.info("[Node 3 — Synthesizer] Calling Groq llama3-8b-8192...")
        errors = list(state.get("errors", []))

        flagged_text = "\n".join(
            f"[Clause {c['clause_index']}] {c['clause_text']}"
            for c in state["flagged_clauses"]
        )
        context_block = (
            f"\n\nRelevant similar legal clauses from our knowledge base for reference:\n"
            f"{state['retrieved_context']}"
            if state["retrieved_context"]
            else ""
        )

        system_prompt = (
            "You are SpecterScan, an expert legal risk analysis AI. "
            "Analyze the provided flagged contract clauses and produce a structured JSON risk report. "
            "For each clause, assign a severity (High, Medium, or Low), explain the risk clearly "
            "for a non-lawyer audience, and suggest a concrete mitigation action. "
            "Always end with a standard legal AI disclaimer."
        )
        user_prompt = (
            f"Analyze the following flagged contract clauses and produce a structured risk report."
            f"{context_block}\n\n"
            f"Flagged Clauses to analyze:\n{flagged_text}"
        )

        try:
            structured_llm = groq_llm.with_structured_output(StructuredReport)
            report: StructuredReport = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            logger.info("[Synthesizer] Structured report generated successfully.")
            return {"structured_report": report.model_dump(), "errors": errors}

        except Exception as e:
            logger.error(f"[Synthesizer] LLM call failed: {e}")
            errors.append(f"Synthesis error: {str(e)}")

            # Graceful fallback — classifier results without LLM explanation
            fallback = StructuredReport(
                summary=(
                    "The AI synthesis step encountered an error. The following clauses were "
                    "flagged by the ML classifier as potentially risky. Please review them manually."
                ),
                risks=[
                    RiskItem(
                        clause_index=c["clause_index"],
                        clause_text=c["clause_text"],
                        severity="High",
                        explanation=(
                            "This clause was flagged by the ML classifier as potentially risky. "
                            "Use the 'Explain Clause' button for an AI explanation."
                        ),
                        mitigation="Consult a qualified legal professional to review this clause.",
                    )
                    for c in state["flagged_clauses"]
                ],
                disclaimer=(
                    "This report is generated by an AI system and does not constitute legal advice. "
                    "Always consult a qualified attorney before signing any legal agreement."
                ),
            )
            return {"structured_report": fallback.model_dump(), "errors": errors}


    # ── No-Risks Fast Path ────────────────────────────────────────────────────
    def no_risks_found(state: AgentState) -> dict:
        """
        If the ML classifier found zero risky clauses, skip the LLM entirely
        and return a clean report. Saves tokens and latency.
        """
        logger.info("[no_risks_found] No risky clauses detected — skipping LLM call.")
        clean_report = StructuredReport(
            summary=(
                "No risky clauses were detected in this contract. "
                "The document appears to be standard and compliant based on our analysis."
            ),
            risks=[],
            disclaimer=(
                "This report is generated by an AI system and does not constitute legal advice. "
                "Always consult a qualified attorney before signing any legal agreement."
            ),
        )
        return {"structured_report": clean_report.model_dump()}


    # ── Assemble Graph ────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("extract_risks",    extract_risks)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("synthesize_report", synthesize_report)
    graph.add_node("no_risks_found",   no_risks_found)

    graph.set_entry_point("extract_risks")

    graph.add_conditional_edges(
        "extract_risks",
        route_after_extraction,
        {
            "retrieve_context": "retrieve_context",
            "no_risks_found":   "no_risks_found",
        },
    )

    graph.add_edge("retrieve_context",   "synthesize_report")
    graph.add_edge("synthesize_report",  END)
    graph.add_edge("no_risks_found",     END)

    return graph.compile()
