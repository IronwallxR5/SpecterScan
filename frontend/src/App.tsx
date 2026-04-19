import { useState } from "react";
import "./App.css";
import { UploadView } from "./components/UploadView/UploadView";
import { ResultsView } from "./components/ResultsView/ResultsView";

// ── Type Definitions (mirrors backend Pydantic schema) ────────────────────────

export interface RiskItem {
  clause_index: number;
  clause_text: string;
  severity: "High" | "Medium" | "Low";
  explanation: string;
  mitigation: string;
}

export interface StructuredReport {
  summary: string;
  risks: RiskItem[];
  disclaimer: string;
}

export interface AnalysisResponse {
  filename: string;
  original_text: string; // Full raw extracted text for the DocumentViewer
  report: StructuredReport;
}

const DEMO_ANALYSIS: AnalysisResponse = {
  filename: "demo-master-services-agreement.txt",
  original_text: `MASTER SERVICES AGREEMENT

1. Scope of Services
Provider will deliver implementation and support services as outlined in Exhibit A.

2. Payment Terms
Client agrees to pay all invoices within 15 calendar days of receipt.

3. Indemnity
The Client shall indemnify and hold harmless Provider from any and all claims, damages, losses, and expenses arising out of this agreement, including attorney fees.

4. Termination
Provider may terminate this agreement immediately at its sole discretion without prior notice and without liability.

5. Governing Law
Any dispute under this agreement shall be resolved exclusively in the courts of an out-of-state jurisdiction selected by Provider.

6. Confidentiality
Both parties agree to keep all non-public information confidential for a period of three years.

7. Limitation of Liability
Neither party will be liable for indirect or consequential damages except where prohibited by law.

End of Agreement`,
  report: {
    summary:
      "The contract includes three clauses with elevated legal and operational risk, primarily related to indemnity scope, unilateral termination authority, and restrictive dispute venue.",
    risks: [
      {
        clause_index: 3,
        clause_text:
          "The Client shall indemnify and hold harmless Provider from any and all claims, damages, losses, and expenses arising out of this agreement, including attorney fees.",
        severity: "High",
        explanation:
          "This clause places broad and uncapped liability on the client, including legal defense costs.",
        mitigation:
          "Limit indemnity to third-party claims caused by client negligence and add reasonable caps.",
      },
      {
        clause_index: 4,
        clause_text:
          "Provider may terminate this agreement immediately at its sole discretion without prior notice and without liability.",
        severity: "High",
        explanation:
          "The provider can exit at any time without warning, creating serious delivery and continuity risk.",
        mitigation:
          "Require notice period and permit immediate termination only for material breach.",
      },
      {
        clause_index: 5,
        clause_text:
          "Any dispute under this agreement shall be resolved exclusively in the courts of an out-of-state jurisdiction selected by Provider.",
        severity: "Medium",
        explanation:
          "This venue requirement may increase cost and complexity for dispute resolution.",
        mitigation:
          "Use a mutually agreed neutral venue or arbitration framework.",
      },
    ],
    disclaimer:
      "Demo output for product walkthrough only. Final legal review should be performed by qualified counsel.",
  },
};

// ── App ───────────────────────────────────────────────────────────────────────

function App() {
  const [currentView, setCurrentView] = useState<"upload" | "results">("upload");
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyze = async (file: File) => {
    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:7860";
      const response = await fetch(`${apiUrl}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Analysis failed. Please try again.");
      }

      const data: AnalysisResponse = await response.json();
      setAnalysisData(data);
      setCurrentView("results");
    } catch (err: unknown) {
      console.error(err);
      const message = err instanceof Error ? err.message : "An unexpected error occurred.";
      alert(message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleBack = () => {
    setCurrentView("upload");
    setAnalysisData(null);
  };

  const handleDemoMode = () => {
    setAnalysisData(DEMO_ANALYSIS);
    setCurrentView("results");
  };

  return (
    <>
      {currentView === "upload" ? (
        <UploadView
          onAnalyze={handleAnalyze}
          isAnalyzing={isAnalyzing}
          onDemoMode={handleDemoMode}
        />
      ) : (
        <ResultsView data={analysisData} onBack={handleBack} />
      )}
    </>
  );
}

export default App;
