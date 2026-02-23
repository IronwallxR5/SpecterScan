import { useState } from 'react';
import './App.css';
import { UploadView } from './components/UploadView/UploadView';
import { ResultsView } from './components/ResultsView/ResultsView';

export interface ClauseResult {
  clause_index: number;
  clause_text: string;
  risk_label: number;
  risk_category: string;
}

export interface AnalysisResponse {
  filename: string;
  total_clauses: number;
  results: ClauseResult[];
}

function App() {
  const [currentView, setCurrentView] = useState<'upload' | 'results'>('upload');
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed. Please try again.');
      }

      const data: AnalysisResponse = await response.json();
      setAnalysisData(data);
      setCurrentView('results');
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An unexpected error occurred.');
      alert(err.message || 'An unexpected error occurred.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleBack = () => {
    setCurrentView('upload');
    setAnalysisData(null);
  };

  return (
    <>
      {currentView === 'upload' ? (
        <UploadView onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
      ) : (
        <ResultsView data={analysisData} onBack={handleBack} />
      )}
    </>
  );
}

export default App;
