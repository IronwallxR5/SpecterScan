import { useState } from 'react';
import './App.css';
import { UploadView } from './components/UploadView/UploadView';
import { ResultsView } from './components/ResultsView/ResultsView';

function App() {
  const [currentView, setCurrentView] = useState<'upload' | 'results'>('upload');
  const [fileName, setFileName] = useState<string>('');

  const handleAnalyze = (name: string) => {
    setFileName(name);
    setCurrentView('results');
  };

  return (
    <>
      {currentView === 'upload' ? (
        <UploadView onAnalyze={handleAnalyze} />
      ) : (
        <ResultsView fileName={fileName} onBack={() => setCurrentView('upload')} />
      )}
    </>
  );
}

export default App;
