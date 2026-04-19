import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle, Loader2, ShieldCheck, Sparkles } from 'lucide-react';
import styles from './UploadView.module.css';

interface UploadViewProps {
  onAnalyze: (file: File) => void;
  isAnalyzing: boolean;
  onDemoMode: () => void;
}

export function UploadView({ onAnalyze, isAnalyzing, onDemoMode }: UploadViewProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (isAnalyzing) return;
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    if (isAnalyzing) return;
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (isAnalyzing) return;
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleClick = () => {
    if (isAnalyzing) return;
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.brandRow}>
          <div className={styles.logo}>
            <div className={styles.logoIcon}></div>
            <h1>SpecterScan</h1>
          </div>
          <span className={styles.badge}>AI Contract Intelligence</span>
        </div>
        <h2>Simple upload. Precise post-processing. Professional highlights.</h2>
        <p>
          Upload your agreement and get a clean, structured risk report with severity-tagged
          clauses and plain-language mitigation guidance.
        </p>

        <div className={styles.heroMeta}>
          <span><ShieldCheck size={16} /> Reliable clause classification</span>
          <span><Sparkles size={16} /> Executive-ready output</span>
        </div>
      </header>

      <main className={styles.main}>
        <div
          className={`${styles.dropzone} ${isDragging ? styles.dragging : ''} ${file ? styles.hasFile : ''} ${isAnalyzing ? styles.analyzing : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleClick}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf,.txt"
            style={{ display: 'none' }}
            disabled={isAnalyzing}
          />

          <div className={styles.dropContent}>
            {file ? (
              <div className={styles.fileInfo}>
                <FileText className={styles.iconFile} size={48} />
                <h3>{file.name}</h3>
                <p className={styles.fileSize}>{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                <div className={styles.readyBadge}>
                  <CheckCircle size={16} /> Ready for analysis
                </div>
              </div>
            ) : (
              <>
                <div className={styles.iconCircle}>
                  <UploadCloud className={styles.icon} size={32} />
                </div>
                <h3>Upload contract document</h3>
                <p>Drag and drop your PDF or TXT file, or click to browse.</p>
                <span className={styles.supportedFormats}>Supports .pdf, .txt</span>
              </>
            )}
          </div>
        </div>

        <div className={styles.actionsRow}>
          <button
            className={styles.analyzeBtn}
            disabled={!file || isAnalyzing}
            onClick={() => file ? onAnalyze(file) : undefined}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className={styles.spinner} size={20} />
                Analyzing...
              </>
            ) : (
              'Analyze Document'
            )}
          </button>

          <button
            className={styles.demoBtn}
            disabled={isAnalyzing}
            onClick={onDemoMode}
            type="button"
          >
            View Demo Mode
          </button>
        </div>
      </main>
    </div>
  );
}
