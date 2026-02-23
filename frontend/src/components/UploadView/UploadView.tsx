import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle, Loader2 } from 'lucide-react';
import styles from './UploadView.module.css';

interface UploadViewProps {
  onAnalyze: (file: File) => void;
  isAnalyzing: boolean;
}

export function UploadView({ onAnalyze, isAnalyzing }: UploadViewProps) {
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
        <div className={styles.logo}>
          <div className={styles.logoIcon}></div>
          <h1>SpecterScan</h1>
        </div>
        <p>Contract Risk Classification System</p>
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
                <h3>Upload Contract Document</h3>
                <p>Drag and drop your PDF or text file here, or click to browse</p>
                <span className={styles.supportedFormats}>Supports .pdf, .txt</span>
              </>
            )}
          </div>
        </div>

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
      </main>
    </div>
  );
}
