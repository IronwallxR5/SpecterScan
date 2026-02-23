import { ArrowLeft } from 'lucide-react';
import styles from './ResultsView.module.css';
import { DocumentViewer } from '../DocumentViewer/DocumentViewer';
import { ClausesList } from '../ClausesList/ClausesList';

interface ResultsViewProps {
  fileName: string;
  onBack: () => void;
}

export function ResultsView({ fileName, onBack }: ResultsViewProps) {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <button className={styles.backBtn} onClick={onBack}>
            <ArrowLeft size={20} />
          </button>
          <div className={styles.fileInfo}>
            <h2>Analysis Results</h2>
            <p className={styles.fileName}>{fileName}</p>
          </div>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.summaryBadge}>
            <span className={styles.badgeLabel}>Total Risk Score</span>
            <span className={styles.badgeValue}>0.87</span>
          </div>
        </div>
      </header>

      <main className={styles.splitLayout}>
        <section className={styles.leftColumn}>
          <h3 className={styles.columnTitle}>Document Content</h3>
          <DocumentViewer />
        </section>
        
        <section className={styles.rightColumn}>
          <h3 className={styles.columnTitle}>Flagged Clauses List</h3>
          <ClausesList />
        </section>
      </main>
    </div>
  );
}
