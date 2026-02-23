import { ArrowLeft } from 'lucide-react';
import styles from './ResultsView.module.css';
import { DocumentViewer } from '../DocumentViewer/DocumentViewer';
import { ClausesList } from '../ClausesList/ClausesList';
import type { AnalysisResponse } from '../../App';

interface ResultsViewProps {
  data: AnalysisResponse | null;
  onBack: () => void;
}

export function ResultsView({ data, onBack }: ResultsViewProps) {
  if (!data) return null;

  // Calculate a simple risk score (e.g., percentage of risky clauses)
  const totalClauses = data.total_clauses;
  const riskyClauses = data.results.filter((c) => c.risk_label === 1).length;
  const totalRiskScore = totalClauses > 0 ? (riskyClauses / totalClauses).toFixed(2) : '0.00';

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <button className={styles.backBtn} onClick={onBack}>
            <ArrowLeft size={20} />
          </button>
          <div className={styles.fileInfo}>
            <h2>Analysis Results</h2>
            <p className={styles.fileName}>{data.filename}</p>
          </div>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.summaryBadge}>
            <span className={styles.badgeLabel}>Total Risk Score</span>
            <span className={styles.badgeValue}>{totalRiskScore}</span>
          </div>
        </div>
      </header>

      <main className={styles.splitLayout}>
        <section className={styles.leftColumn}>
          <h3 className={styles.columnTitle}>Document Content</h3>
          <DocumentViewer results={data.results} />
        </section>

        <section className={styles.rightColumn}>
          <h3 className={styles.columnTitle}>Flagged Clauses List</h3>
          <ClausesList results={data.results} />
        </section>
      </main>
    </div>
  );
}
