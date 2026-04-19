import { ArrowLeft, BadgeCheck, ShieldAlert } from 'lucide-react';
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

  const { report, original_text, filename } = data;
  const riskCount = report.risks.length;
  const highCount = report.risks.filter((risk) => risk.severity === 'High').length;
  const mediumCount = report.risks.filter((risk) => risk.severity === 'Medium').length;
  const lowCount = report.risks.filter((risk) => risk.severity === 'Low').length;
  const riskSeverityColor = riskCount === 0 ? 'green' : riskCount <= 3 ? 'amber' : 'red';

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerMain}>
          <div className={styles.headerLeft}>
            <button className={styles.backBtn} onClick={onBack} aria-label="Back to upload">
              <ArrowLeft size={20} />
            </button>
            <div className={styles.fileInfo}>
              <h2>Post-Processing Report</h2>
              <p className={styles.fileName}>{filename}</p>
            </div>
          </div>

          <div className={styles.headerRight}>
            <div className={`${styles.summaryBadge} ${styles[`severity_${riskSeverityColor}`]}`}>
              <ShieldAlert size={16} className={styles.badgeIcon} />
              <div>
                <span className={styles.badgeLabel}>Total Flags</span>
                <span className={styles.badgeValue}>{riskCount}</span>
              </div>
            </div>
          </div>
        </div>

        <p className={styles.processingStatus}>
          <BadgeCheck size={16} />
          Post-processing complete. Highlights and recommendations are ready.
        </p>

        <div className={styles.statsRow}>
          <div className={styles.statCard}>
            <span className={styles.statLabel}>High Risk</span>
            <span className={styles.statValue}>{highCount}</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statLabel}>Medium Risk</span>
            <span className={styles.statValue}>{mediumCount}</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statLabel}>Low Risk</span>
            <span className={styles.statValue}>{lowCount}</span>
          </div>
        </div>
      </header>

      {report.summary && (
        <div className={styles.summaryBanner}>
          <p className={styles.summaryText}>
            <strong>AI Summary:</strong> {report.summary}
          </p>
        </div>
      )}

      <main className={styles.splitLayout}>
        <section className={styles.leftColumn}>
          <h3 className={styles.columnTitle}>Document Content</h3>
          <DocumentViewer originalText={original_text} risks={report.risks} />
        </section>

        <section className={styles.rightColumn}>
          <h3 className={styles.columnTitle}>Flagged Clauses</h3>
          <ClausesList risks={report.risks} />
        </section>
      </main>

      <footer className={styles.disclaimer}>
        <p>{report.disclaimer}</p>
      </footer>
    </div>
  );
}
