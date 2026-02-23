import styles from './DocumentViewer.module.css';
import type { ClauseResult } from '../../App';

interface DocumentViewerProps {
  results: ClauseResult[];
}

export function DocumentViewer({ results }: DocumentViewerProps) {
  return (
    <div className={styles.viewerContainer}>
      <div className={styles.documentBody}>
        {results.map((clause, idx) => {
          const isRisky = clause.risk_label === 1;
          return (
            <span
              key={idx}
              className={isRisky ? styles.highlightHigh : ''}
              style={{ paddingRight: '0.25rem', lineHeight: '1.8' }}
            >
              {clause.clause_text}
            </span>
          );
        })}

        <div className={styles.documentFooter}>
          <p>End of Document</p>
        </div>
      </div>
    </div>
  );
}
