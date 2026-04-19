import { ClauseCard } from './ClauseCard';
import { CheckCircle } from 'lucide-react';
import styles from './ClausesList.module.css';
import type { RiskItem } from '../../App';

interface ClausesListProps {
  risks: RiskItem[];
}

export function ClausesList({ risks }: ClausesListProps) {
  const highCount   = risks.filter((r) => r.severity === 'High').length;
  const mediumCount = risks.filter((r) => r.severity === 'Medium').length;
  const lowCount    = risks.filter((r) => r.severity === 'Low').length;

  return (
    <div className={styles.listContainer}>
      <div className={styles.listHeader}>
        <span className={styles.count}>{risks.length} Flagged Clauses</span>
        <div className={styles.severityPills}>
          {highCount > 0   && <span className={`${styles.pill} ${styles.pillHigh}`}>{highCount} High</span>}
          {mediumCount > 0 && <span className={`${styles.pill} ${styles.pillMedium}`}>{mediumCount} Med</span>}
          {lowCount > 0    && <span className={`${styles.pill} ${styles.pillLow}`}>{lowCount} Low</span>}
        </div>
      </div>

      <div className={styles.cardsWrapper}>
        {risks.length === 0 ? (
          <div className={styles.emptyState}>
            <span className={styles.emptyIcon}><CheckCircle size={40} className={styles.emptyIconSvg} /></span>
            <p>No risky clauses detected!</p>
            <p className={styles.emptySubtext}>This contract appears to be compliant.</p>
          </div>
        ) : (
          risks.map((risk) => (
            <ClauseCard key={risk.clause_index} clause={risk} />
          ))
        )}
      </div>
    </div>
  );
}
