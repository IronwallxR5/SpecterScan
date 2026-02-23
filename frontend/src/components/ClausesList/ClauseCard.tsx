import { AlertTriangle } from 'lucide-react';
import styles from './ClausesList.module.css';
import type { ClauseResult } from '../../App';

interface ClauseCardProps {
  clause: ClauseResult;
}

export function ClauseCard({ clause }: ClauseCardProps) {
  // Currently the API only returns label 0 or 1, without a probability score.
  // We'll treat all 1s as "High" risk for this UI step.
  const riskLevel = 'High';

  return (
    <div className={`${styles.card} ${styles[`risk${riskLevel}`]}`}>
      <div className={styles.cardHeader}>
        <div className={styles.riskBadgeWrapper}>
          <AlertTriangle size={16} className={styles.iconHigh} />
          <span className={styles.riskLabel}>{clause.risk_category}</span>
        </div>
        <div className={styles.scoreWrapper}>
          <span className={styles.scoreLabel}>Clause</span>
          <span className={styles.scoreValue}>#{clause.clause_index}</span>
        </div>
      </div>
      <div className={styles.cardBody}>
        <p className={styles.snippet}>"{clause.clause_text}"</p>
      </div>
      <div className={styles.cardFooter}>
        <button className={styles.actionBtn}>Review Section</button>
      </div>
    </div>
  );
}
