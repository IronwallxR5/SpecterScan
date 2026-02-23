import { ClauseCard } from './ClauseCard';
import styles from './ClausesList.module.css';
import type { ClauseResult } from '../../App';

interface ClausesListProps {
  results: ClauseResult[];
}

export function ClausesList({ results }: ClausesListProps) {
  // Only show the risky clauses in the side panel
  const flaggedClauses = results.filter((clause) => clause.risk_label === 1);

  return (
    <div className={styles.listContainer}>
      <div className={styles.listHeader}>
        <span className={styles.count}>{flaggedClauses.length} Flagged Clauses</span>
        <button className={styles.filterBtn}></button>
      </div>
      <div className={styles.cardsWrapper}>
        {flaggedClauses.length === 0 ? (
          <p className={styles.emptyState}>No risky clauses detected!</p>
        ) : (
          flaggedClauses.map((clause) => (
            <ClauseCard key={clause.clause_index} clause={clause} />
          ))
        )}
      </div>
    </div>
  );
}
