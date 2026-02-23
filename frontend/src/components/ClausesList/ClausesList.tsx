import { ClauseCard, type ClauseData } from './ClauseCard';
import styles from './ClausesList.module.css';

const mockClauses: ClauseData[] = [
  {
    id: 'c1',
    snippet: 'Receiving Party shall be liable for any punitive damages and consequential damages resulting from any breach, regardless of intent.',
    riskLevel: 'High',
    riskScore: 0.92
  },
  {
    id: 'c2',
    snippet: 'Either party may terminate this Agreement at any time upon three (3) days written notice to the other party.',
    riskLevel: 'Medium',
    riskScore: 0.68
  }
];

export function ClausesList() {
  return (
    <div className={styles.listContainer}>
      <div className={styles.listHeader}>
        <span className={styles.count}>{mockClauses.length} Flagged Clauses</span>
        <button className={styles.filterBtn}>Filter</button>
      </div>
      <div className={styles.cardsWrapper}>
        {mockClauses.map(clause => (
          <ClauseCard key={clause.id} clause={clause} />
        ))}
      </div>
    </div>
  );
}
