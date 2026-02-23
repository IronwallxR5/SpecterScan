import { AlertTriangle, AlertCircle } from 'lucide-react';
import styles from './ClausesList.module.css';

export interface ClauseData {
  id: string;
  snippet: string;
  riskLevel: 'High' | 'Medium' | 'Low';
  riskScore: number;
}

interface ClauseCardProps {
  clause: ClauseData;
}

export function ClauseCard({ clause }: ClauseCardProps) {
  const getRiskIcon = () => {
    switch (clause.riskLevel) {
      case 'High':
        return <AlertTriangle size={16} className={styles.iconHigh} />;
      case 'Medium':
        return <AlertCircle size={16} className={styles.iconMedium} />;
      default:
        return null;
    }
  };

  return (
    <div className={`${styles.card} ${styles[`risk${clause.riskLevel}`]}`}>
      <div className={styles.cardHeader}>
        <div className={styles.riskBadgeWrapper}>
          {getRiskIcon()}
          <span className={styles.riskLabel}>{clause.riskLevel} Risk</span>
        </div>
        <div className={styles.scoreWrapper}>
          <span className={styles.scoreLabel}>Score</span>
          <span className={styles.scoreValue}>{clause.riskScore.toFixed(2)}</span>
        </div>
      </div>
      <div className={styles.cardBody}>
        <p className={styles.snippet}>"{clause.snippet}"</p>
      </div>
      <div className={styles.cardFooter}>
        <button className={styles.actionBtn}>Review Section</button>
      </div>
    </div>
  );
}
