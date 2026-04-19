import { useState } from 'react';
import { AlertTriangle, AlertCircle, Info, Sparkles, Loader2 } from 'lucide-react';
import styles from './ClausesList.module.css';
import type { RiskItem } from '../../App';

interface ClauseCardProps {
  clause: RiskItem;
}

const SEVERITY_CONFIG = {
  High:   { icon: AlertTriangle, cardClass: 'riskHigh',   badgeClass: 'badgeHigh',   iconClass: 'iconHigh'   },
  Medium: { icon: AlertCircle,   cardClass: 'riskMedium', badgeClass: 'badgeMedium', iconClass: 'iconMedium' },
  Low:    { icon: Info,          cardClass: 'riskLow',    badgeClass: 'badgeLow',    iconClass: 'iconLow'    },
} as const;

export function ClauseCard({ clause }: ClauseCardProps) {
  const [explanation, setExplanation]     = useState<string | null>(null);
  const [isLoading, setIsLoading]         = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const config = SEVERITY_CONFIG[clause.severity] ?? SEVERITY_CONFIG['Medium'];
  const SeverityIcon = config.icon;

  const handleExplain = async () => {
    // If already loaded, toggle visibility
    if (explanation !== null) {
      setShowExplanation((prev) => !prev);
      return;
    }

    setIsLoading(true);
    setShowExplanation(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:7860';
      const res = await fetch(`${apiUrl}/explain_clause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clause_text: clause.clause_text }),
      });

      if (!res.ok) {
        throw new Error('Server returned an error. Please try again.');
      }

      const data = await res.json();
      setExplanation(data.explanation);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load explanation.';
      setExplanation(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`${styles.card} ${styles[config.cardClass]}`}>
      {/* Card Header */}
      <div className={styles.cardHeader}>
        <div className={`${styles.riskBadgeWrapper} ${styles[config.badgeClass]}`}>
          <SeverityIcon size={14} className={styles[config.iconClass]} />
          <span className={styles.riskLabel}>{clause.severity} Risk</span>
        </div>
        <div className={styles.scoreWrapper}>
          <span className={styles.scoreLabel}>Clause</span>
          <span className={styles.scoreValue}>#{clause.clause_index}</span>
        </div>
      </div>

      {/* Clause Text */}
      <div className={styles.cardBody}>
        <p className={styles.snippet}>"{clause.clause_text}"</p>
      </div>

      {/* AI Explanation (from classifier) */}
      {clause.explanation && (
        <div className={styles.infoSection}>
          <p className={styles.infoLabel}>Why it's risky</p>
          <p className={styles.infoText}>{clause.explanation}</p>
        </div>
      )}

      {/* Mitigation */}
      {clause.mitigation && (
        <div className={`${styles.infoSection} ${styles.mitigationSection}`}>
          <p className={styles.infoLabel}>Suggested action</p>
          <p className={styles.infoText}>{clause.mitigation}</p>
        </div>
      )}

      {/* Footer: Explain Clause Button */}
      <div className={styles.cardFooter}>
        <button
          className={styles.actionBtn}
          onClick={handleExplain}
          disabled={isLoading}
          aria-label={`Get plain-English explanation for clause ${clause.clause_index}`}
        >
          {isLoading ? (
            <Loader2 size={14} className={styles.spinnerIcon} />
          ) : (
            <Sparkles size={14} />
          )}
          {explanation && !isLoading
            ? (showExplanation ? 'Hide Explanation' : 'Show Explanation')
            : 'Explain to Me'}
        </button>
      </div>

      {/* AI plain-English explanation box */}
      {showExplanation && (
        <div className={styles.explanationBox}>
          {isLoading ? (
            <div className={styles.loadingRow}>
              <Loader2 size={16} className={styles.spinnerIcon} />
              <span>Asking AI to explain this clause…</span>
            </div>
          ) : (
            <>
              <p className={styles.explanationLabel}>Plain-English Explanation</p>
              <p className={styles.explanationText}>{explanation}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
