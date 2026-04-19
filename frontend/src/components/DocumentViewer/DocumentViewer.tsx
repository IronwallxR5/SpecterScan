import { useMemo } from 'react';
import styles from './DocumentViewer.module.css';
import type { RiskItem } from '../../App';

interface DocumentViewerProps {
  originalText: string;
  risks: RiskItem[];
}

// ── Segment-based text highlighting ──────────────────────────────────────────
type Segment = {
  text: string;
  isRisk: boolean;
  severity?: 'High' | 'Medium' | 'Low';
};

function buildSegments(originalText: string, risks: RiskItem[]): Segment[] {
  // Start with one big un-highlighted segment
  let segments: Segment[] = [{ text: originalText, isRisk: false }];

  for (const risk of risks) {
    const clauseText = risk.clause_text;
    if (!clauseText) continue;

    const updated: Segment[] = [];

    for (const seg of segments) {
      // Already highlighted — skip
      if (seg.isRisk) {
        updated.push(seg);
        continue;
      }

      const idx = seg.text.indexOf(clauseText);

      if (idx === -1) {
        // Not found in this segment — leave as-is
        updated.push(seg);
      } else {
        // Split into: before | match | after
        if (idx > 0) {
          updated.push({ text: seg.text.slice(0, idx), isRisk: false });
        }
        updated.push({ text: clauseText, isRisk: true, severity: risk.severity });
        const after = seg.text.slice(idx + clauseText.length);
        if (after) {
          updated.push({ text: after, isRisk: false });
        }
      }
    }

    segments = updated;
  }

  return segments;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function DocumentViewer({ originalText, risks }: DocumentViewerProps) {
  const segments = useMemo(
    () => buildSegments(originalText, risks),
    [originalText, risks]
  );

  return (
    <div className={styles.viewerContainer}>
      <div className={styles.legendBar}>
        <span className={styles.legendLabel}>Highlight Legend</span>
        <div className={styles.legendItems}>
          <span className={`${styles.legendItem} ${styles.legendHigh}`}>High</span>
          <span className={`${styles.legendItem} ${styles.legendMedium}`}>Medium</span>
          <span className={`${styles.legendItem} ${styles.legendLow}`}>Low</span>
        </div>
      </div>

      <div className={styles.documentBody}>
        {segments.map((seg, idx) => {
          if (!seg.isRisk) {
            return <span key={idx}>{seg.text}</span>;
          }
          const severityClass =
            seg.severity === 'High'
              ? styles.highlightHigh
              : seg.severity === 'Medium'
              ? styles.highlightMedium
              : styles.highlightLow;

          return (
            <mark key={idx} className={`${styles.highlight} ${severityClass}`} title={`${seg.severity} Risk`}>
              {seg.text}
            </mark>
          );
        })}

        <div className={styles.documentFooter}>
          <p>End of Document</p>
        </div>
      </div>
    </div>
  );
}
