import styles from './DocumentViewer.module.css';

export function DocumentViewer() {
  return (
    <div className={styles.viewerContainer}>
      <div className={styles.documentBody}>
        <p>This NON-DISCLOSURE AGREEMENT (this <strong>"Agreement"</strong>) is entered into as of January 1, 2024, by and between Company A ("Disclosing Party") and Company B ("Receiving Party").</p>
        
        <h4>1. Definition of Confidential Information</h4>
        <p>For purposes of this Agreement, "Confidential Information" shall include all information or material that has or could have commercial value or other utility in the business in which Disclosing Party is engaged.</p>

        <h4>2. Exclusions from Confidential Information</h4>
        <p>Receiving Party's obligations under this Agreement do not extend to information that is: (a) publicly known at the time of disclosure or subsequently becomes publicly known through no fault of the Receiving Party; (b) discovered or created by the Receiving Party before disclosure by Disclosing Party; (c) learned by the Receiving Party through legitimate means other than from the Disclosing Party or Disclosing Party's representatives; or (d) is disclosed by Receiving Party with Disclosing Party's prior written approval.</p>

        <h4>3. Obligations of Receiving Party</h4>
        <p>Receiving Party shall hold and maintain the Confidential Information in strictest confidence for the sole and exclusive benefit of the Disclosing Party. <span className={styles.highlightHigh}>Receiving Party shall be liable for any punitive damages and consequential damages resulting from any breach, regardless of intent.</span> Receiving Party shall carefully restrict access to Confidential Information to employees, contractors and third parties as is reasonably required.</p>

        <h4>4. Term and Termination</h4>
        <p>The nondisclosure provisions of this Agreement shall survive the termination of this Agreement and Receiving Party's duty to hold Confidential Information in confidence shall remain in effect until the Confidential Information no longer qualifies as a trade secret or until Disclosing Party sends Receiving Party written notice releasing Receiving Party from this Agreement, whichever occurs first. <span className={styles.highlightMedium}>Either party may terminate this Agreement at any time upon three (3) days written notice to the other party.</span></p>

        <h4>5. Governing Law</h4>
        <p>This Agreement shall be governed by and construed in accordance with the laws of the State. Any disputes arising from this Agreement shall be resolved in arbitration.</p>
        
        <div className={styles.documentFooter}>
          <p>End of Document</p>
        </div>
      </div>
    </div>
  );
}
