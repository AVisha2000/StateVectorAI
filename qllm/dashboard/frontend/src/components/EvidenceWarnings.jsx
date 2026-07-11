import { uniqueWarnings } from '../evidenceView'

export default function EvidenceWarnings({ warnings, compact = false }) {
  const visible = uniqueWarnings(warnings)
  if (!visible.length) return null
  return (
    <section className={`evidence-warnings ${compact ? 'compact' : ''}`} aria-label="Interpretation warnings">
      {visible.map((warning, index) => (
        <div className={`evidence-warning ${warning.severity || 'warning'}`} key={`${warning.code || 'warning'}-${index}`}>
          <div className="evidence-warning-heading">
            <span className={`badge ${warning.severity === 'error' ? 'error' : 'queued'}`}>{warning.severity || 'warning'}</span>
            <b>{warning.title || warning.code || 'Interpretation warning'}</b>
            {warning.code && <span className="mono warning-code">{warning.code}</span>}
          </div>
          {!compact && <p>{warning.message || 'Recorded evidence requires review.'}</p>}
          {!compact && warning.evidence != null && (
            <div className="warning-evidence mono">{JSON.stringify(warning.evidence)}</div>
          )}
        </div>
      ))}
    </section>
  )
}
