import { displayValue } from '../evidenceView'

function fmt(value, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) return 'unavailable'
  return Number(value).toFixed(digits)
}

function Field({ label, value }) {
  return <div><span className="k">{label}</span><span className="v evidence-value">{displayValue(value)}</span></div>
}

function Statistics({ paired, equivalence, power }) {
  if (!paired && !equivalence && !power) return <p className="muted">Paired, equivalence, and power statistics are pending or unavailable.</p>
  return (
    <div className="evidence-stat-grid">
      <div>
        <h4>Paired inference</h4>
        <Field label="pairs" value={paired?.n_pairs} />
        <Field label="mean improvement" value={paired?.mean_improvement == null ? null : fmt(paired.mean_improvement)} />
        <Field label="confidence interval" value={paired ? `[${fmt(paired.ci_low)}, ${fmt(paired.ci_high)}]` : null} />
        <Field label="p-value" value={paired?.p_value == null ? null : fmt(paired.p_value)} />
        <Field label="methods" value={paired ? `${paired.ci_method || 'unavailable'} / ${paired.sign_flip_method || 'unavailable'}` : null} />
      </div>
      <div>
        <h4>Practical equivalence</h4>
        <Field label="status" value={equivalence?.status} />
        <Field label="margin" value={equivalence?.margin} />
        <Field label="equivalent" value={equivalence?.equivalent} />
      </div>
      <div>
        <h4>Power plan</h4>
        <Field label="status" value={power?.status || power?.assessment_status} />
        <Field label="observed pairs" value={power?.observed_pairs ?? power?.n_pairs} />
        <Field label="recommended pairs" value={power?.recommended_pairs} />
        <Field label="adequately powered" value={power?.adequately_powered} />
      </div>
    </div>
  )
}

function Fairness({ mismatches }) {
  if (!mismatches?.length) return <p className="muted">No recorded fairness mismatches.</p>
  return (
    <div className="fairness-list">
      {mismatches.map((item, index) => {
        const allowed = item.allowed === true || item.intentional === true || item.status === 'allowed'
        return (
          <div className={`fairness-row ${allowed ? 'allowed' : 'disallowed'}`} key={`${item.path || item.field || 'mismatch'}-${index}`}>
            <span className={`badge ${allowed ? 'done' : 'error'}`}>{allowed ? 'allowed' : 'disallowed'}</span>
            <b>{item.path || item.field || item.key || 'protocol mismatch'}</b>
            <span className="mono evidence-value">{JSON.stringify(item)}</span>
          </div>
        )
      })}
    </div>
  )
}

function Analogue({ ladder, limitations }) {
  const rungs = ladder?.rungs || []
  if (!ladder && !limitations?.length) return <p className="muted">Analogue ladder is pending or unavailable.</p>
  return (
    <>
      <div className="kv compact">
        <div className="k">required complete</div><div className="v">{displayValue(ladder?.required_complete)}</div>
        <div className="k">missing required</div><div className="v evidence-value">{(ladder?.missing_required || []).join(', ') || 'none recorded'}</div>
      </div>
      <div className="analogue-rungs">
        {rungs.map((rung, index) => (
          <div className="comparison-row" key={rung.id || rung.rung_id || index}>
            <div><b>{rung.label || rung.id || rung.rung_id || 'analogue rung'}</b><div className="muted evidence-value">{displayValue(rung.detail ?? rung.limitation, 'No detail recorded.')}</div></div>
            <span className={`badge ${rung.status === 'met' ? 'done' : 'error'}`}>{rung.status || 'unknown'}</span>
          </div>
        ))}
      </div>
      {(limitations || []).map((item, index) => <div className="muted" key={index}>{typeof item === 'string' ? item : JSON.stringify(item)}</div>)}
    </>
  )
}

export default function EvidenceSummary({ evidence, title = 'Evidence contract', showAnalyses = true }) {
  if (!evidence) return <section className="panel evidence-summary"><h3>{title}</h3><p className="muted">Structured evidence is unavailable.</p></section>
  const analyses = showAnalyses ? (evidence.analyses || []) : []
  return (
    <section className="panel evidence-summary">
      <div className="workspace-header">
        <h3>{title}</h3>
        <span className="badge">{evidence.assessment_status || 'unavailable'}</span>
      </div>
      <div className="evidence-identity">
        <Field label="claim ID" value={evidence.claim_id} />
        <Field label="metric type" value={evidence.metric_type} />
        <Field label="claim level" value={evidence.claim?.level || evidence.claim?.claim_level} />
        <Field label="claim status" value={evidence.claim?.status} />
      </div>
      <h4>Statistics</h4>
      <Statistics paired={evidence.paired_stats} equivalence={evidence.equivalence} power={evidence.power} />
      <h4>Fairness mismatches</h4>
      <Fairness mismatches={evidence.fairness_mismatches} />
      <h4>Analogue ladder and limitations</h4>
      <Analogue ladder={evidence.analogue_ladder} limitations={evidence.analogue_limitations} />
      <details>
        <summary>Seed axes</summary>
        <pre className="code-block evidence-raw">{JSON.stringify(evidence.seed_axes ?? null, null, 2)}</pre>
      </details>
      {analyses.length > 0 && (
        <div className="analysis-cells">
          <h4>Separate analysis cells</h4>
          <p className="muted">Cells are reported separately and are never pooled in this view.</p>
          {analyses.map((cell, index) => {
            const scale = cell.sweep?.n_qubits == null
              ? ''
              : ` · q${cell.sweep.n_qubits}/d${cell.sweep.n_circuit_layers ?? 'unavailable'}`
            const label = cell.label || cell.cell_id || `${cell.dataset || `Analysis cell ${index + 1}`}${scale}`
            return (
            <div className="analysis-cell" key={cell.cell_id || cell.id || `${cell.dataset || 'cell'}-${index}`}>
              <div className="workspace-header"><b>{label}</b><span className="badge">{cell.assessment_status || cell.verdict?.assessment_status || 'unavailable'}</span></div>
              <div className="evidence-identity"><Field label="claim ID" value={cell.claim_id} /><Field label="metric type" value={cell.metric_type} /></div>
              <Statistics paired={cell.paired_stats} equivalence={cell.equivalence} power={cell.power} />
              <Fairness mismatches={cell.fairness_mismatches} />
              <Analogue ladder={cell.analogue_ladder} limitations={cell.analogue_limitations} />
            </div>
          )})}
        </div>
      )}
    </section>
  )
}
