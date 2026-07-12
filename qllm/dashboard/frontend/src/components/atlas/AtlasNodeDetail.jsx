import { Link } from 'react-router-dom'
import { OUTCOME_LABELS } from '../../lib/atlasModel.js'

// Side panel for a selected cell. Claim level and replication status are shown
// as SEPARATE labeled rows and never combined; a matched verdict's canonical
// claim (a different vocabulary) is surfaced distinctly. Null / classical-holds
// cells read with the same weight as positive ones.
export default function AtlasNodeDetail({ cell }) {
  if (!cell) {
    return (
      <div className="card">
        <div className="hd"><h3>Select a cell</h3></div>
        <div className="bd">
          <p className="hint" style={{ margin: 0 }}>
            Pick any cell to see its head-to-head status, claim level, replication, and the latest verdict — or a
            "no advantage found" result, shown with equal prominence.
          </p>
        </div>
      </div>
    )
  }
  const seed = cell.provenance === 'seed'
  return (
    <div className="card">
      <div className="hd">
        <span className={`atlas-oc atlas-oc-${cell.outcome_class}`}>{OUTCOME_LABELS[cell.outcome_class]}</span>
        <span className="spacer" />
        <span className={`tag ${seed ? 'plain' : 'good'}`}>{seed ? 'seed · unverified' : 'derived verdict'}</span>
      </div>
      <div className="bd">
        <div style={{ fontWeight: 600, marginBottom: 8 }}>{cell.label}</div>
        <div className="atlas-kv">
          <div><span className="microlabel">Claim level (map)</span><div>{cell.claim_level || '—'}</div></div>
          <div><span className="microlabel">Replication</span><div>{cell.replication_status || '—'}</div></div>
          <div><span className="microlabel">Status</span><div>{cell.status || '—'}</div></div>
          <div><span className="microlabel">Pipeline stage</span><div>{cell.pipeline_stage || '—'}</div></div>
          <div><span className="microlabel">Quantum resource</span><div>{cell.quantum_resource || '—'}</div></div>
          <div><span className="microlabel">Advantage target</span><div>{cell.advantage_target || '—'}</div></div>
        </div>

        {cell.verdict_claim_level || cell.verdict_claim_status ? (
          <div className="notice" style={{ marginTop: 12 }}>
            <span className="microlabel">Latest verdict (ledger)</span>
            <div style={{ marginTop: 4 }}>
              claim level <b>{cell.verdict_claim_level || '—'}</b>
              {cell.verdict_claim_status ? <> · status <b>{cell.verdict_claim_status}</b></> : null}
              {' '}— canonical vocabulary, kept distinct from the map's claim ladder above.
            </div>
          </div>
        ) : null}

        <div className="row" style={{ marginTop: 12, gap: 8 }}>
          {cell.verdict_id != null ? (
            <Link className="btn sm" to={`/verdicts/${cell.verdict_id}`}>Open verdict →</Link>
          ) : null}
          {cell.outcome_class === 'unexplored' || cell.outcome_class === 'suggested' ? (
            <Link className="btn sm primary" to="/bench">Design a test →</Link>
          ) : null}
        </div>

        <p className="hint" style={{ marginTop: 10 }}>
          Area <span className="mono">{cell.area_id}</span>. Colors reflect the curated RESEARCH_MAP status; no composite
          advantage score is computed. Simulator results only.
        </p>
      </div>
    </div>
  )
}
