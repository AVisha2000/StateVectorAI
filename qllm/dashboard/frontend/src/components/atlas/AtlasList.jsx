import { Fragment } from 'react'
import { OUTCOME_LABELS } from '../../lib/atlasModel.js'

// Accessible table alternative to the graph (and the render path a future static
// public export reuses). Grouped by domain; every row — including
// "classical holds / no advantage" and "unexplored" — carries equal weight, with
// claim level and replication as separate columns.
export default function AtlasList({ domains, onSelect, selectedId }) {
  return (
    <div className="card scroll-x">
      <table className="data atlas-list">
        <thead>
          <tr>
            <th>Cell</th><th>Outcome</th><th>Claim level</th><th>Replication</th>
            <th>Verdict</th><th>Provenance</th>
          </tr>
        </thead>
        <tbody>
          {domains.map((d) => (
            <Fragment key={d.id}>
              <tr className="atlas-group">
                <td colSpan="6"><span className="microlabel">{d.label}</span></td>
              </tr>
              {d.cells.map((c) => (
                <tr
                  key={c.id}
                  className={`click ${selectedId === c.id ? 'sel' : ''}`}
                  onClick={() => onSelect?.(c.id)}
                >
                  <td>{c.label}</td>
                  <td><span className={`atlas-oc atlas-oc-${c.outcome_class}`}>{OUTCOME_LABELS[c.outcome_class]}</span></td>
                  <td>{c.claim_level || '—'}</td>
                  <td>{c.replication_status || '—'}</td>
                  <td>{c.verdict_claim_level ? <span className="tag good">{c.verdict_claim_level}</span> : <span className="hint">—</span>}</td>
                  <td><span className={`tag ${c.provenance === 'seed' ? 'plain' : 'good'}`}>{c.provenance === 'seed' ? 'seed' : 'verdict'}</span></td>
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}
