import { OUTCOME_ORDER, OUTCOME_LABELS } from '../../lib/atlasModel.js'

// Spells out the four orthogonal encoding channels so nothing is ambiguous, and
// gives the null / "classical holds" outcome its own equal-weight legend row.
export default function AtlasLegend() {
  return (
    <div className="atlas-legend">
      <div className="atlas-legend-row">
        <span className="microlabel">Outcome (color)</span>
        {OUTCOME_ORDER.map((o) => (
          <span key={o} className={`atlas-oc atlas-oc-${o}`}>{OUTCOME_LABELS[o]}</span>
        ))}
      </div>
      <div className="atlas-legend-row">
        <span className="microlabel">Claim level → border width</span>
        <span className="hint">thin = untested · thick = formal (RESEARCH_MAP ladder)</span>
        <span className="microlabel" style={{ marginLeft: 12 }}>Replication → border style</span>
        <span className="hint">dashed = none · solid = replicated</span>
        <span className="microlabel" style={{ marginLeft: 12 }}>Cell type → shape</span>
        <span className="hint">▭ head-to-head · ⬡ quantum-only · ◇ suggested · ○ unexplored</span>
      </div>
    </div>
  )
}
