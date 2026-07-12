// Pure shaping of a backend comparison payload into the Verdicts surface's
// scorecard, fairness checks, and claim ladder. Integrity rules (encoded here so
// they are testable):
//   * No composite / averaged "advantage score" is ever produced — the scorecard
//     is per-dimension only, and a strong dimension cannot raise a claim level.
//   * Claim classification stays backend-owned: the ladder is passed through from
//     `evidence_ladder`, never derived here.
//   * Wall-clock is labeled simulator cost, never presented as a QPU cost.

// The dimensions we surface, in display order. `kind` drives labeling and how a
// winner is read; `lowerIsBetter` only applies to quality/cost dimensions.
export const SCORECARD_DIMENSIONS = Object.freeze([
  { key: 'val_ppl', label: 'Final val perplexity', kind: 'quality', lowerIsBetter: true },
  { key: 'val_loss', label: 'Final val loss', kind: 'quality', lowerIsBetter: true },
  { key: 'val_bpc', label: 'Final val bpc', kind: 'quality', lowerIsBetter: true },
  { key: 'wall_seconds', label: 'Wall-clock (simulator cost)', kind: 'cost', lowerIsBetter: true },
  { key: 'n_params', label: 'Parameters', kind: 'size', lowerIsBetter: false },
])

function pickNum(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

// One scorecard row per dimension present on either arm. `favors` is a plain
// per-dimension winner marker; it is NOT summed anywhere.
export function scorecardRows(comparison, dims = SCORECARD_DIMENSIONS) {
  const cand = comparison?.candidate?.final_run || {}
  const base = comparison?.baseline?.final_run || {}
  return dims
    .map((d) => {
      const q = pickNum(cand[d.key])
      const c = pickNum(base[d.key])
      if (q == null && c == null) return null
      let favors = null
      if (q != null && c != null && q !== c) {
        if (d.kind === 'size') favors = 'tie' // param-match is a fairness gate, not a win
        else favors = d.lowerIsBetter ? (q < c ? 'quantum' : 'classical') : (q > c ? 'quantum' : 'classical')
      } else if (q != null && c != null) {
        favors = 'tie'
      }
      const deltaPct = q != null && c != null && c !== 0 ? (q - c) / Math.abs(c) : null
      return { key: d.key, label: d.label, kind: d.kind, quantum: q, classical: c, favors, deltaPct }
    })
    .filter(Boolean)
}

// Backend-owned fairness gate → check rows. Never recompute thresholds here; a
// mismatch listed in `fairness_mismatches` marks the corresponding row failed.
const FAIRNESS_FIELDS = [
  { key: 'same_dataset', label: 'Same dataset' },
  { key: 'same_seed', label: 'Same seed' },
  { key: 'same_steps', label: 'Same step budget' },
  { key: 'same_eval_interval', label: 'Same eval schedule' },
  { key: 'same_device_target', label: 'Same device target' },
  { key: 'role_validation', label: 'Roles valid (candidate vs control)' },
]

export function fairnessChecks(comparison) {
  const fairness = comparison?.fairness || {}
  const rows = FAIRNESS_FIELDS
    .filter((f) => fairness[f.key] != null)
    .map((f) => ({ key: f.key, label: f.label, ok: Boolean(fairness[f.key]) }))
  const ratio = pickNum(fairness.parameter_delta_ratio)
  if (ratio != null) {
    rows.push({
      key: 'parameter_delta_ratio',
      label: `Parameter match ${ratio.toFixed(2)}×`,
      ok: null, // pass/fail of the match is a backend gate; show the ratio neutrally
      detail: `${ratio.toFixed(2)}× candidate/control parameters`,
    })
  }
  return rows
}

export function passedFairnessCount(rows) {
  const gated = rows.filter((r) => r.ok === true || r.ok === false)
  return { passed: gated.filter((r) => r.ok).length, total: gated.length }
}

// Pass the backend claim ladder straight through with null-safety. Returns the
// step rungs plus the headline label / claim level / reason as-is.
export function ladderView(source) {
  const ladder = source?.evidence_ladder || source || {}
  const steps = Array.isArray(ladder.steps) ? ladder.steps : []
  return {
    label: ladder.label ?? source?.verdict?.label ?? null,
    claimLevel: ladder.claim_level ?? source?.claim?.claim_level ?? null,
    reason: ladder.reason ?? source?.verdict?.reason ?? null,
    metCount: ladder.met_count ?? steps.filter((s) => s?.ok).length,
    totalCount: ladder.total_count ?? steps.length,
    assessmentStatus: ladder.assessment_status ?? source?.assessment_status ?? null,
    steps,
  }
}

// Auto caveats come from backend interpretation_warnings verbatim.
export function caveats(comparison) {
  const warnings = comparison?.interpretation_warnings
  return Array.isArray(warnings) ? warnings : []
}

// ---- Persistent verdict-store snapshots (/verdicts, /verdicts/{id}) ---------
// The store keeps claim_level / claim_status / replication_status as canonical,
// ledger-bound fields, DISTINCT from the derived assessment_level/assessment_status.
// We surface both but never conflate them, and never synthesize a composite score.

export function snapshotClaim(snapshot) {
  const s = snapshot || {}
  return {
    claimId: s.claim_id ?? null,
    claimLevel: s.claim_level ?? null, // canonical (ledger)
    claimStatus: s.claim_status ?? null, // canonical (ledger)
    replicationStatus: s.replication_status ?? null, // canonical, distinct from claim
    assessmentLevel: s.assessment_level ?? null, // derived, separate
    assessmentStatus: s.assessment_status ?? null, // derived, separate
    revision: s.revision ?? null,
    sourceKind: s.source_kind ?? null,
    sourceId: s.source_id ?? null,
    createdTs: s.created_ts ?? null,
    verdictKey: s.verdict_key ?? null,
  }
}

// Named scorecard dimensions as labeled per-dimension deltas — no aggregate.
export function snapshotScorecardRows(snapshot) {
  const dims = snapshot?.scorecard?.dimensions
  if (!dims || typeof dims !== 'object') return []
  const deltas = dims.deltas && typeof dims.deltas === 'object' ? dims.deltas : {}
  return Object.entries(deltas)
    .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
    .map(([key, delta]) => ({ key, label: key, delta }))
}

export function snapshotMetricType(snapshot) {
  return snapshot?.scorecard?.dimensions?.metric_type ?? null
}

// Turn a loosely-typed fairness/controls object into boolean check rows.
export function booleanChecks(obj) {
  if (!obj || typeof obj !== 'object') return []
  return Object.entries(obj)
    .filter(([, v]) => typeof v === 'boolean')
    .map(([key, ok]) => ({ key, label: key.replace(/_/g, ' '), ok }))
}

export function snapshotCaveats(snapshot) {
  const c = snapshot?.caveats
  return Array.isArray(c) ? c : []
}

// The claim ledger is append-only and content-addressed: each revision is a new
// snapshot, never an in-place edit. This turns the raw history list into a
// newest-first timeline that makes supersession and correction *visible* — it
// flags which fields changed from the immediately-older revision so a reader can
// see how the claim evolved without any history being rewritten. It never
// invents a verdict: claim_level/status/replication are rendered verbatim.
export function revisionHistory(history, currentRevision = null) {
  const rows = (Array.isArray(history) ? history : [])
    .filter((s) => s && typeof s === 'object')
    .map((s) => ({
      revision: s.revision ?? null,
      contentHash: s.content_hash ?? null,
      claimLevel: s.claim_level ?? null,
      claimStatus: s.claim_status ?? null,
      replicationStatus: s.replication_status ?? null,
      createdTs: s.created_ts ?? null,
    }))
  // Newest first. Fall back to created_ts when revision numbers are absent.
  rows.sort((a, b) => {
    if (a.revision != null && b.revision != null) return b.revision - a.revision
    return String(b.createdTs ?? '').localeCompare(String(a.createdTs ?? ''))
  })
  const currentRev = currentRevision ?? (rows.length ? rows[0].revision : null)
  return rows.map((row, i) => {
    const prev = rows[i + 1] // the immediately-older revision
    const changed = prev
      ? {
        level: row.claimLevel !== prev.claimLevel,
        status: row.claimStatus !== prev.claimStatus,
        replication: row.replicationStatus !== prev.replicationStatus,
      }
      : { level: false, status: false, replication: false }
    return {
      ...row,
      isCurrent: currentRev != null && row.revision === currentRev,
      isOldest: i === rows.length - 1,
      changed,
      changedAny: changed.level || changed.status || changed.replication,
    }
  })
}
