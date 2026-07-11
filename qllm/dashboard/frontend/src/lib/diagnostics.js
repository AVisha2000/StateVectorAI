// Extract per-run quantum diagnostics for the Run-detail KPI row. Two sources,
// preferred in order:
//   1. the proposed GET /jobs/{id}/diagnostics endpoint (richer: adds SNR,
//      scaling fit) — not shipped yet, degrades to null;
//   2. summary.quantum_diagnostics, already persisted at train time and served
//      via GET /jobs/{id}/model-tests.
// Every diagnostic is labeled a *diagnostic / mechanism candidate*, never an
// advantage — the labels live in the presentation layer; this module only
// reshapes numbers and carries the backend `availability` reason for absents.

// The KPI dimensions the run-detail row shows, in order. `absent` metrics render
// as unavailable with the backend-provided reason rather than a fabricated value.
export const DIAGNOSTIC_KPIS = Object.freeze([
  { key: 'grad_var_mean', label: 'Grad variance', hint: 'barren-plateau watch', kind: 'sci' },
  { key: 'grad_snr', label: 'Grad SNR', hint: 'param-shift', kind: 'num' },
  { key: 'expressibility_kl', label: 'Expressibility KL', hint: 'vs Haar', kind: 'num' },
  { key: 'meyer_wallach_q', label: 'Entanglement', hint: 'Meyer–Wallach', kind: 'num' },
])

function num(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

// Normalize whatever source we have into { value, available, reason } per key.
// `diagnostics` is the (future) /diagnostics payload; `summary` is
// summary.quantum_diagnostics. Diagnostics endpoint wins when it carries a key.
export function diagnosticValues(diagnostics, summary) {
  const d = diagnostics && typeof diagnostics === 'object' ? diagnostics : {}
  const s = summary && typeof summary === 'object' ? summary : {}
  const availability = (s.availability && typeof s.availability === 'object') ? s.availability : {}

  // Map KPI keys to their source keys; grad_snr comes only from the richer
  // /diagnostics endpoint (parameter_shift_gradient_snr median), absent today.
  const resolve = (kpiKey) => {
    if (kpiKey === 'grad_snr') {
      const v = num(d.grad_snr) ?? num(d.median_snr)
      return { value: v, available: v != null, reason: v == null ? 'not computed for dashboard runs yet' : null }
    }
    const fromDiag = num(d[kpiKey])
    if (fromDiag != null) return { value: fromDiag, available: true, reason: null }
    const fromSummary = num(s[kpiKey])
    const avail = availability[kpiKey]
    if (fromSummary != null) return { value: fromSummary, available: true, reason: null }
    return {
      value: null,
      available: false,
      reason: (avail && (avail.reason || avail.semantics)) || 'unavailable for this run',
    }
  }

  const out = {}
  for (const kpi of DIAGNOSTIC_KPIS) out[kpi.key] = resolve(kpi.key)
  return out
}

export function hasAnyDiagnostic(values) {
  return Object.values(values || {}).some((v) => v && v.available)
}
