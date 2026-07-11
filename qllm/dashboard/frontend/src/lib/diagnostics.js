// Extract per-run quantum diagnostics for the Run-detail KPI row from the shipped
// GET /jobs/{id}/diagnostics payload. That endpoint is retrieval-only (never runs
// circuits) and returns one object per dimension:
//   { status: "measured"|"unavailable", value: number|object|null, source, reason,
//     provenance }
// gradient_variance and parameter_shift_gradient_snr are mapping-valued; a KPI
// picks one field out of the mapping. expressibility_kl and meyer_wallach_q are
// scalar-valued. Every diagnostic is a *diagnostic / mechanism candidate*, never
// an advantage — the endpoint's own interpretation_warnings say so, and the
// labels live in the presentation layer; this module only reshapes numbers.

// KPI dimensions the run-detail row shows, in order.
export const DIAGNOSTIC_KPIS = Object.freeze([
  { key: 'grad_var_mean', label: 'Grad variance', hint: 'barren-plateau watch', kind: 'sci' },
  { key: 'grad_snr', label: 'Grad SNR', hint: 'param-shift', kind: 'num' },
  { key: 'expressibility_kl', label: 'Expressibility KL', hint: 'vs Haar', kind: 'num' },
  { key: 'meyer_wallach_q', label: 'Entanglement', hint: 'Meyer–Wallach', kind: 'num' },
])

// Map each KPI to its backend diagnostics dimension and (for mapping-valued
// dimensions) the field to pick out of `value`.
const KPI_SOURCE = {
  grad_var_mean: { dim: 'gradient_variance', pick: 'grad_var_mean' },
  grad_snr: { dim: 'parameter_shift_gradient_snr', pick: 'median_snr' },
  expressibility_kl: { dim: 'expressibility_kl', pick: null },
  meyer_wallach_q: { dim: 'meyer_wallach_q', pick: null },
}

function num(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

// `diagnostics` is the endpoint payload's `.diagnostics` object; `summary` is the
// legacy summary.quantum_diagnostics fallback (used only before the endpoint is
// reachable). The endpoint wins when it carries a measured dimension.
export function diagnosticValues(diagnostics, summary) {
  const d = diagnostics && typeof diagnostics === 'object' ? diagnostics : {}
  const s = summary && typeof summary === 'object' ? summary : {}
  const availability = s.availability && typeof s.availability === 'object' ? s.availability : {}

  const resolve = (kpiKey) => {
    const { dim, pick } = KPI_SOURCE[kpiKey]
    const dimension = d[dim]
    if (dimension && dimension.status === 'measured') {
      const raw = pick ? dimension.value?.[pick] : dimension.value
      const v = num(raw)
      if (v != null) return { value: v, available: true, reason: null, source: dimension.source }
      return { value: null, available: false, reason: dimension.reason || 'measured but field missing' }
    }
    if (dimension && dimension.status === 'unavailable') {
      return { value: null, available: false, reason: dimension.reason || 'unavailable for this run' }
    }
    // Fallback: the legacy summary carries flat scalar keys, but no SNR.
    if (kpiKey !== 'grad_snr') {
      const sv = num(s[kpiKey])
      if (sv != null) return { value: sv, available: true, reason: null, source: 'summary.quantum_diagnostics' }
    }
    const avail = availability[kpiKey]
    return {
      value: null,
      available: false,
      reason: (avail && (avail.reason || avail.semantics)) ||
        (kpiKey === 'grad_snr' ? 'not computed for dashboard runs yet' : 'awaiting backend diagnostics'),
    }
  }

  const out = {}
  for (const kpi of DIAGNOSTIC_KPIS) out[kpi.key] = resolve(kpi.key)
  return out
}

export function hasAnyDiagnostic(values) {
  return Object.values(values || {}).some((v) => v && v.available)
}
