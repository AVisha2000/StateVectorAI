// Shape a /scaling-tests/{groupId} payload into chart rows + a summary. Points
// flagged rerun_required, or with no measured metric, are dropped from the chart
// (and reported as dropped so the omission is never silent).

function num(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function scalingChartRows(points) {
  const list = Array.isArray(points) ? points : []
  const kept = []
  let dropped = 0
  for (const p of list) {
    const valPpl = num(p?.val_ppl)
    const wall = num(p?.wall_seconds)
    if (p?.rerun_required || (valPpl == null && wall == null)) {
      dropped += 1
      continue
    }
    const nq = num(p?.n_qubits)
    const nd = num(p?.n_circuit_layers)
    kept.push({
      x: `q${nq ?? '?'}/d${nd ?? '?'}`,
      n_qubits: nq,
      n_circuit_layers: nd,
      scale: num(p?.scale),
      val_ppl: valPpl,
      wall_seconds: wall,
      n_params: num(p?.n_params),
    })
  }
  kept.sort((a, b) => (a.n_qubits ?? 0) - (b.n_qubits ?? 0) || (a.n_circuit_layers ?? 0) - (b.n_circuit_layers ?? 0))
  return { rows: kept, dropped }
}

// Every point in the group as a table row — including the ones dropped from the
// chart — so "never silently dropped" is honoured: each carries the job id for a
// backlink, whether it was omitted from the chart, and why. Mirrors the drop
// rule in scalingChartRows so the table and chart never disagree.
export function scalingPointRows(points) {
  const list = Array.isArray(points) ? points : []
  return list
    .map((p) => {
      const valPpl = num(p?.val_ppl)
      const wall = num(p?.wall_seconds)
      const rerun = !!p?.rerun_required
      const noMetric = valPpl == null && wall == null
      const omitted = rerun || noMetric
      return {
        jobId: p?.job?.id ?? null,
        runName: p?.job?.run_name ?? null,
        n_qubits: num(p?.n_qubits),
        n_circuit_layers: num(p?.n_circuit_layers),
        val_ppl: valPpl,
        wall_seconds: wall,
        n_params: num(p?.n_params),
        status: p?.status ?? null,
        omitted,
        omittedReason: rerun ? 'rerun required' : noMetric ? 'no measured metric' : null,
      }
    })
    .sort((a, b) => (a.n_qubits ?? 0) - (b.n_qubits ?? 0) || (a.n_circuit_layers ?? 0) - (b.n_circuit_layers ?? 0))
}

// A representative job id from the group — used to fetch the group-level
// scaling_fit diagnostic (which the backend computes from same-group persisted
// qubit counts, so any measured member returns the same fit). Prefer a done
// point, then any point that carries a job id.
export function representativeJobId(points) {
  const list = Array.isArray(points) ? points : []
  const done = list.find((p) => p?.status === 'done' && p?.job?.id != null)
  if (done) return done.job.id
  const any = list.find((p) => p?.job?.id != null)
  return any ? any.job.id : null
}

// Shape the scaling_fit DiagnosticDimension into display fields, or null.
export function scalingFitView(dimension) {
  if (!dimension || dimension.status !== 'measured' || !dimension.value) {
    return { available: false, reason: dimension?.reason || 'not computed for this group yet' }
  }
  const v = dimension.value
  return {
    available: true,
    slope: typeof v.log_var_slope === 'number' ? v.log_var_slope : null,
    decayPerQubit: typeof v.variance_decay_factor_per_qubit === 'number' ? v.variance_decay_factor_per_qubit : null,
    exponentialDecay: Boolean(v.exponential_decay_detected),
    source: dimension.source ?? null,
  }
}

export function scalingSummary(payload) {
  const best = payload?.best || null
  return {
    complete: num(payload?.complete_count) ?? 0,
    total: num(payload?.total_count) ?? 0,
    best,
    protocolWarnings: Array.isArray(payload?.protocol_warnings) ? payload.protocol_warnings : [],
  }
}
