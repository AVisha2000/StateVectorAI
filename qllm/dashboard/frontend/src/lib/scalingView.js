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

export function scalingSummary(payload) {
  const best = payload?.best || null
  return {
    complete: num(payload?.complete_count) ?? 0,
    total: num(payload?.total_count) ?? 0,
    best,
    protocolWarnings: Array.isArray(payload?.protocol_warnings) ? payload.protocol_warnings : [],
  }
}
