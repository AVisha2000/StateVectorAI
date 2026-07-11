// Pure curve-shaping helpers shared by Run detail, Verdicts, and the scaling
// view. Kept framework-free so they can be unit-tested with `node --test`.
// The backend returns each metric as an array of { step, value } points; these
// helpers reshape those into the row format recharts consumes.

// Merge several metric-keyed curves into rows keyed by step:
//   mergeCurve({ val_ppl: [{step:0,value:9}], train_loss: [{step:0,value:2}] })
//   -> [{ step: 0, val_ppl: 9, train_loss: 2 }]
// `metrics` optionally restricts/order the columns; otherwise every key is used.
export function mergeCurve(curve, metrics) {
  if (!curve || typeof curve !== 'object') return []
  const keys = metrics && metrics.length ? metrics.filter((m) => curve[m]) : Object.keys(curve)
  const byStep = new Map()
  for (const metric of keys) {
    const points = Array.isArray(curve[metric]) ? curve[metric] : []
    for (const point of points) {
      if (!point || point.step == null || point.value == null) continue
      const row = byStep.get(point.step) || { step: point.step }
      row[metric] = point.value
      byStep.set(point.step, row)
    }
  }
  return [...byStep.values()].sort((a, b) => a.step - b.step)
}

// Overlay a candidate curve and its matched-control (baseline) curve for a
// single metric into rows { step, candidate, baseline } for a two-line chart.
export function mergeComparison(candidateCurve, baselineCurve, metric) {
  const cand = Array.isArray(candidateCurve?.[metric]) ? candidateCurve[metric] : []
  const base = Array.isArray(baselineCurve?.[metric]) ? baselineCurve[metric] : []
  const byStep = new Map()
  for (const point of cand) {
    if (!point || point.step == null) continue
    byStep.set(point.step, { step: point.step, candidate: point.value ?? null, baseline: null })
  }
  for (const point of base) {
    if (!point || point.step == null) continue
    const row = byStep.get(point.step) || { step: point.step, candidate: null, baseline: null }
    row.baseline = point.value ?? null
    byStep.set(point.step, row)
  }
  return [...byStep.values()].sort((a, b) => a.step - b.step)
}

// Aggregate several per-seed curves for one metric into a min–max band with a
// mean line: [{ step, min, max, mean, band: [min, max] }]. `band` is the tuple
// recharts' Area wants for a min–max ribbon. Steps present in any seed are kept;
// missing seeds simply don't contribute to that step.
export function seedBand(seedCurves, metric) {
  const curves = Array.isArray(seedCurves) ? seedCurves : []
  const byStep = new Map()
  for (const curve of curves) {
    const points = Array.isArray(curve?.[metric]) ? curve[metric] : []
    for (const point of points) {
      if (!point || point.step == null || point.value == null) continue
      const bucket = byStep.get(point.step) || []
      bucket.push(point.value)
      byStep.set(point.step, bucket)
    }
  }
  return [...byStep.entries()]
    .map(([step, values]) => {
      const min = Math.min(...values)
      const max = Math.max(...values)
      const mean = values.reduce((a, b) => a + b, 0) / values.length
      return { step, min, max, mean, band: [min, max], n: values.length }
    })
    .sort((a, b) => a.step - b.step)
}

// Least-squares fit of log10(y) against x, for the barren-plateau scaling view
// (gradient variance vs qubit count decays ~exponentially). Display-only: it
// draws the trend/extrapolation line and reports R². It deliberately does NOT
// decide a plateau onset or a trainability verdict — those thresholds and any
// claim classification stay backend-owned (docs/UI_REDESIGN_PLAN §12).
export function logLinearFit(points) {
  const pts = (Array.isArray(points) ? points : [])
    .filter((p) => p && p.x != null && p.y != null && p.y > 0)
    .map((p) => ({ x: p.x, ly: Math.log10(p.y) }))
  if (pts.length < 2) return null
  const n = pts.length
  const sx = pts.reduce((a, p) => a + p.x, 0)
  const sy = pts.reduce((a, p) => a + p.ly, 0)
  const sxx = pts.reduce((a, p) => a + p.x * p.x, 0)
  const sxy = pts.reduce((a, p) => a + p.x * p.ly, 0)
  const denom = n * sxx - sx * sx
  if (denom === 0) return null
  const slope = (n * sxy - sx * sy) / denom
  const intercept = (sy - slope * sx) / n
  // R² on the log scale.
  const meanLy = sy / n
  let ssTot = 0
  let ssRes = 0
  for (const p of pts) {
    const pred = slope * p.x + intercept
    ssTot += (p.ly - meanLy) ** 2
    ssRes += (p.ly - pred) ** 2
  }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot
  return {
    slope, // decades of variance per qubit (negative = decaying)
    intercept,
    r2,
    predict: (x) => 10 ** (slope * x + intercept),
  }
}
