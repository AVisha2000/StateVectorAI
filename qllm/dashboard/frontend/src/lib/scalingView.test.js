import assert from 'node:assert/strict'
import test from 'node:test'
import { scalingChartRows, scalingSummary, representativeJobId, scalingFitView, scalingPointRows } from './scalingView.js'

test('scalingChartRows drops rerun-required / empty points and sorts by scale', () => {
  const { rows, dropped } = scalingChartRows([
    { n_qubits: 6, n_circuit_layers: 2, val_ppl: 5.2, wall_seconds: 30 },
    { n_qubits: 4, n_circuit_layers: 2, val_ppl: 5.8, wall_seconds: 12 },
    { n_qubits: 8, n_circuit_layers: 2, rerun_required: true, val_ppl: 4.9 },
    { n_qubits: 10, n_circuit_layers: 2, val_ppl: null, wall_seconds: null },
  ])
  assert.equal(dropped, 2)
  assert.deepEqual(rows.map((r) => r.x), ['q4/d2', 'q6/d2'])
  assert.equal(rows[0].val_ppl, 5.8)
})

test('scalingSummary reads counts, best, and protocol warnings safely', () => {
  const s = scalingSummary({ complete_count: 3, total_count: 6, best: { val_ppl: 4.9 }, protocol_warnings: ['x'] })
  assert.equal(s.complete, 3)
  assert.equal(s.total, 6)
  assert.equal(s.best.val_ppl, 4.9)
  assert.deepEqual(s.protocolWarnings, ['x'])
  const empty = scalingSummary(null)
  assert.equal(empty.complete, 0)
  assert.equal(empty.best, null)
})

test('representativeJobId prefers a done point, then any with a job id', () => {
  assert.equal(representativeJobId([{ status: 'running', job: { id: 1 } }, { status: 'done', job: { id: 2 } }]), 2)
  assert.equal(representativeJobId([{ status: 'queued', job: { id: 7 } }]), 7)
  assert.equal(representativeJobId([{ status: 'done' }]), null)
  assert.equal(representativeJobId(null), null)
})

test('scalingFitView shapes a measured fit and marks unavailable ones', () => {
  const view = scalingFitView({
    status: 'measured', source: 'scaling',
    value: { log_var_slope: -0.34, variance_decay_factor_per_qubit: 0.71, exponential_decay_detected: true },
  })
  assert.equal(view.available, true)
  assert.equal(view.slope, -0.34)
  assert.equal(view.decayPerQubit, 0.71)
  assert.equal(view.exponentialDecay, true)
  const missing = scalingFitView({ status: 'unavailable', reason: 'need two qubit counts' })
  assert.equal(missing.available, false)
  assert.match(missing.reason, /two qubit counts/)
})

test('scalingPointRows keeps every point, flagging the omitted ones with a reason', () => {
  const rows = scalingPointRows([
    { job: { id: 11, run_name: 'q6' }, status: 'done', n_qubits: 6, n_circuit_layers: 2, val_ppl: 5.2, wall_seconds: 30, n_params: 2000 },
    { job: { id: 7, run_name: 'q4' }, status: 'done', n_qubits: 4, n_circuit_layers: 2, val_ppl: 5.8, wall_seconds: 12, n_params: 1000 },
    { job: { id: 12, run_name: 'q8' }, status: 'queued', n_qubits: 8, n_circuit_layers: 2, rerun_required: true },
    { job: { id: 13, run_name: 'q10' }, status: 'error', n_qubits: 10, n_circuit_layers: 2, val_ppl: null, wall_seconds: null },
  ])
  // nothing is dropped from the table — all four survive, sorted by qubits
  assert.deepEqual(rows.map((r) => r.n_qubits), [4, 6, 8, 10])
  assert.equal(rows.length, 4)
  // the measured points are not omitted
  assert.equal(rows[0].omitted, false)
  assert.equal(rows[0].jobId, 7)
  // rerun-required and no-metric points are flagged, with a reason
  const q8 = rows.find((r) => r.n_qubits === 8)
  assert.equal(q8.omitted, true)
  assert.equal(q8.omittedReason, 'rerun required')
  const q10 = rows.find((r) => r.n_qubits === 10)
  assert.equal(q10.omitted, true)
  assert.equal(q10.omittedReason, 'no measured metric')
})

test('scalingPointRows tolerates missing points / job ids', () => {
  assert.deepEqual(scalingPointRows(null), [])
  const rows = scalingPointRows([{ n_qubits: 4, n_circuit_layers: 1, val_ppl: 3 }])
  assert.equal(rows[0].jobId, null)
  assert.equal(rows[0].omitted, false)
})
