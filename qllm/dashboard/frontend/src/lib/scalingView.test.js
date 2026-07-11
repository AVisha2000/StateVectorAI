import assert from 'node:assert/strict'
import test from 'node:test'
import { scalingChartRows, scalingSummary, representativeJobId, scalingFitView } from './scalingView.js'

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
