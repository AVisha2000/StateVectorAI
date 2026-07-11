import assert from 'node:assert/strict'
import test from 'node:test'
import { scalingChartRows, scalingSummary } from './scalingView.js'

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
