import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildJobPayloads,
  buildSweepPayload,
  estimateRuns,
  parseIntList,
  parsePositiveIntList,
  rigorLevel,
  requiresGpuGate,
} from './benchConfig.js'

const base = {
  presetId: 'quantum-ffn-4q',
  datasetName: 'contextual',
  runName: 'two-stream-q',
  steps: 2000,
  evalEvery: 100,
  batchSize: 16,
  seqLen: 64,
  deviceTarget: 'cpu',
}

test('single-seed quick probe keeps the base run name and no comparison', () => {
  const payloads = buildJobPayloads({ ...base, seeds: [42], queueComparison: false })
  assert.equal(payloads.length, 1)
  assert.equal(payloads[0].run_name, 'two-stream-q')
  assert.equal(payloads[0].queue_classical_comparison, false)
  assert.equal(payloads[0].preset_id, 'quantum-ffn-4q')
  assert.equal(payloads[0].device_target, 'cpu')
  assert.ok(!('quantum_overrides' in payloads[0]))
})

test('multi-seed standard pair suffixes run names and queues comparison', () => {
  const payloads = buildJobPayloads({ ...base, seeds: [11, 23], queueComparison: true })
  assert.deepEqual(payloads.map((p) => p.run_name), ['two-stream-q-s11', 'two-stream-q-s23'])
  assert.deepEqual(payloads.map((p) => p.seed), [11, 23])
  assert.ok(payloads.every((p) => p.queue_classical_comparison === true))
})

test('quantum overrides attach only when non-empty', () => {
  const withOv = buildJobPayloads({ ...base, seeds: [1], quantumOverrides: { n_qubits: 6 } })
  assert.deepEqual(withOv[0].quantum_overrides, { n_qubits: 6 })
  const noOv = buildJobPayloads({ ...base, seeds: [1], quantumOverrides: {} })
  assert.ok(!('quantum_overrides' in noOv[0]))
})

test('sweep payload carries the qubit/depth grid and a single base seed', () => {
  const payload = buildSweepPayload({ ...base, seeds: [11, 23], qubits: [4, 6, 8], depths: [2] })
  assert.deepEqual(payload.qubits, [4, 6, 8])
  assert.deepEqual(payload.depths, [2])
  assert.equal(payload.seed, 11)
  assert.equal(payload.run_name, 'two-stream-q')
})

test('estimateRuns doubles for controls and multiplies the grid for full studies', () => {
  assert.deepEqual(estimateRuns({ rigor: 'quick', seeds: [42], queueComparison: false }), { candidate: 1, control: 0, total: 1 })
  assert.deepEqual(estimateRuns({ rigor: 'standard', seeds: [1, 2, 3, 4, 5], queueComparison: true }), { candidate: 5, control: 5, total: 10 })
  assert.deepEqual(estimateRuns({ rigor: 'full', qubits: [4, 6, 8], depths: [2, 3], queueComparison: true }), { candidate: 6, control: 6, total: 12 })
})

test('list parsers handle commas, spaces, and reject junk', () => {
  assert.deepEqual(parseIntList('11, 23  42'), [11, 23, 42])
  assert.deepEqual(parseIntList('4,,x,6'), [4, 6])
  assert.deepEqual(parsePositiveIntList('0, 4, -2, 8'), [4, 8])
})

test('rigor lookup falls back and gpu gate flags non-cpu targets', () => {
  assert.equal(rigorLevel('nope').key, 'quick')
  assert.equal(rigorLevel('full').sweep, true)
  assert.equal(requiresGpuGate('cpu'), false)
  assert.equal(requiresGpuGate('auto'), true)
  assert.equal(requiresGpuGate('gpu'), true)
})
