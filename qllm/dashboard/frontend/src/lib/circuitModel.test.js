import assert from 'node:assert/strict'
import test from 'node:test'
import {
  ansatzCircuit, gateCount, paramCount, entanglingCount, circuitDepth,
  toBenchSpec, toQuantumOverrides, ANSATZE, BACKENDS,
} from './circuitModel.js'

test('hardware_efficient: RY per qubit + CNOT ladder per layer', () => {
  const c = ansatzCircuit('hardware_efficient', 4, 2)
  assert.equal(c.n_qubits, 4)
  assert.equal(paramCount(c), 8) // 4 RY × 2 layers
  assert.equal(entanglingCount(c), 6) // 3 CNOTs × 2 layers
  assert.ok(c.gates.every((g) => 'kind' in g && 'id' in g))
})

test('reuploading and ising produce distinct, non-empty layouts', () => {
  const r = ansatzCircuit('reuploading', 3, 1)
  assert.equal(paramCount(r), 6) // RX+RY per qubit
  assert.equal(entanglingCount(r), 2) // CZ ladder
  const i = ansatzCircuit('ising', 3, 1)
  assert.ok(entanglingCount(i) === 2 && paramCount(i) === 3) // ZZ ladder + RX
})

test('depth counts distinct columns and clamps invalid sizes', () => {
  assert.equal(circuitDepth(ansatzCircuit('hardware_efficient', 2, 3)), 6) // 2 cols/layer × 3
  const clamped = ansatzCircuit('hardware_efficient', 0, 0)
  assert.equal(clamped.n_qubits, 1)
  assert.equal(clamped.depth, 1)
  assert.ok(gateCount(clamped) >= 1)
})

test('toBenchSpec / toQuantumOverrides expose only registry-meaningful knobs', () => {
  const c = ansatzCircuit('reuploading', 6, 2)
  const spec = toBenchSpec(c, { backend: 'tensorcircuit', readout: 'zz' })
  assert.deepEqual(
    { ansatz: spec.ansatz, n_qubits: spec.n_qubits, n_circuit_layers: spec.n_circuit_layers, backend: spec.backend },
    { ansatz: 'reuploading', n_qubits: 6, n_circuit_layers: 2, backend: 'tensorcircuit' },
  )
  assert.deepEqual(toQuantumOverrides(c), { n_qubits: 6, n_circuit_layers: 2 })
  assert.ok(!('advantage' in spec) && !('score' in spec)) // no advantage claim in a spec
})

test('registry option lists match registry.py', () => {
  assert.deepEqual([...ANSATZE], ['hardware_efficient', 'reuploading', 'ising'])
  assert.deepEqual([...BACKENDS], ['pennylane', 'tensorcircuit', 'tensorcircuit_mps'])
})
