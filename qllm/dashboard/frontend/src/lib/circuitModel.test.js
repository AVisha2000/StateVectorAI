import assert from 'node:assert/strict'
import test from 'node:test'
import {
  ansatzCircuit, gateCount, paramCount, entanglingCount, circuitDepth,
  toBenchSpec, toQuantumOverrides, designerConstraints,
  ANSATZE, BACKENDS, READOUTS,
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
  assert.deepEqual([...READOUTS], ['z', 'zz']) // READOUT_TYPES — 'all' is not a registry readout
})

test('designerConstraints: ising is QRNN-only with pinned compatibility values', () => {
  const r = designerConstraints({ ansatz: 'ising', backend: 'tensorcircuit' })
  assert.equal(r.architecture, 'qrnn')
  assert.equal(r.backendLocked, 'pennylane')
  assert.equal(r.readoutLocked, 'z')
  assert.equal(r.needsBondDim, false) // qrnn pins pennylane, so no MPS bond dim
  const open = designerConstraints({ ansatz: 'reuploading', backend: 'pennylane' })
  assert.deepEqual(open, { architecture: null, backendLocked: null, readoutLocked: null, needsBondDim: false })
})

test('designerConstraints: tensorcircuit_mps requires a bond dimension', () => {
  const r = designerConstraints({ ansatz: 'hardware_efficient', backend: 'tensorcircuit_mps' })
  assert.equal(r.needsBondDim, true)
  assert.equal(designerConstraints({ ansatz: 'hardware_efficient', backend: 'tensorcircuit' }).needsBondDim, false)
})

test('toBenchSpec: ising emits architecture=qrnn with pennylane/z compat values', () => {
  const spec = toBenchSpec(ansatzCircuit('ising', 4, 2), { backend: 'tensorcircuit', readout: 'zz' })
  assert.equal(spec.architecture, 'qrnn')
  assert.equal(spec.backend, 'pennylane') // compatibility value, not the requested backend
  assert.equal(spec.readout, 'z')
  assert.equal(spec.mps_max_bond_dimension, null)
})

test('toBenchSpec: bond dimension rides only with tensorcircuit_mps', () => {
  const c = ansatzCircuit('reuploading', 4, 2)
  const mps = toBenchSpec(c, { backend: 'tensorcircuit_mps', readout: 'z', mpsMaxBondDimension: 32 })
  assert.equal(mps.mps_max_bond_dimension, 32)
  const exact = toBenchSpec(c, { backend: 'pennylane', readout: 'z', mpsMaxBondDimension: 32 })
  assert.equal(exact.mps_max_bond_dimension, null) // backend rejects it outside mps
})
