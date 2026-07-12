// Pure circuit model for the Designer. The registry runs parameterized ansatz
// FAMILIES (registry.py CIRCUIT_ANSATZ_TYPES + QRNN_ONLY_ANSATZ_TYPES), not
// free-form gate lists — so this derives the gate layout from (ansatz, n_qubits,
// depth) for an honest visualization that actually round-trips to a runnable
// Bench experiment. Framework-free for node --test.

export const BACKENDS = Object.freeze(['pennylane', 'tensorcircuit', 'tensorcircuit_mps'])
export const ANSATZE = Object.freeze(['hardware_efficient', 'reuploading', 'ising'])
export const READOUTS = Object.freeze(['zz', 'z', 'all'])

// Which gate types carry trainable rotation parameters.
const PARAM_GATES = new Set(['RX', 'RY', 'RZ'])
const TWO_QUBIT = new Set(['CNOT', 'CZ', 'ZZ'])

// Build the gate layout for one ansatz family. `col` is the time step.
export function ansatzCircuit(ansatz, nQubits, depth) {
  const n = Math.max(1, Math.floor(nQubits) || 1)
  const d = Math.max(1, Math.floor(depth) || 1)
  const gates = []
  const push = (g) => gates.push({ ...g, kind: TWO_QUBIT.has(g.type) ? 'two' : 'single', id: `${g.type}-${g.qubit}-${g.col}-${gates.length}` })
  for (let layer = 0; layer < d; layer += 1) {
    if (ansatz === 'hardware_efficient') {
      for (let q = 0; q < n; q += 1) push({ type: 'RY', qubit: q, col: layer * 2 })
      for (let q = 0; q < n - 1; q += 1) push({ type: 'CNOT', control: q, qubit: q + 1, col: layer * 2 + 1 })
    } else if (ansatz === 'reuploading') {
      for (let q = 0; q < n; q += 1) { push({ type: 'RX', qubit: q, col: layer * 3 }); push({ type: 'RY', qubit: q, col: layer * 3 + 1 }) }
      for (let q = 0; q < n - 1; q += 1) push({ type: 'CZ', control: q, qubit: q + 1, col: layer * 3 + 2 })
    } else { // ising (QRNN-only family)
      for (let q = 0; q < n - 1; q += 1) push({ type: 'ZZ', control: q, qubit: q + 1, col: layer * 2 })
      for (let q = 0; q < n; q += 1) push({ type: 'RX', qubit: q, col: layer * 2 + 1 })
    }
  }
  return { n_qubits: n, depth: d, ansatz, gates }
}

export function gateCount(circuit) {
  return (circuit?.gates || []).length
}

export function paramCount(circuit) {
  return (circuit?.gates || []).filter((g) => PARAM_GATES.has(g.type)).length
}

export function entanglingCount(circuit) {
  return (circuit?.gates || []).filter((g) => g.kind === 'two').length
}

export function columns(circuit) {
  const cols = new Set((circuit?.gates || []).map((g) => g.col))
  return [...cols].sort((a, b) => a - b)
}

export function circuitDepth(circuit) {
  return columns(circuit).length
}

// The spec handed to the proposed /designer/circuit round-trip and, on "Send to
// Bench", to a quantum preset's overrides. Only the registry-meaningful knobs.
export function toBenchSpec(circuit, { backend, readout } = {}) {
  return {
    ansatz: circuit?.ansatz ?? null,
    n_qubits: circuit?.n_qubits ?? null,
    n_circuit_layers: circuit?.depth ?? null,
    backend: backend ?? 'pennylane',
    readout: readout ?? 'zz',
    trainable_params: paramCount(circuit),
    entangling_gates: entanglingCount(circuit),
  }
}

// Bench consumes quantum_overrides keyed like preset quantum controls.
export function toQuantumOverrides(circuit) {
  return { n_qubits: circuit?.n_qubits ?? null, n_circuit_layers: circuit?.depth ?? null }
}
