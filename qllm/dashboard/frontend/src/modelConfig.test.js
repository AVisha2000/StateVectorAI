import assert from 'node:assert/strict'
import test from 'node:test'

import { changeArchitecture, changeQuantumBackend, ensureBlocks } from './modelConfig.js'

function transformerConfig() {
  return {
    model: {
      arch: 'transformer',
      n_blocks: 2,
      attn_type: 'quantum_proj',
      ffn_type: 'quantum',
      embed_type: 'quantum',
      head_type: 'interference',
      encoder_kind: 'none',
      quantum: { n_qubits: 4 },
      blocks: [
        { attn_type: 'quantum_proj', ffn_type: 'quantum' },
        { attn_type: 'classical', ffn_type: 'classical' },
      ],
    },
  }
}

for (const arch of ['qrnn', 'gru', 'contextual_qrnn', 'routed_contextual']) {
  test(`switching to ${arch} removes ignored transformer components`, () => {
    const changed = changeArchitecture(transformerConfig(), arch)
    assert.equal(changed.model.arch, arch)
    assert.equal(changed.model.blocks, null)
    assert.equal(changed.model.attn_type, 'classical')
    assert.equal(changed.model.ffn_type, 'classical')
    assert.equal(changed.model.embed_type, 'classical')
    assert.equal(changed.model.head_type, 'linear')
    assert.equal(changed.model.encoder_kind, 'none')
  })
}

test('switching back to transformer restores the configured block count', () => {
  const recurrent = changeArchitecture(transformerConfig(), 'qrnn')
  const changed = changeArchitecture(recurrent, 'transformer')
  assert.equal(changed.model.blocks.length, 2)
  assert.equal(changed.model.encoder_kind, 'none')
})

test('loading a recurrent spec never synthesizes transformer blocks', () => {
  const recurrent = changeArchitecture(transformerConfig(), 'contextual_qrnn')
  assert.equal(ensureBlocks(recurrent).model.blocks, null)
})

test('switching a quantum-free classical spec seeds server-provided quantum defaults', () => {
  const config = transformerConfig()
  config.model.quantum = null
  const quantumDefault = { n_qubits: 6, ansatz: 'reuploading' }
  const changed = changeArchitecture(config, 'contextual_qrnn', {
    quantumArchitectures: ['qrnn', 'contextual_qrnn', 'routed_contextual'],
    quantumDefault,
  })
  assert.deepEqual(changed.model.quantum, quantumDefault)
  assert.notEqual(changed.model.quantum, quantumDefault)
})

test('MPS selection applies an explicit fixed-bond approximate mode', () => {
  const changed = changeQuantumBackend({
    backend: 'pennylane',
    mps_max_truncation_error: 1e-6,
    mps_relative_truncation: true,
  }, 'tensorcircuit_mps')
  assert.equal(changed.backend, 'tensorcircuit_mps')
  assert.equal(changed.device, 'mps')
  assert.equal(changed.diff_method, 'backprop')
  assert.equal(changed.shots, null)
  assert.equal(changed.mps_max_bond_dimension, 64)
  assert.equal(changed.mps_max_truncation_error, null)
  assert.equal(changed.mps_relative_truncation, false)
})

test('leaving MPS clears approximation controls instead of ignoring them', () => {
  const changed = changeQuantumBackend({
    backend: 'tensorcircuit_mps',
    mps_max_bond_dimension: 8,
    mps_max_truncation_error: 1e-6,
    mps_relative_truncation: true,
  }, 'pennylane')
  assert.equal(changed.device, 'default.qubit')
  assert.equal(changed.mps_max_bond_dimension, null)
  assert.equal(changed.mps_max_truncation_error, null)
  assert.equal(changed.mps_relative_truncation, false)
})
