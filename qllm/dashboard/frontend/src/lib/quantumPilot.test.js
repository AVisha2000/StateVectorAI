import test from 'node:test'
import assert from 'node:assert/strict'
import { INITIAL_BLOCH, QUANTUM_TARGETS, rotateBloch, simulatePulses, targetOverlap, initialQuantumVisit, captureQuantumAttempt, quantumNotebook } from './quantumPilot.js'

const close = (actual, expected, tolerance = 1e-12) => actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < tolerance, `${actual} != ${expected}`))
test('Bloch controls use the standard Pauli rotation signs and reach the three targets', () => {
  close(rotateBloch(INITIAL_BLOCH, 'y', Math.PI / 2), [1, 0, 0])
  close(rotateBloch(INITIAL_BLOCH, 'x', Math.PI), [0, 0, -1])
  close(rotateBloch(INITIAL_BLOCH, 'x', -Math.PI / 2), [0, 1, 0])
  close(rotateBloch(INITIAL_BLOCH, 'z', Math.PI / 2), INITIAL_BLOCH)
  assert.equal(targetOverlap(INITIAL_BLOCH, QUANTUM_TARGETS[1].vector), 0)
  assert.equal(targetOverlap(INITIAL_BLOCH, QUANTUM_TARGETS[0].vector), 0.5)
})
test('rotations preserve pure and mixed-state norm and have inverses', () => {
  for (const input of [[0.2, -0.3, 0.4], [...INITIAL_BLOCH], [0, 0, 0]]) {
    let state = input
    for (let i = 0; i < 100; i++) {
      const axis = ['x', 'y', 'z'][i % 3], angle = Math.sin(i) * Math.PI
      const next = rotateBloch(state, axis, angle)
      close(rotateBloch(next, axis, -angle), state)
      assert.ok(Math.abs(Math.hypot(...next) - Math.hypot(...input)) < 1e-12)
      state = next
    }
  }
})
test('pulse order matters and partial replay cannot change the recorded sequence', () => {
  const pulses = [{ axis: 'y', angle: Math.PI / 2 }, { axis: 'z', angle: Math.PI / 2 }]
  const before = structuredClone(pulses)
  close(simulatePulses(pulses).state, [0, 1, 0])
  close(simulatePulses([...pulses].reverse()).state, [1, 0, 0])
  close(simulatePulses(pulses, 0).state, INITIAL_BLOCH)
  close(simulatePulses(pulses, 1).state, [1, 0, 0])
  close(simulatePulses(pulses, 0.5).state, [Math.SQRT1_2, 0, Math.SQRT1_2])
  assert.deepEqual(pulses, before)
  close(simulatePulses(pulses.slice(0, -1)).state, [1, 0, 0])
})
test('invalid controls and states fail explicitly', () => {
  for (const axis of ['?', '__proto__', null]) assert.throws(() => rotateBloch(INITIAL_BLOCH, axis, 1))
  for (const angle of [NaN, Infinity, 7]) assert.throws(() => rotateBloch(INITIAL_BLOCH, 'x', angle))
  assert.throws(() => rotateBloch([0, 0, 2], 'x', 1))
  assert.throws(() => simulatePulses([{ axis: 'x', angle: NaN }], 0))
  assert.throws(() => simulatePulses([], -1))
  assert.throws(() => simulatePulses(Array.from({ length: 129 }, () => ({ axis: 'x', angle: 1 }))))
  assert.throws(() => targetOverlap(INITIAL_BLOCH, [0, 0, 2]))
})
test('attempts capture their target, note and controls independently of later edits', () => {
  const visit = initialQuantumVisit()
  visit.pulses.push({ axis: 'y', angle: Math.PI / 2 })
  visit.note = 'Try reversing pulse order.'
  const attempt = captureQuantumAttempt(visit, '2026-09-05T12:00:00Z')
  visit.attempts.push(attempt)
  const notebook = quantumNotebook(visit)
  visit.pulses[0].angle = 0
  visit.targetId = 'one'
  visit.note = 'Different question'
  assert.equal(attempt.overlap, 1)
  assert.equal(attempt.targetId, 'plus')
  assert.equal(attempt.note, 'Try reversing pulse order.')
  assert.equal(attempt.pulses[0].angle, Math.PI / 2)
  assert.equal(notebook.current.pulses[0].angle, Math.PI / 2)
  assert.match(notebook.provenance, /No shots, noise, hardware or live AI/)
})
test('empty attempts and full notebooks cannot create misleading saved results', () => {
  assert.throws(() => captureQuantumAttempt(initialQuantumVisit(), '2026-09-05T12:00:00Z'))
  const visit = { ...initialQuantumVisit(), pulses: [{ axis: 'y', angle: 1 }], attempts: Array(12).fill({}) }
  assert.throws(() => captureQuantumAttempt(visit, '2026-09-05T12:00:00Z'))
})
