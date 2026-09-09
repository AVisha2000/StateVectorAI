import test from 'node:test'
import assert from 'node:assert/strict'
import { DEMO_MODELS, simulateStudio } from './studioSimulation.js'
test('every illustrative model returns bounded, finite, deterministic samples', () => {
  for (const [id, model] of Object.entries(DEMO_MODELS))
    for (const parameter of [model.min, model.initial, model.max]) {
      const result = simulateStudio(id, parameter)
      assert.deepEqual(result, simulateStudio(id, parameter))
      assert.equal(result.points.length, 121)
      assert.ok(result.points.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y)))
      assert.match(result.provenance, /synthetic inputs/)
    }
})
test('simple model boundary values agree with their displayed equations', () => {
  assert.equal(simulateStudio('physics', 45).points[0].y, 0)
  assert.equal(simulateStudio('chemistry', 0.5).points[0].y, 1)
  assert.equal(simulateStudio('biology', 0.5).points[0].y, 0.1)
  assert.equal(simulateStudio('machine-learning', 0.2).points[0].y, 0.5)
  const gate = simulateStudio('ai-safety', 0.5).points
  assert.equal(gate[59].y, 1)
  assert.equal(gate[60].y, 0)
})
test('invalid models and unbounded parameters fail closed', () => {
  for (const value of [NaN, Infinity, -1, 1000]) assert.throws(() => simulateStudio('physics', value))
  assert.throws(() => simulateStudio('unknown', 1))
})
