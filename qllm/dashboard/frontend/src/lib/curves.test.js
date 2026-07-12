import assert from 'node:assert/strict'
import test from 'node:test'
import { mergeCurve, mergeComparison, seedBand, logLinearFit } from './curves.js'

test('mergeCurve keys rows by step and merges metrics', () => {
  const rows = mergeCurve({
    val_ppl: [{ step: 0, value: 9 }, { step: 10, value: 5 }],
    train_loss: [{ step: 0, value: 2 }],
  })
  assert.deepEqual(rows, [
    { step: 0, val_ppl: 9, train_loss: 2 },
    { step: 10, val_ppl: 5 },
  ])
})

test('mergeCurve tolerates missing curve and restricts to requested metrics', () => {
  assert.deepEqual(mergeCurve(null), [])
  const rows = mergeCurve({ a: [{ step: 1, value: 1 }], b: [{ step: 1, value: 2 }] }, ['a'])
  assert.deepEqual(rows, [{ step: 1, a: 1 }])
})

test('mergeCurve skips null values but keeps 0', () => {
  const rows = mergeCurve({ m: [{ step: 0, value: 0 }, { step: 1, value: null }] })
  assert.deepEqual(rows, [{ step: 0, m: 0 }])
})

test('mergeComparison overlays candidate and baseline, filling gaps with null', () => {
  const rows = mergeComparison(
    { val_ppl: [{ step: 0, value: 9 }, { step: 10, value: 5 }] },
    { val_ppl: [{ step: 0, value: 8 }] },
    'val_ppl',
  )
  assert.deepEqual(rows, [
    { step: 0, candidate: 9, baseline: 8 },
    { step: 10, candidate: 5, baseline: null },
  ])
})

test('seedBand computes min/max/mean band per step', () => {
  const band = seedBand(
    [
      { val_ppl: [{ step: 0, value: 4 }, { step: 1, value: 3 }] },
      { val_ppl: [{ step: 0, value: 6 }, { step: 1, value: 5 }] },
    ],
    'val_ppl',
  )
  assert.deepEqual(band, [
    { step: 0, min: 4, max: 6, mean: 5, band: [4, 6], n: 2 },
    { step: 1, min: 3, max: 5, mean: 4, band: [3, 5], n: 2 },
  ])
})

test('logLinearFit recovers a clean exponential decay and R²', () => {
  // y = 10 ** (-0.5 x + 1): slope -0.5, perfect fit.
  const points = [1, 2, 3, 4].map((x) => ({ x, y: 10 ** (-0.5 * x + 1) }))
  const fit = logLinearFit(points)
  assert.ok(Math.abs(fit.slope - -0.5) < 1e-9)
  assert.ok(Math.abs(fit.r2 - 1) < 1e-9)
  assert.ok(Math.abs(fit.predict(5) - 10 ** (-0.5 * 5 + 1)) < 1e-6)
})

test('logLinearFit returns null when it cannot fit', () => {
  assert.equal(logLinearFit([{ x: 1, y: 1 }]), null)
  assert.equal(logLinearFit([{ x: 1, y: -1 }, { x: 2, y: 0 }]), null)
})
