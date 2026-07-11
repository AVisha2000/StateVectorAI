import assert from 'node:assert/strict'
import test from 'node:test'
import {
  scorecardRows,
  fairnessChecks,
  passedFairnessCount,
  ladderView,
  caveats,
} from './verdictView.js'
import { diagnosticValues, hasAnyDiagnostic } from './diagnostics.js'

const comparison = {
  candidate: { final_run: { val_ppl: 3.39, val_loss: 1.2, wall_seconds: 852, n_params: 18100 } },
  baseline: { final_run: { val_ppl: 3.55, val_loss: 1.3, wall_seconds: 228, n_params: 18400 } },
  fairness: {
    same_dataset: true, same_seed: true, same_steps: true, same_eval_interval: true,
    same_device_target: false, role_validation: true, parameter_delta_ratio: 0.984,
  },
  evidence_ladder: {
    label: 'paired empirical', claim_level: 'empirical', reason: 'candidate leads its matched control',
    met_count: 4, total_count: 8,
    steps: [
      { key: 'matched_baseline', label: 'Matched baseline', ok: true, detail: '', caution: '' },
      { key: 'multi_seed', label: 'Multiple seeds', ok: false, detail: 'single seed', caution: '' },
    ],
  },
  interpretation_warnings: [
    { code: 'single_seed', severity: 'warning', title: 'One pair', message: 'Single seed.' },
  ],
}

test('scorecard is per-dimension with a winner marker and no composite score', () => {
  const rows = scorecardRows(comparison)
  const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
  assert.equal(byKey.val_ppl.favors, 'quantum') // lower ppl wins
  assert.equal(byKey.wall_seconds.favors, 'classical') // simulator cost favors classical
  assert.equal(byKey.wall_seconds.label, 'Wall-clock (simulator cost)')
  assert.equal(byKey.n_params.favors, 'tie') // param match is a gate, never a win
  // The module exposes rows only — there is no aggregate/score field anywhere.
  assert.ok(!('score' in byKey.val_ppl))
})

test('scorecard drops dimensions absent on both arms', () => {
  const rows = scorecardRows({ candidate: { final_run: { val_ppl: 3 } }, baseline: { final_run: { val_ppl: 4 } } })
  assert.deepEqual(rows.map((r) => r.key), ['val_ppl'])
})

test('fairness checks pass through backend booleans and show ratio neutrally', () => {
  const rows = fairnessChecks(comparison)
  const device = rows.find((r) => r.key === 'same_device_target')
  assert.equal(device.ok, false)
  const ratio = rows.find((r) => r.key === 'parameter_delta_ratio')
  assert.equal(ratio.ok, null) // neutral — the gate itself is backend-owned
  const { passed, total } = passedFairnessCount(rows)
  assert.equal(total, 6) // ratio row is not counted (ok === null)
  assert.equal(passed, 5)
})

test('ladderView passes the backend ladder through verbatim', () => {
  const v = ladderView(comparison)
  assert.equal(v.label, 'paired empirical')
  assert.equal(v.claimLevel, 'empirical')
  assert.equal(v.metCount, 4)
  assert.equal(v.steps.length, 2)
  assert.equal(v.steps[1].ok, false)
})

test('caveats surface backend interpretation warnings', () => {
  assert.equal(caveats(comparison)[0].code, 'single_seed')
  assert.deepEqual(caveats({}), [])
})

test('diagnosticValues prefers the diagnostics endpoint, falls back to summary, marks absents', () => {
  const summary = {
    grad_var_mean: 1.2e-3, meyer_wallach_q: 0.61, expressibility_kl: null,
    availability: { expressibility_kl: { available: false, reason: 'MPS backend has no statevector' } },
  }
  const values = diagnosticValues({ grad_snr: 8.4 }, summary)
  assert.equal(values.grad_snr.value, 8.4) // from diagnostics endpoint
  assert.equal(values.grad_var_mean.value, 1.2e-3) // from summary
  assert.equal(values.expressibility_kl.available, false)
  assert.match(values.expressibility_kl.reason, /MPS backend/)
  assert.equal(hasAnyDiagnostic(values), true)
})

test('diagnosticValues reports grad_snr absent when nothing provides it', () => {
  const values = diagnosticValues(null, { grad_var_mean: 1e-3 })
  assert.equal(values.grad_snr.available, false)
  assert.match(values.grad_snr.reason, /not computed/)
})
