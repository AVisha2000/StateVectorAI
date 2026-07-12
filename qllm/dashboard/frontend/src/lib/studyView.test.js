import assert from 'node:assert/strict'
import test from 'node:test'
import { studySummary, deltaPairs, winConsistency, studyLadder, studyJobs, studyCaveats } from './studyView.js'

const STUDY = {
  name: 'qffn-multiseed',
  evidence: {
    label: 'paired empirical', reason: 'candidate leads across seeds',
    fair_pairs: 4, complete_pairs: 5, wins: 3, mean_delta_val_ppl: -0.12, std_delta_val_ppl: 0.08,
    rerun_required_pairs: 1,
    ladder: [{ key: 'multi_seed', label: 'Multiple seeds', ok: true }],
    comparisons: [
      { delta_val_ppl: -0.2, fair: true, rerun_required: false, cell: 'q4/d2' },
      { delta_val_ppl: -0.1, fair: true, rerun_required: false, cell: 'q6/d2' },
      { delta_val_ppl: 0.05, fair: true, rerun_required: false, cell: 'q8/d2' },
      { delta_val_ppl: -0.3, fair: true, rerun_required: true, cell: 'q4/d3' }, // dropped (rerun)
      { delta_val_ppl: null, fair: true, rerun_required: false }, // dropped (no delta)
      { delta_val_ppl: -0.4, fair: false, rerun_required: false }, // dropped (unfair)
    ],
  },
  jobs: [{ study_sweep: { n_qubits: 4, n_circuit_layers: 2 }, final_run: { val_ppl: 3.4 } }],
  interpretation_warnings: [{ code: 'single_task_instance', title: 'One instance', message: 'Multi-seed, single task.' }],
}

test('studySummary reports aggregate counts and mean/std, not a composite score', () => {
  const s = studySummary(STUDY)
  assert.equal(s.fairPairs, 4)
  assert.equal(s.wins, 3)
  assert.equal(s.meanDelta, -0.12)
  assert.equal(s.stdDelta, 0.08)
  assert.ok(!('score' in s) && !('advantage' in s))
})

test('deltaPairs keeps only fair, non-rerun, non-null pairs and indexes them', () => {
  const pairs = deltaPairs(STUDY)
  assert.deepEqual(pairs.map((p) => p.delta), [-0.2, -0.1, 0.05])
  assert.deepEqual(pairs.map((p) => p.i), [1, 2, 3])
  assert.equal(pairs[0].cell, 'q4/d2')
})

test('winConsistency is a fraction of fair pairs, reported with its denominator', () => {
  const w = winConsistency(STUDY)
  assert.deepEqual(w, { wins: 2, total: 3, fraction: 2 / 3 }) // -0.2 and -0.1 win; +0.05 loses
  assert.deepEqual(winConsistency({}), { wins: 0, total: 0, fraction: null })
})

test('ladder / jobs / caveats accessors are null-safe', () => {
  assert.equal(studyLadder(STUDY).length, 1)
  assert.equal(studyJobs(STUDY)[0].final_run.val_ppl, 3.4)
  assert.equal(studyCaveats(STUDY)[0].code, 'single_task_instance')
  assert.deepEqual(studyLadder(null), [])
})
