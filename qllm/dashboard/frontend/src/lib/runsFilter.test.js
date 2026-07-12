import assert from 'node:assert/strict'
import test from 'node:test'
import { filterRuns, uniqueDatasets } from './runsFilter.js'

const JOBS = [
  { id: 1, run_name: 'qrnn-s42', status: 'running', preset_id: 'quantum-ffn-4q', dataset_name: 'monitored_ising', model_family: 'qrnn' },
  { id: 2, run_name: 'gru-s42', status: 'done', preset_id: 'classical-small', dataset_name: 'monitored_ising', model_family: 'gru' },
  { id: 3, run_name: 'qattn-s77', status: 'error', preset_id: 'quantum-attn', dataset_name: 'contextual', model_family: 'qattn' },
]

test('status filter', () => {
  assert.deepEqual(filterRuns(JOBS, { status: 'error' }).map((j) => j.id), [3])
  assert.equal(filterRuns(JOBS, { status: 'all' }).length, 3)
})

test('dataset filter', () => {
  assert.deepEqual(filterRuns(JOBS, { dataset: 'contextual' }).map((j) => j.id), [3])
})

test('search matches run name, preset, dataset, family, id — case-insensitive', () => {
  assert.deepEqual(filterRuns(JOBS, { search: 'QRNN' }).map((j) => j.id), [1])
  assert.deepEqual(filterRuns(JOBS, { search: 'classical' }).map((j) => j.id), [2])
  assert.deepEqual(filterRuns(JOBS, { search: 'contextual' }).map((j) => j.id), [3])
  assert.deepEqual(filterRuns(JOBS, { search: '  ' }).length, 3) // blank = no filter
})

test('filters compose (status AND dataset AND search)', () => {
  assert.deepEqual(filterRuns(JOBS, { status: 'done', dataset: 'monitored_ising', search: 'gru' }).map((j) => j.id), [2])
  assert.equal(filterRuns(JOBS, { status: 'running', dataset: 'contextual' }).length, 0)
})

test('uniqueDatasets is sorted and de-duplicated', () => {
  assert.deepEqual(uniqueDatasets(JOBS), ['contextual', 'monitored_ising'])
  assert.deepEqual(uniqueDatasets(null), [])
})
