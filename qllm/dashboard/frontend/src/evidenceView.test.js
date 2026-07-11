import assert from 'node:assert/strict'
import test from 'node:test'
import { capabilityRows, displayValue, uniqueWarnings } from './evidenceView.js'

test('warnings deduplicate only identical structured evidence', () => {
  const base = { code: 'single_seed', severity: 'warning', title: 'One pair', message: 'Repeat.', evidence: { independent_pairs: 1 } }
  const distinct = { ...base, evidence: { independent_pairs: 1, cell: 'q4' } }
  assert.deepEqual(uniqueWarnings([base, { ...base }, distinct]), [base, distinct])
})

test('missing values remain unavailable rather than becoming zero', () => {
  assert.equal(displayValue(null), 'unavailable')
  assert.equal(displayValue(0), '0')
  assert.equal(displayValue(false), 'no')
})

test('capability rows retain component, status, and exactness', () => {
  const rows = capabilityRows({ attention: { exactness: 'sampled', capabilities: { sampling: { status: 'supported', exactness: 'sampled' } } } })
  assert.deepEqual(rows, [{ component: 'attention', capability: 'sampling', status: 'supported', exactness: 'sampled' }])
})
