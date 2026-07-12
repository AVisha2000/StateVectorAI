import assert from 'node:assert/strict'
import test from 'node:test'
import { readChangeToken, STREAM_REFRESH_KEYS } from './stream.js'

test('readChangeToken extracts the content-addressed token', () => {
  assert.equal(readChangeToken({ change_token: 'abc123', jobs: [] }), 'abc123')
})

test('readChangeToken returns null for junk / heartbeats', () => {
  assert.equal(readChangeToken(null), null)
  assert.equal(readChangeToken({ jobs: [] }), null)
  assert.equal(readChangeToken({ change_token: 42 }), null)
})

test('the stream refreshes jobs, overview, and active run detail — never overwrites them', () => {
  // A changed token must trigger a refetch of the full authoritative queries,
  // because the stream payload is a lean projection missing run_name/preset/etc.
  assert.deepEqual(STREAM_REFRESH_KEYS, [['jobs'], ['overview'], ['workspace']])
})
