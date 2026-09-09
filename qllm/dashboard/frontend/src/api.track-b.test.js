import assert from 'node:assert/strict'
import test from 'node:test'

import { api } from './api.js'

test('trackBReadiness uses the read-only Atlas endpoint', async () => {
  const previousFetch = globalThis.fetch
  let requested = null
  globalThis.fetch = async (url, options) => {
    requested = { url, options }
    return new Response(JSON.stringify({ proposal_only: true }), { status: 200 })
  }
  try {
    assert.deepEqual(await api.trackBReadiness(), { proposal_only: true })
    assert.deepEqual(requested, { url: '/api/atlas/track-b-readiness', options: undefined })
  } finally {
    globalThis.fetch = previousFetch
  }
})
