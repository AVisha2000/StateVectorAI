import assert from 'node:assert/strict'
import test from 'node:test'

import { get, responseError } from './api.js'

test('responseError uses FastAPI detail when available', async () => {
  const error = await responseError({
    status: 404,
    json: async () => ({ detail: 'Run does not exist' }),
  }, '/run/99')

  assert.equal(error.message, 'Run does not exist')
})

test('responseError falls back to path and status for non-JSON errors', async () => {
  const error = await responseError({
    status: 503,
    json: async () => { throw new SyntaxError('not JSON') },
  }, '/suites')

  assert.equal(error.message, '/suites: 503')
})

test('GET surfaces FastAPI detail', async (context) => {
  const originalFetch = globalThis.fetch
  context.after(() => { globalThis.fetch = originalFetch })
  globalThis.fetch = async () => ({
    ok: false,
    status: 400,
    json: async () => ({ detail: 'Invalid research slice' }),
  })

  await assert.rejects(get('/explore/task/bad'), /Invalid research slice/)
})
