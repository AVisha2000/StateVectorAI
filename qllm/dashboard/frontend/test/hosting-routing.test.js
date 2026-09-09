import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

test('Vercel keeps API handling ahead of the SPA fallback for every deep link', async () => {
  const config = JSON.parse(await readFile(new URL('../vercel.json', import.meta.url), 'utf8'))
  assert.deepEqual(config.rewrites, [
    { source: '/api/:path*', destination: '/api/playground' },
    { source: '/(.*)', destination: '/index.html' },
  ])
})
