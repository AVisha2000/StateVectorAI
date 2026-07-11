import assert from 'node:assert/strict'
import test from 'node:test'

import { NAV_ITEMS, THEME_STORAGE_KEY, resolveInitialTheme } from './appShell.js'

test('saved theme wins and invalid storage falls back to system preference', () => {
  assert.equal(THEME_STORAGE_KEY, 'qllm-theme')
  assert.equal(resolveInitialTheme('dark', true), 'dark')
  assert.equal(resolveInitialTheme('light', false), 'light')
  assert.equal(resolveInitialTheme('invalid', true), 'light')
  assert.equal(resolveInitialTheme(null, false), 'dark')
})

test('research workflow navigation remains complete and ordered', () => {
  assert.deepEqual(
    NAV_ITEMS.map(({ to, label }) => [to, label]),
    [
      ['/', 'Overview'],
      ['/explore', 'Explore'],
      ['/experiments', 'Experiments'],
      ['/models', 'Model Builder'],
      ['/studies', 'Studies'],
      ['/results', 'Results'],
      ['/datasets', 'Datasets & Tasks'],
      ['/gpu', 'System'],
      ['/docs', 'Docs'],
    ],
  )
})
