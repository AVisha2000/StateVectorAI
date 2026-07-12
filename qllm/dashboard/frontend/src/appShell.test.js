import assert from 'node:assert/strict'
import test from 'node:test'

import {
  NAV_GROUPS,
  NAV_ITEMS,
  LEGACY_REDIRECTS,
  THEME_STORAGE_KEY,
  resolveInitialTheme,
  navTitleForPath,
} from './appShell.js'

test('saved theme wins and invalid storage falls back to system preference', () => {
  assert.equal(THEME_STORAGE_KEY, 'qllm-theme')
  assert.equal(resolveInitialTheme('dark', true), 'dark')
  assert.equal(resolveInitialTheme('light', false), 'light')
  assert.equal(resolveInitialTheme('invalid', true), 'light')
  assert.equal(resolveInitialTheme(null, false), 'dark')
})

test('redesign navigation groups are ordered and complete', () => {
  assert.deepEqual(
    NAV_GROUPS.map((group) => group.title),
    ['Research', 'Experiments', 'System'],
  )
  assert.deepEqual(
    NAV_ITEMS.map((item) => [item.to, item.label]),
    [
      ['/', 'Overview'],
      ['/discover', 'Discover'],
      ['/library', 'Library'],
      ['/atlas', 'Atlas'],
      ['/designer', 'Designer'],
      ['/bench', 'Bench'],
      ['/runs', 'Runs'],
      ['/studies', 'Studies'],
      ['/verdicts', 'Verdicts'],
      ['/datasets', 'Datasets'],
      ['/system', 'Queue & Backends'],
    ],
  )
})

test('every nav item carries an icon and the overview item ends its route', () => {
  for (const item of NAV_ITEMS) {
    assert.ok(item.icon, `${item.to} is missing an icon`)
  }
  const overview = NAV_ITEMS.find((item) => item.to === '/')
  assert.equal(overview.end, true)
})

test('legacy routes redirect to a real new surface', () => {
  const surfaces = new Set(NAV_ITEMS.map((item) => item.to))
  for (const [from, to] of Object.entries(LEGACY_REDIRECTS)) {
    assert.ok(from.startsWith('/'), `${from} must be an absolute path`)
    assert.ok(surfaces.has(to), `${from} redirects to unknown surface ${to}`)
  }
})

test('breadcrumb title resolves from the active path', () => {
  assert.equal(navTitleForPath('/'), 'Overview')
  assert.equal(navTitleForPath('/runs'), 'Runs')
  assert.equal(navTitleForPath('/runs/abc123'), 'Runs')
  assert.equal(navTitleForPath('/atlas'), 'Atlas')
  assert.equal(navTitleForPath('/unknown'), '')
})
