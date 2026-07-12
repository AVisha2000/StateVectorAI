import assert from 'node:assert/strict'
import test from 'node:test'
import { toStylesheet, ATLAS_TOKEN_KEYS } from './atlasStyle.js'

const tokens = { '--q': '#c05fc0', '--c': '#3987e5', '--warn': '#e0a52e', '--accent': '#8f8af0', '--null': '#8f8ea0', '--ink': '#fff', '--ink2': '#bbb', '--axis': '#3a3a42', '--surface2': '#1e1e25', '--hair': 'rgba(255,255,255,.1)' }

test('toStylesheet colors outcomes from tokens — classical_holds is classical-blue, never green', () => {
  const sheet = toStylesheet(tokens)
  const bySel = Object.fromEntries(sheet.filter((r) => r.selector.startsWith('.outcome-')).map((r) => [r.selector, r.style['background-color']]))
  assert.equal(bySel['.outcome-classical_holds'], '#3987e5') // classical blue, full prominence
  assert.equal(bySel['.outcome-quantum_candidate'], '#c05fc0')
  assert.equal(bySel['.outcome-unexplored'], '#8f8ea0')
  // no outcome fill is the good/green token
  assert.ok(!Object.values(bySel).includes('#3fbf3f'))
})

test('claim and replication use different channels (border-width vs border-style)', () => {
  const sheet = toStylesheet(tokens)
  const cell = sheet.find((r) => r.selector === '.cell')
  assert.match(String(cell.style['border-width']), /mapData\(claimRank/) // claim → width
  const repDashed = sheet.find((r) => r.selector === '.cell[replicationRank = 0]')
  const repSolid = sheet.find((r) => r.selector === '.cell[replicationRank > 0]')
  assert.equal(repDashed.style['border-style'], 'dashed') // replication → style, a different channel
  assert.equal(repSolid.style['border-style'], 'solid')
})

test('kind maps to node shape', () => {
  const sheet = toStylesheet(tokens)
  const shape = (sel) => sheet.find((r) => r.selector === sel)?.style.shape
  assert.equal(shape('.kind-quantum_only'), 'hexagon')
  assert.equal(shape('.kind-suggested'), 'diamond')
})

test('toStylesheet degrades to fallbacks with an empty tokenMap and exposes required keys', () => {
  const sheet = toStylesheet({})
  assert.ok(sheet.length > 5)
  assert.ok(ATLAS_TOKEN_KEYS.includes('--q') && ATLAS_TOKEN_KEYS.includes('--null'))
})
