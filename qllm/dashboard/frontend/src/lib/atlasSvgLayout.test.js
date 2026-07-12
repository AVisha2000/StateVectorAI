import assert from 'node:assert/strict'
import test from 'node:test'
import { layoutTree, NODE_W } from './atlasSvgLayout.js'
import { resolveOntology } from './atlasModel.js'
import { ATLAS_SEED } from './atlasOntology.seed.js'

const resolved = resolveOntology(ATLAS_SEED, [], null)

test('layoutTree places domains, components, and cells in three columns', () => {
  const { nodes, width, height } = layoutTree(resolved)
  const cols = { domain: new Set(), component: new Set(), cell: new Set() }
  for (const n of nodes) cols[n.type].add(n.x)
  assert.equal(cols.domain.size, 1) // all domains share the domain column
  assert.equal(cols.component.size, 1)
  assert.equal(cols.cell.size, 1)
  assert.ok([...cols.domain][0] < [...cols.component][0])
  assert.ok([...cols.component][0] < [...cols.cell][0])
  assert.ok(width > 0 && height > 200)
})

test('every cell gets a distinct row and carries its distinct channels', () => {
  const { nodes } = layoutTree(resolved)
  const cells = nodes.filter((n) => n.type === 'cell')
  const ys = cells.map((c) => c.y)
  assert.equal(new Set(ys).size, ys.length) // no two cells overlap
  const c = cells[0]
  assert.ok('outcome' in c && 'claimRank' in c && 'replicationRank' in c && 'kind' in c)
})

test('parents center on their children vertically', () => {
  const { nodes } = layoutTree(resolved)
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const d0 = resolved.domains[0]
  const comp0 = d0.components[0]
  const compNode = byId.get(comp0.id)
  const cellNodes = comp0.cells.map((c) => byId.get(c.id))
  const mid = (cellNodes[0].y + cellNodes[cellNodes.length - 1].y) / 2
  assert.ok(Math.abs(compNode.y - mid) < 0.001)
})

test('collapsing a domain drops its components and cells', () => {
  const first = resolved.domains[0].id
  const expanded = new Set(resolved.domains.map((d) => d.id))
  expanded.delete(first)
  const { nodes } = layoutTree(resolved, { expanded })
  assert.ok(nodes.some((n) => n.id === first && n.type === 'domain'))
  assert.ok(!nodes.some((n) => String(n.id).startsWith(`${first}::`)))
})

test('hierarchy edges run left-to-right; relation edges link two cells', () => {
  const { edges } = layoutTree(resolved)
  const hier = edges.filter((e) => e.kind === 'hierarchy')
  const rel = edges.filter((e) => e.kind === 'relation')
  assert.ok(hier.length >= 1 && rel.length >= 1)
  assert.ok(hier.every((e) => e.x1 <= e.x2 + 0.001)) // tree flows left→right
  assert.ok(hier.every((e) => typeof e.x1 === 'number' && typeof e.y2 === 'number'))
  assert.ok(rel.every((e) => e.relation)) // relation edges carry their type
  assert.ok(NODE_W.domain > 0)
})
