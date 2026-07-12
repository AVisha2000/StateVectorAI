import assert from 'node:assert/strict'
import test from 'node:test'
import { toElements, toElementArray } from './atlasGraph.js'
import { resolveOntology } from './atlasModel.js'
import { ATLAS_SEED } from './atlasOntology.seed.js'

const resolved = resolveOntology(ATLAS_SEED, [], null)

test('toElements emits a domain → component → cell tree via hierarchy edges', () => {
  const { nodes, edges } = toElements(resolved)
  const domains = nodes.filter((n) => n.data.type === 'domain')
  const components = nodes.filter((n) => n.data.type === 'component')
  const cells = nodes.filter((n) => n.data.type === 'cell')
  const hier = edges.filter((e) => e.classes === 'hierarchy')
  assert.ok(domains.length >= 4)
  // every component and cell is reachable via a hierarchy edge (tree, not compound)
  const targets = new Set(hier.map((e) => e.data.target))
  assert.ok(components.every((c) => targets.has(c.data.id)))
  assert.ok(cells.every((c) => targets.has(c.data.id)))
  // cell carries the distinct channels as separate data fields
  const cell = cells[0]
  assert.ok('outcome' in cell.data && 'claimRank' in cell.data && 'replicationRank' in cell.data && 'kind' in cell.data)
  assert.ok(!('parent' in cell.data)) // no compound parent — layout is edge-driven
  assert.ok(cell.classes.includes(`outcome-${cell.data.outcome}`))
})

test('collapsing a domain hides its components, cells, and their hierarchy edges', () => {
  const first = resolved.domains[0].id
  const expanded = new Set(resolved.domains.map((d) => d.id))
  expanded.delete(first)
  const { nodes, edges } = toElements(resolved, { expanded })
  assert.ok(nodes.some((n) => n.data.id === first)) // domain node still shown
  assert.ok(!nodes.some((n) => String(n.data.id).startsWith(`${first}::`))) // its components hidden
  assert.ok(!edges.some((e) => e.data.source === first)) // no hierarchy edges out of it
})

test('relation edges appear only when both endpoint cells are present', () => {
  const { edges } = toElements(resolved)
  const relations = edges.filter((e) => e.classes === 'relation')
  assert.ok(relations.length >= 1)
  assert.ok(relations.every((e) => e.data.source && e.data.target))
  // collapse everything: no cells → no relation edges (hierarchy edges also gone)
  const { edges: none } = toElements(resolved, { expanded: new Set() })
  assert.equal(none.filter((e) => e.classes === 'relation').length, 0)
})

test('toElementArray concatenates nodes and edges', () => {
  const arr = toElementArray(resolved)
  const { nodes, edges } = toElements(resolved)
  assert.equal(arr.length, nodes.length + edges.length)
})
