import assert from 'node:assert/strict'
import test from 'node:test'
import { toElements, toElementArray } from './atlasGraph.js'
import { resolveOntology } from './atlasModel.js'
import { ATLAS_SEED } from './atlasOntology.seed.js'

const resolved = resolveOntology(ATLAS_SEED, [], null)

test('toElements emits domain → component → cell compound nodes when expanded', () => {
  const { nodes } = toElements(resolved)
  const domains = nodes.filter((n) => n.data.type === 'domain')
  const components = nodes.filter((n) => n.data.type === 'component')
  const cells = nodes.filter((n) => n.data.type === 'cell')
  assert.ok(domains.length >= 4)
  assert.ok(components.every((c) => c.data.parent)) // components parented to a domain
  assert.ok(cells.every((c) => c.data.parent)) // cells parented to a component
  // cell carries the distinct channels as separate data fields
  const cell = cells[0]
  assert.ok('outcome' in cell.data && 'claimRank' in cell.data && 'replicationRank' in cell.data && 'kind' in cell.data)
  assert.ok(cell.classes.includes(`outcome-${cell.data.outcome}`))
})

test('collapsing a domain hides its components and cells', () => {
  const first = resolved.domains[0].id
  const expanded = new Set(resolved.domains.map((d) => d.id))
  expanded.delete(first)
  const { nodes } = toElements(resolved, { expanded })
  assert.ok(nodes.some((n) => n.data.id === first)) // domain node still shown
  assert.ok(!nodes.some((n) => n.data.parent === first)) // its components hidden
})

test('relation edges appear only when both endpoint cells are present', () => {
  const { edges } = toElements(resolved)
  // seed relations connect existing cells → at least one edge present when all expanded
  assert.ok(edges.length >= 1)
  assert.ok(edges.every((e) => e.data.source && e.data.target && e.classes === 'relation'))
  // collapse everything: no cells → no relation edges
  const { edges: none } = toElements(resolved, { expanded: new Set() })
  assert.equal(none.length, 0)
})

test('toElementArray concatenates nodes and edges', () => {
  const arr = toElementArray(resolved)
  const { nodes, edges } = toElements(resolved)
  assert.equal(arr.length, nodes.length + edges.length)
})
