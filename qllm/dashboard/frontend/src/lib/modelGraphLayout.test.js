import assert from 'node:assert/strict'
import test from 'node:test'
import { layoutModelGraph, nodeResourceHint, NODE_W } from './modelGraphLayout.js'

const GRAPH = {
  nodes: [
    { id: 'tokens', label: 'Tokens', kind: 'input' },
    { id: 'embed', label: 'Embedding', kind: 'classical' },
    { id: 'qffn', label: 'Quantum FFN', kind: 'quantum', meta: { resource: { n_qubits: 4, n_circuit_layers: 2, backend: 'pennylane' } } },
    { id: 'head', label: 'Head', kind: 'classical' },
    { id: 'out', label: 'Logits', kind: 'output' },
  ],
  edges: [['tokens', 'embed'], ['embed', 'qffn'], ['qffn', 'head'], ['head', 'out']],
}

test('layoutModelGraph lays a linear model out left-to-right by layer', () => {
  const { nodes, edges, width, height } = layoutModelGraph(GRAPH)
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))
  assert.equal(byId.tokens.layer, 0)
  assert.equal(byId.embed.layer, 1)
  assert.equal(byId.qffn.layer, 2)
  assert.equal(byId.out.layer, 4)
  // x increases with layer
  assert.ok(byId.tokens.x < byId.embed.x && byId.embed.x < byId.qffn.x)
  // edges run left→right, from right edge of source
  assert.ok(edges.every((e) => e.x1 <= e.x2 + 0.001))
  assert.ok(edges.every((e) => Math.abs(e.x1 - (byId[e.from].x + NODE_W)) < 0.001))
  assert.ok(width > 0 && height > 0)
})

test('branching layers stay centered and share a column', () => {
  const g = {
    nodes: [
      { id: 'in', label: 'in', kind: 'input' },
      { id: 'a', label: 'a', kind: 'classical' },
      { id: 'b', label: 'b', kind: 'quantum' },
      { id: 'merge', label: 'm', kind: 'classical' },
    ],
    edges: [['in', 'a'], ['in', 'b'], ['a', 'merge'], ['b', 'merge']],
  }
  const { nodes } = layoutModelGraph(g)
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))
  assert.equal(byId.a.x, byId.b.x) // same layer/column
  assert.notEqual(byId.a.y, byId.b.y) // different rows
  assert.equal(byId.merge.layer, 2)
})

test('empty / malformed graphs are safe', () => {
  assert.deepEqual(layoutModelGraph(null), { nodes: [], edges: [], width: 0, height: 0 })
  assert.deepEqual(layoutModelGraph({ nodes: [], edges: [] }).nodes, [])
})

test('nodeResourceHint summarizes a quantum node', () => {
  assert.equal(nodeResourceHint({ meta: { resource: { n_qubits: 4, n_circuit_layers: 2, backend: 'pennylane' } } }), '4q · d2 · pennylane')
  assert.equal(nodeResourceHint({ kind: 'classical' }), null)
})
