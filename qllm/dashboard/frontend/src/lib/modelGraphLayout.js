// Pure layered-DAG layout for a run's model graph (GET /jobs/{id}/model-graph:
// { nodes:[{id,label,kind,meta}], edges:[[from,to]] }). Longest-path layering
// (Kahn topo) puts inputs on the left and outputs on the right; each layer is
// centered vertically. Framework-free for node --test.

export const NODE_W = 150
export const NODE_H = 44
const COL_W = 188
const ROW_H = 64
const PAD_X = 16
const PAD_T = 18

export function layoutModelGraph(graph) {
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes : []
  const edges = Array.isArray(graph?.edges) ? graph.edges : []
  if (nodes.length === 0) return { nodes: [], edges: [], width: 0, height: 0 }

  const byId = new Map(nodes.map((n) => [n.id, n]))
  const adj = new Map(nodes.map((n) => [n.id, []]))
  const indeg = new Map(nodes.map((n) => [n.id, 0]))
  for (const e of edges) {
    const a = Array.isArray(e) ? e[0] : e?.from
    const b = Array.isArray(e) ? e[1] : e?.to
    if (byId.has(a) && byId.has(b)) { adj.get(a).push(b); indeg.set(b, indeg.get(b) + 1) }
  }

  // Longest-path layer assignment via topological order.
  const layer = new Map(nodes.map((n) => [n.id, 0]))
  const deg = new Map(indeg)
  const queue = nodes.filter((n) => deg.get(n.id) === 0).map((n) => n.id)
  while (queue.length) {
    const u = queue.shift()
    for (const v of adj.get(u)) {
      layer.set(v, Math.max(layer.get(v), layer.get(u) + 1))
      deg.set(v, deg.get(v) - 1)
      if (deg.get(v) === 0) queue.push(v)
    }
  }

  const layers = new Map()
  for (const n of nodes) {
    const L = layer.get(n.id) || 0
    if (!layers.has(L)) layers.set(L, [])
    layers.get(L).push(n)
  }
  const maxRows = Math.max(...[...layers.values()].map((l) => l.length))
  const maxLayer = Math.max(...[...layers.keys()])

  const positioned = []
  for (const [L, ns] of [...layers.entries()].sort((a, b) => a[0] - b[0])) {
    const offset = (maxRows - ns.length) / 2 // center the layer vertically
    ns.forEach((n, i) => {
      positioned.push({ ...n, x: PAD_X + L * COL_W, y: PAD_T + (i + offset) * ROW_H, layer: L })
    })
  }

  const pmap = new Map(positioned.map((n) => [n.id, n]))
  const lines = edges
    .map((e) => {
      const a = Array.isArray(e) ? e[0] : e?.from
      const b = Array.isArray(e) ? e[1] : e?.to
      const s = pmap.get(a)
      const t = pmap.get(b)
      if (!s || !t) return null
      return { from: a, to: b, x1: s.x + NODE_W, y1: s.y + NODE_H / 2, x2: t.x, y2: t.y + NODE_H / 2 }
    })
    .filter(Boolean)

  return {
    nodes: positioned,
    edges: lines,
    width: PAD_X + (maxLayer + 1) * COL_W + 12,
    height: PAD_T + maxRows * ROW_H + 12,
  }
}

// Pull a short resource hint (e.g. "4q · pennylane") off a quantum node.
export function nodeResourceHint(node) {
  const r = node?.meta?.resource
  if (!r) return null
  const bits = []
  if (r.n_qubits) bits.push(`${r.n_qubits}q`)
  if (r.n_circuit_layers) bits.push(`d${r.n_circuit_layers}`)
  if (r.backend) bits.push(r.backend)
  return bits.length ? bits.join(' · ') : null
}
