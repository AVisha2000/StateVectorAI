// Pure layered-tree layout for the Atlas graph, rendered as hand-authored SVG
// (reliable everywhere, unlike a canvas lib). Deterministic and framework-free
// so it runs under node --test. Produces absolute node boxes + edge line
// endpoints; the SVG component just draws them.
import { claimRank, replicationRank } from './atlasModel.js'

export const NODE_W = Object.freeze({ domain: 150, component: 140, cell: 170 })
export const NODE_H = Object.freeze({ domain: 34, component: 28, cell: 40 })
const COL_X = Object.freeze({ domain: 16, component: 240, cell: 470 })
const ROW_H = 50
const TOP = 26

// Lay out domains → components → cells as a left-to-right tree. Each cell claims
// a row; parents center on their children. Collapsed domains render alone.
export function layoutTree(resolved, { expanded } = {}) {
  const domains = resolved?.domains || []
  const exp = expanded instanceof Set ? expanded : new Set(domains.map((d) => d.id))
  const nodes = []
  const edges = []
  let row = 0

  const rowY = (r) => TOP + r * ROW_H

  for (const d of domains) {
    const compNodes = []
    if (exp.has(d.id)) {
      for (const comp of d.components || []) {
        const cellYs = []
        for (const c of comp.cells || []) {
          const y = rowY(row)
          nodes.push({
            id: c.id, type: 'cell', label: c.label, x: COL_X.cell, y,
            outcome: c.outcome_class, kind: c.kind,
            claimRank: claimRank(c.claim_level), replicationRank: replicationRank(c.replication_status),
            cell: c,
          })
          cellYs.push(y)
          row += 1
        }
        if (cellYs.length === 0) { row += 1; continue }
        const compY = (cellYs[0] + cellYs[cellYs.length - 1]) / 2
        const compNode = { id: comp.id, type: 'component', label: comp.label, x: COL_X.component, y: compY }
        nodes.push(compNode)
        compNodes.push(compNode)
        for (const c of comp.cells || []) edges.push({ from: comp.id, to: c.id, kind: 'hierarchy' })
      }
    }
    const domY = compNodes.length ? (compNodes[0].y + compNodes[compNodes.length - 1].y) / 2 : rowY(row)
    nodes.push({ id: d.id, type: 'domain', label: d.label, x: COL_X.domain, y: domY })
    for (const comp of compNodes) edges.push({ from: d.id, to: comp.id, kind: 'hierarchy' })
    if (compNodes.length === 0) row += 1 // collapsed domain still occupies a row
    row += 1 // gap between domains
  }

  // relation edges only between present cells
  const present = new Set(nodes.filter((n) => n.type === 'cell').map((n) => n.id))
  for (const r of resolved?.relations || []) {
    if (present.has(r.from_cell) && present.has(r.to_cell)) {
      edges.push({ from: r.from_cell, to: r.to_cell, kind: 'relation', relation: r.type })
    }
  }

  // resolve edge endpoints: right edge of source → left edge of target
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const lines = edges
    .map((e) => {
      const a = byId.get(e.from)
      const b = byId.get(e.to)
      if (!a || !b) return null
      return {
        ...e,
        x1: a.x + NODE_W[a.type], y1: a.y + NODE_H[a.type] / 2,
        x2: b.x, y2: b.y + NODE_H[b.type] / 2,
      }
    })
    .filter(Boolean)

  const height = Math.max(rowY(row), 240)
  const width = COL_X.cell + NODE_W.cell + 20
  return { nodes, edges: lines, width, height }
}
