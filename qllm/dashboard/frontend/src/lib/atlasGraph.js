// Pure ontology → Cytoscape elements transform. Returns plain arrays in
// Cytoscape's documented element JSON shape; imports NOTHING from cytoscape, so
// it runs under node --test with no DOM. Styling is data-driven: each cell node
// carries outcome / claimRank / replicationRank / kind so the stylesheet
// (atlasStyle.js) can encode them on orthogonal channels.
import { claimRank, replicationRank } from './atlasModel.js'

// `expanded` is a Set of domain ids to reveal; omit to expand all.
export function toElements(resolved, { expanded } = {}) {
  const domains = resolved?.domains || []
  const exp = expanded instanceof Set ? expanded : new Set(domains.map((d) => d.id))
  const nodes = []
  const edges = []

  for (const d of domains) {
    nodes.push({ data: { id: d.id, label: d.label, type: 'domain' }, classes: 'domain' })
    if (!exp.has(d.id)) continue
    for (const comp of d.components || []) {
      nodes.push({ data: { id: comp.id, label: comp.label, parent: d.id, type: 'component' }, classes: 'component' })
      for (const c of comp.cells || []) {
        nodes.push({
          data: {
            id: c.id,
            label: c.label,
            parent: comp.id,
            type: 'cell',
            outcome: c.outcome_class,
            kind: c.kind,
            claimLevel: c.claim_level,
            replicationStatus: c.replication_status,
            claimRank: claimRank(c.claim_level),
            replicationRank: replicationRank(c.replication_status),
            provenance: c.provenance,
            verdictId: c.verdict_id ?? null,
            areaId: c.area_id ?? null,
          },
          classes: `cell outcome-${c.outcome_class} kind-${c.kind} prov-${c.provenance}`,
        })
      }
    }
  }

  const presentCells = new Set(nodes.filter((n) => n.data.type === 'cell').map((n) => n.data.id))
  for (const r of resolved?.relations || []) {
    if (presentCells.has(r.from_cell) && presentCells.has(r.to_cell)) {
      edges.push({
        data: { id: `rel:${r.from_cell}:${r.to_cell}`, source: r.from_cell, target: r.to_cell, relation: r.type },
        classes: 'relation',
      })
    }
  }

  return { nodes, edges }
}

// Convenience for the wrapper: a flat elements array (Cytoscape accepts either).
export function toElementArray(resolved, opts) {
  const { nodes, edges } = toElements(resolved, opts)
  return [...nodes, ...edges]
}
