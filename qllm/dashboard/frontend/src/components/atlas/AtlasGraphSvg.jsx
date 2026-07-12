import { useMemo } from 'react'
import { layoutTree, NODE_W, NODE_H } from '../../lib/atlasSvgLayout.js'

// Hand-authored SVG Atlas graph — renders reliably in any browser (no canvas lib).
// Encodes the same four orthogonal channels as the list/legend:
//   fill = outcome, stroke-width = claim rank, dashed stroke = no replication,
//   shape = cell kind.
const OUTCOME_FILL = {
  quantum_candidate: 'var(--q)',
  quantum_only: 'var(--q)',
  classical_holds: 'var(--c)',
  open: 'var(--warn)',
  suggested: 'var(--accent)',
  unexplored: 'var(--null)',
}

function truncate(s, n = 26) {
  const str = String(s || '')
  return str.length > n ? `${str.slice(0, n - 1)}…` : str
}

function CellShape({ n, selected, onSelect }) {
  const w = NODE_W.cell
  const h = NODE_H.cell
  const { x, y } = n
  const fill = OUTCOME_FILL[n.outcome] || 'var(--null)'
  const strokeWidth = 1 + n.claimRank * 0.55 // claim strength → border width
  const dash = n.replicationRank === 0 ? '5 3' : undefined // replication → style
  const common = {
    fill,
    stroke: selected ? 'var(--accent)' : 'var(--ink)',
    strokeWidth: selected ? 3 : strokeWidth,
    strokeDasharray: dash,
    style: { cursor: 'pointer' },
  }
  let shape
  if (n.kind === 'quantum_only') {
    const p = `${x + 14},${y} ${x + w - 14},${y} ${x + w},${y + h / 2} ${x + w - 14},${y + h} ${x + 14},${y + h} ${x},${y + h / 2}`
    shape = <polygon points={p} {...common} />
  } else if (n.kind === 'suggested') {
    const p = `${x + w / 2},${y} ${x + w},${y + h / 2} ${x + w / 2},${y + h} ${x},${y + h / 2}`
    shape = <polygon points={p} {...common} />
  } else if (n.kind === 'unexplored') {
    shape = <ellipse cx={x + w / 2} cy={y + h / 2} rx={w / 2} ry={h / 2} {...common} />
  } else {
    shape = <rect x={x} y={y} width={w} height={h} rx={8} {...common} />
  }
  return (
    <g onClick={() => onSelect?.(n.id)} role="button" aria-label={n.label}>
      {shape}
      <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fontSize="10" fill="#fff" style={{ pointerEvents: 'none' }}>
        {truncate(n.label)}
      </text>
    </g>
  )
}

function BoxNode({ n }) {
  const w = NODE_W[n.type]
  const h = NODE_H[n.type]
  return (
    <g>
      <rect x={n.x} y={n.y} width={w} height={h} rx={7}
        fill="var(--surface2)" stroke="var(--hair)" strokeWidth={1} />
      <text x={n.x + w / 2} y={n.y + h / 2 + 4} textAnchor="middle"
        fontSize={n.type === 'domain' ? 11 : 9.5} fontWeight={n.type === 'domain' ? 700 : 500}
        fill="var(--ink2)">
        {truncate(n.label, n.type === 'domain' ? 20 : 16)}
      </text>
    </g>
  )
}

export default function AtlasGraphSvg({ resolved, expanded, onSelect, selectedId }) {
  const { nodes, edges, width, height } = useMemo(() => layoutTree(resolved, { expanded }), [resolved, expanded])
  const cells = nodes.filter((n) => n.type === 'cell')
  const boxes = nodes.filter((n) => n.type !== 'cell')

  return (
    <div className="atlas-graph-svg card">
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={Math.min(height, 620)}
        role="img" preserveAspectRatio="xMinYMin meet"
        aria-label="Atlas tree of ML domains, pipeline components, and quantum-versus-classical outcomes. The List view has the same data.">
        {edges.map((e, i) => (
          <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
            stroke={e.kind === 'relation' ? 'var(--accent)' : 'var(--axis)'}
            strokeWidth={e.kind === 'relation' ? 1.5 : 1}
            strokeDasharray={e.kind === 'relation' ? '4 3' : undefined}
            opacity={e.kind === 'relation' ? 0.8 : 0.5} />
        ))}
        {boxes.map((n) => <BoxNode key={n.id} n={n} />)}
        {cells.map((n) => (
          <CellShape key={n.id} n={n} selected={selectedId === n.id} onSelect={onSelect} />
        ))}
      </svg>
    </div>
  )
}
