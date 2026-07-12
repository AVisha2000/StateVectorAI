import { useMemo } from 'react'
import { layoutModelGraph, nodeResourceHint, NODE_W, NODE_H } from '../lib/modelGraphLayout.js'

// Hand-authored SVG DAG of a run's architecture. Quantum vs classical blocks are
// the whole point, so they get the validated magenta/blue tokens; input/output
// are neutral. Renders reliably everywhere and is theme-aware via CSS vars.
const KIND = {
  quantum: { fill: 'var(--q-soft)', stroke: 'var(--q)', text: 'var(--q)' },
  classical: { fill: 'var(--c-soft)', stroke: 'var(--c)', text: 'var(--c)' },
  input: { fill: 'var(--surface2)', stroke: 'var(--hair)', text: 'var(--ink2)' },
  output: { fill: 'var(--surface2)', stroke: 'var(--hair)', text: 'var(--ink2)' },
}

function truncate(s, n = 20) {
  const str = String(s || '')
  return str.length > n ? `${str.slice(0, n - 1)}…` : str
}

export default function ModelGraphSvg({ graph }) {
  const { nodes, edges, width, height } = useMemo(() => layoutModelGraph(graph), [graph])
  if (nodes.length === 0) {
    return <div className="state" style={{ padding: '22px 8px' }}>No model graph for this run.</div>
  }
  return (
    <div className="model-graph card">
      <svg viewBox={`0 0 ${width} ${Math.max(height, 80)}`} width="100%" height={Math.min(height, 420)}
        role="img" preserveAspectRatio="xMinYMin meet"
        aria-label="Model architecture graph: input, classical and quantum blocks, and output, left to right.">
        <defs>
          <marker id="mg-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L8,4 L0,8 z" fill="var(--axis)" />
          </marker>
        </defs>
        {edges.map((e, i) => (
          <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
            stroke="var(--axis)" strokeWidth={1.5} opacity={0.7} markerEnd="url(#mg-arrow)" />
        ))}
        {nodes.map((n) => {
          const c = KIND[n.kind] || KIND.classical
          const hint = nodeResourceHint(n)
          return (
            <g key={n.id}>
              <rect x={n.x} y={n.y} width={NODE_W} height={NODE_H} rx={9}
                fill={c.fill} stroke={c.stroke} strokeWidth={1.4} />
              <text x={n.x + NODE_W / 2} y={n.y + (hint ? 18 : NODE_H / 2 + 4)} textAnchor="middle"
                fontSize="11" fontWeight="600" fill={c.text}>{truncate(n.label)}</text>
              {hint ? (
                <text x={n.x + NODE_W / 2} y={n.y + 33} textAnchor="middle" fontSize="9" fill="var(--muted)">{hint}</text>
              ) : null}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
