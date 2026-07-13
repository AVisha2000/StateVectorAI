import { useEffect, useMemo, useRef, useState } from 'react'
import {
  layoutMap, shapeGeom, zoomAtPoint, clampTransform, focusTransform, toUser,
  CELL_W, CELL_H, SEAL_W, SEAL_H,
} from '../../lib/atlasMapLayout.js'

// The Atlas "research map": hand-authored SVG over a deterministic
// force-clustered layout. Domains render as smoothed territory hulls on a
// dotted survey grid, cells as shaped cards, typed relations as curved accent
// routes. Encoding channels are unchanged from the legend: fill = outcome
// (soft token) + stroke = outcome (full token), stroke-width = claim rank,
// dashed stroke = no replication, silhouette = cell kind. Selection draws a
// separate ring so it never overwrites the claim/replication border. Emphasis
// is additive-only: nothing renders below its resting state because of hover
// or selection — null outcomes stay as prominent as positive ones.
const VIEW_MAX_H = 620

function CellNode({ n, selected, hovered, onSelect, onHover }) {
  const face = shapeGeom(n.kind)
  const ring = shapeGeom(n.kind, CELL_W, CELL_H, 4)
  const Face = face.tag
  const Ring = ring.tag
  const strokeWidth = 1 + n.claimRank * 0.5
  const dash = n.replicationRank === 0 ? '5 3' : undefined
  const lineOffset = n.stage ? -6 : 0
  return (
    <g
      className={`atlas-cell atlas-node-oc-${n.outcome}${hovered ? ' hovered' : ''}`}
      role="button"
      tabIndex={0}
      aria-label={n.ariaLabel}
      aria-pressed={selected || undefined}
      data-cell-id={n.id}
      style={{ transform: `translate(${n.x}px, ${n.y}px)` }}
      onClick={() => onSelect?.(n.id)}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect?.(n.id) } }}
      onPointerEnter={() => onHover?.(n.id)}
      onPointerLeave={() => onHover?.(null)}
    >
      <g className="atlas-cell-body">
        <Face className="atlas-cell-back" {...face.attrs} />
        <Face
          className="atlas-cell-face"
          {...face.attrs}
          strokeWidth={strokeWidth}
          strokeDasharray={dash}
        />
        {selected ? <Ring className="atlas-sel-ring" {...ring.attrs} /> : null}
        {n.lines.map((line, i) => (
          <text
            key={i}
            className="atlas-node-title"
            x={0}
            y={lineOffset + (i - (n.lines.length - 1) / 2) * 12}
            textAnchor="middle"
            dominantBaseline="central"
            style={{ pointerEvents: 'none' }}
          >
            {line}
          </text>
        ))}
        {n.stage ? (
          <text className="atlas-node-stage" x={0} y={CELL_H / 2 - 10} textAnchor="middle" style={{ pointerEvents: 'none' }}>
            {n.stage}
          </text>
        ) : null}
        <title>{n.label}</title>
      </g>
    </g>
  )
}

export default function AtlasGraphSvg({ resolved, expanded, onSelect, selectedId }) {
  const layout = useMemo(() => layoutMap(resolved, { expanded }), [resolved, expanded])
  const { nodes, hulls, seals, edges, width, height, bounds } = layout

  const [t, setT] = useState({ k: 1, tx: 0, ty: 0 })
  const [hoveredCell, setHoveredCell] = useState(null)
  const [hoveredRoute, setHoveredRoute] = useState(null)
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef(null)
  const dragRef = useRef(null)
  const viewport = { width, height }

  // Cursor-anchored zoom on ctrl/cmd + wheel only (never hijack page scroll).
  // Attached manually so preventDefault works regardless of passive defaults.
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return undefined
    const onWheel = (e) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      const f = Math.min(1.25, Math.max(0.8, Math.exp(-e.deltaY * 0.0015)))
      const rect = svg.getBoundingClientRect()
      const u = toUser(e.clientX, e.clientY, rect, width, height)
      setT((prev) => zoomAtPoint(prev, u.x, u.y, f, bounds, { width, height }))
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => svg.removeEventListener('wheel', onWheel)
  }, [width, height, bounds])

  // Deep-link focus: when the selected cell is outside the visible user-rect,
  // glide the viewport to center it (same zoom level; end state deterministic).
  useEffect(() => {
    if (!selectedId) return
    const node = nodes.find((n) => n.id === selectedId)
    if (!node) return
    setT((prev) => {
      const vx = (node.x * prev.k + prev.tx) / prev.k
      const visX = -prev.tx / prev.k
      const visW = width / prev.k
      const visY = -prev.ty / prev.k
      const visH = height / prev.k
      const inside = node.x >= visX && node.x <= visX + visW && node.y >= visY && node.y <= visY + visH && vx != null
      if (inside) return prev
      return focusTransform(node, { width, height }, prev, bounds)
    })
  }, [selectedId, nodes, width, height, bounds])

  const startDrag = (e) => {
    const svg = svgRef.current
    if (!svg) return
    e.target.setPointerCapture?.(e.pointerId)
    dragRef.current = { x: e.clientX, y: e.clientY }
    setDragging(true)
  }
  const moveDrag = (e) => {
    if (!dragRef.current) return
    const svg = svgRef.current
    const rect = svg.getBoundingClientRect()
    const sx = width / rect.width
    const sy = height / rect.height
    const dx = (e.clientX - dragRef.current.x) * sx
    const dy = (e.clientY - dragRef.current.y) * sy
    dragRef.current = { x: e.clientX, y: e.clientY }
    setT((prev) => clampTransform({ k: prev.k, tx: prev.tx + dx, ty: prev.ty + dy }, bounds, { width, height }))
  }
  const endDrag = () => { dragRef.current = null; setDragging(false) }

  const zoomBy = (f) => setT((prev) => zoomAtPoint(prev, width / 2, height / 2, f, bounds, viewport))
  const reset = () => setT({ k: 1, tx: 0, ty: 0 })

  const onFrameKeyDown = (e) => {
    if (e.target !== e.currentTarget) return // never swallow Enter/Space on cells
    const pan = 40
    if (e.key === 'ArrowLeft') { setT((p) => clampTransform({ ...p, tx: p.tx + pan }, bounds, viewport)); e.preventDefault() }
    else if (e.key === 'ArrowRight') { setT((p) => clampTransform({ ...p, tx: p.tx - pan }, bounds, viewport)); e.preventDefault() }
    else if (e.key === 'ArrowUp') { setT((p) => clampTransform({ ...p, ty: p.ty + pan }, bounds, viewport)); e.preventDefault() }
    else if (e.key === 'ArrowDown') { setT((p) => clampTransform({ ...p, ty: p.ty - pan }, bounds, viewport)); e.preventDefault() }
    else if (e.key === '+' || e.key === '=') { zoomBy(1.25); e.preventDefault() }
    else if (e.key === '-') { zoomBy(0.8); e.preventDefault() }
    else if (e.key === '0') { reset(); e.preventDefault() }
  }

  // Additive-only emphasis: routes incident to the hovered/selected cell (or
  // directly hovered) get .hot; everything else keeps its resting style.
  const isHot = (e, i) => i === hoveredRoute
    || (hoveredCell && (e.from === hoveredCell || e.to === hoveredCell))
    || (selectedId && (e.from === selectedId || e.to === selectedId))

  const mapLabel = 'Research map of ML domains and quantum-versus-classical outcomes. Arrow keys pan, plus and minus zoom, zero resets. The List view has the same data.'

  return (
    <div className="atlas-graph-svg card">
      <div className="atlas-map-controls">
        <button className="btn sm atlas-zoom-in" type="button" aria-label="Zoom in" onClick={() => zoomBy(1.25)}>+</button>
        <button className="btn sm atlas-zoom-out" type="button" aria-label="Zoom out" onClick={() => zoomBy(0.8)}>−</button>
        <button className="btn sm atlas-zoom-reset" type="button" aria-label="Reset view" onClick={reset}>⤢</button>
      </div>
      <div
        className="atlas-map-frame"
        tabIndex={0}
        role="group"
        aria-label={mapLabel}
        data-zoom={t.k.toFixed(2)}
        onKeyDown={onFrameKeyDown}
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          height={Math.min(height, VIEW_MAX_H)}
          preserveAspectRatio="xMidYMid meet"
          role="group"
          aria-label={mapLabel}
        >
          <defs>
            <pattern id="atlas-dots" width="24" height="24" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="1" fill="var(--axis)" opacity="0.35" />
            </pattern>
            <marker id="atlas-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 1 1 L 7 4 L 1 7" fill="none" stroke="var(--accent)" strokeWidth="1.4" />
            </marker>
            <marker id="atlas-bar" viewBox="0 0 4 12" refX="2" refY="6" markerWidth="4" markerHeight="12" orient="auto">
              <rect x="1" y="1" width="2" height="10" fill="var(--accent)" />
            </marker>
          </defs>
          <g
            className={`atlas-map-viewport${dragging ? ' dragging' : ''}`}
            style={{ transform: `translate(${t.tx}px, ${t.ty}px) scale(${t.k})` }}
          >
            <rect
              className="atlas-graticule"
              x={0} y={0} width={width} height={height}
              fill="url(#atlas-dots)"
              onPointerDown={startDrag}
              onPointerMove={moveDrag}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              onDoubleClick={reset}
            />
            {hulls.map((h) => (
              <g key={h.domainId}>
                <path
                  className={`atlas-hull${h.domainId === nodes.find((n) => n.id === selectedId)?.domainId ? ' sel' : ''}`}
                  data-domain-id={h.domainId}
                  d={h.d}
                />
                <text className="atlas-hull-label" x={h.labelX} y={h.labelY} textAnchor="middle">
                  {`${h.label} · ${h.count}`}
                </text>
              </g>
            ))}
            {seals.map((s) => (
              <g key={s.domainId} className="atlas-seal">
                <rect x={s.x - SEAL_W / 2} y={s.y - SEAL_H / 2} width={SEAL_W} height={SEAL_H} rx={10} />
                <text x={s.x} y={s.y} textAnchor="middle" dominantBaseline="central">
                  {`${s.label} · ${s.count}`}
                </text>
                <title>{`${s.label} (collapsed) — ${s.count} cells`}</title>
              </g>
            ))}
            {edges.map((e, i) => (
              <g key={`${e.from}-${e.to}-${i}`}>
                <path
                  className={`atlas-route${isHot(e, i) ? ' hot' : ''}`}
                  data-relation={e.relation}
                  d={e.d}
                  strokeDasharray={e.dash || undefined}
                  markerEnd={`url(#atlas-${e.marker})`}
                />
                <path
                  className="atlas-route-hit"
                  d={e.d}
                  onPointerEnter={() => setHoveredRoute(i)}
                  onPointerLeave={() => setHoveredRoute(null)}
                />
              </g>
            ))}
            <g className="atlas-node-layer">
              {nodes.map((n) => (
                <CellNode
                  key={n.id}
                  n={n}
                  selected={selectedId === n.id}
                  hovered={hoveredCell === n.id}
                  onSelect={onSelect}
                  onHover={setHoveredCell}
                />
              ))}
            </g>
            {edges.map((e, i) => (
              isHot(e, i) ? (
                <g key={`label-${i}`} className="atlas-route-label" style={{ pointerEvents: 'none' }}>
                  <rect x={e.labelX - 34} y={e.labelY - 9} width={68} height={16} rx={7} />
                  <text x={e.labelX} y={e.labelY} textAnchor="middle" dominantBaseline="central">{e.relation}</text>
                </g>
              ) : null
            ))}
          </g>
        </svg>
      </div>
    </div>
  )
}
