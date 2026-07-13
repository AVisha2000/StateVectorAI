// Pure force-clustered "research map" layout for the Atlas graph, rendered as
// hand-authored SVG. Determinism contract: no Math.random, no Date, no
// locale-dependent ops; all variation derives from hash01(id); fixed iteration
// counts; input-order traversal everywhere; round2 on every emitted coordinate.
// The output is a pure function of (resolved, expanded, width). O(n²) forces
// are fine to ~150 cells. Framework-free so it runs under node --test.
import { claimRank, replicationRank, OUTCOME_LABELS } from './atlasModel.js'

export const CELL_W = 116
export const CELL_H = 58
export const R_COLLIDE = 68 // half-diagonal of the card (64.85) + margin
export const HULL_PAD = 22
export const HULL_SAMPLES = 8
export const GUTTER = 56
export const SEAL_W = 180
export const SEAL_H = 44
export const MARGIN = 40
export const TICKS = 300
export const SETTLE_ITERS = 40
export const ZOOM_MIN = 0.6
export const ZOOM_MAX = 2.5

// Typed-relation styling shared by renderer + legend so they cannot drift.
// Association-like types are dashed with a chevron; constraint-like types are
// solid with an inhibition bar. All routes use var(--accent) — type is carried
// by dash + terminal only (no color channel collision with cell outcomes).
// The keys mirror docs/ATLAS_ONTOLOGY.yaml's relation vocabulary; unknown
// types default to the associative style.
export const RELATION_STYLE = Object.freeze({
  constrains: { dash: null, marker: 'bar' },
  must_not_be_conflated_with: { dash: null, marker: 'bar' },
})
export function relationStyle(type) {
  return RELATION_STYLE[type] || { dash: '6 4', marker: 'arrow' }
}

export function round2(v) {
  return Math.round(v * 100) / 100
}

// Deterministic string → [0,1) hash (mulberry32-style finalizer).
export function hash01(str) {
  let h = 1779033703 ^ String(str).length
  for (let i = 0; i < String(str).length; i += 1) {
    h = Math.imul(h ^ String(str).charCodeAt(i), 3432918353)
    h = (h << 13) | (h >>> 19)
  }
  h = Math.imul(h ^ (h >>> 16), 2246822507)
  h = Math.imul(h ^ (h >>> 13), 3266489909)
  h ^= h >>> 16
  return (h >>> 0) / 4294967296
}

// Cluster radius for n member cells. The second term is the ring-packing lower
// bound: n centers pairwise ≥ 2*R_COLLIDE apart fit on a ring of radius
// R_COLLIDE/sin(π/n), so hard containment + collision stay jointly feasible.
export function clusterRadius(n) {
  if (n <= 1) return 74
  return Math.max(34 * Math.sqrt(n) + 40, R_COLLIDE / Math.sin(Math.PI / n) + 8)
}

// Domain anchors on an ellipse, arc share proportional to cluster size, in
// input (ontology) order. Closed-form, no iteration. The arc demand uses the
// full hull extent (cells may sit at the cluster boundary and the territory
// outline reaches half a card diagonal + padding beyond their centers), plus a
// 12% factor because chords on the flattened ellipse run shorter than arcs.
const HULL_EXTENT = Math.hypot(CELL_W, CELL_H) / 2 + HULL_PAD
export function anchorPositions(domains, expanded) {
  const list = domains.map((d) => {
    const n = (d.cells || []).length
    const r = expanded.has(d.id) ? clusterRadius(n) : 96
    return { domainId: d.id, r, c: 2 * (r + HULL_EXTENT) + GUTTER }
  })
  const C = list.reduce((a, b) => a + b.c, 0) || 1
  // Ramanujan circumference of an ellipse with Ry = 0.62 Rx reduces to ≈5.16 Rx.
  const Rx = Math.max(330, (C * 1.12) / 5.16)
  const Ry = 0.62 * Rx
  let cum = 0
  return list.map((d) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * (cum + d.c / 2)) / C
    cum += d.c
    return { domainId: d.domainId, x: Rx * Math.cos(angle), y: Ry * Math.sin(angle), r: d.r }
  })
}

// Deterministic force simulation over cell centers (in place). Position-based,
// exponential cooling, nodes and pairs iterated in input order.
export function simulate(cells, anchors, relations, { ticks = TICKS } = {}) {
  const anchorById = new Map(anchors.map((a) => [a.domainId, a]))
  const byId = new Map(cells.map((c) => [c.id, c]))
  // Precompute pair lists once (input order).
  const stagePairs = []
  for (let i = 0; i < cells.length; i += 1) {
    for (let j = i + 1; j < cells.length; j += 1) {
      const a = cells[i]
      const b = cells[j]
      if (a.domainId === b.domainId && a.stage && a.stage === b.stage) stagePairs.push([a, b])
    }
  }
  const relPairs = []
  for (const r of relations || []) {
    const a = byId.get(r.from_cell)
    const b = byId.get(r.to_cell)
    if (a && b && a.domainId !== b.domainId) relPairs.push([a, b])
  }

  const axisStep = (a, b, rest, k) => {
    const dx = b.x - a.x
    const dy = b.y - a.y
    const d = Math.hypot(dx, dy) || 1e-6
    const f = (k * (d - rest)) / 2
    const ux = dx / d
    const uy = dy / d
    a.x += ux * f; a.y += uy * f
    b.x -= ux * f; b.y -= uy * f
  }

  for (let t = 0; t < ticks; t += 1) {
    const alpha = 0.5 * Math.pow(0.985, t)
    // 1. anchor spring
    for (const c of cells) {
      const a = anchorById.get(c.domainId)
      c.x += (a.x - c.x) * 0.06 * alpha
      c.y += (a.y - c.y) * 0.06 * alpha
    }
    // 2. same-domain same-stage springs
    for (const [a, b] of stagePairs) axisStep(a, b, 110, 0.04 * alpha)
    // 3. cross-domain relation springs
    for (const [a, b] of relPairs) axisStep(a, b, 260, 0.008 * alpha)
    // 4. collision — two passes per tick
    for (let pass = 0; pass < 2; pass += 1) collidePass(cells)
    // 5. containment — soft pull for most of the run, hard clamp at the end
    for (const c of cells) {
      const a = anchorById.get(c.domainId)
      const rLim = a.r - 4
      const dx = c.x - a.x
      const dy = c.y - a.y
      const d = Math.hypot(dx, dy)
      if (d > rLim) {
        const scale = t < ticks * 0.8 ? d + (rLim - d) * 0.2 : rLim
        c.x = a.x + (dx / d) * scale
        c.y = a.y + (dy / d) * scale
      }
    }
  }
}

function collidePass(cells) {
  for (let i = 0; i < cells.length; i += 1) {
    for (let j = i + 1; j < cells.length; j += 1) {
      const a = cells[i]
      const b = cells[j]
      const dx = b.x - a.x
      const dy = b.y - a.y
      let d = Math.hypot(dx, dy)
      const min = 2 * R_COLLIDE
      if (d >= min) continue
      let ux
      let uy
      if (d === 0) {
        const ang = 2 * Math.PI * hash01(`${a.id}|${b.id}`)
        ux = Math.cos(ang); uy = Math.sin(ang)
        d = 1e-6
      } else {
        ux = dx / d; uy = dy / d
      }
      const push = (min - d) / 2
      a.x -= ux * push; a.y -= uy * push
      b.x += ux * push; b.y += uy * push
    }
  }
}

// Final hard pass: alternate collision separation and containment clamping
// until both hold. With clusterRadius's packing bound this terminates with
// zero overlaps AND containment (asserted by the unit tests).
export function resolveCollisions(cells, anchors, { iterations = SETTLE_ITERS } = {}) {
  const anchorById = new Map(anchors.map((a) => [a.domainId, a]))
  for (let it = 0; it < iterations; it += 1) {
    collidePass(cells)
    for (const c of cells) {
      const a = anchorById.get(c.domainId)
      const rLim = a.r - 4
      const dx = c.x - a.x
      const dy = c.y - a.y
      const d = Math.hypot(dx, dy)
      if (d > rLim) {
        c.x = a.x + (dx / d) * rLim
        c.y = a.y + (dy / d) * rLim
      }
    }
  }
}

// Andrew monotone chain; handles n<=2 and collinear input.
export function convexHull(points) {
  const pts = [...points].sort((a, b) => a.x - b.x || a.y - b.y)
  if (pts.length <= 2) return pts
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
  const lower = []
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop()
    lower.push(p)
  }
  const upper = []
  for (let i = pts.length - 1; i >= 0; i -= 1) {
    const p = pts[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop()
    upper.push(p)
  }
  lower.pop(); upper.pop()
  const hull = lower.concat(upper)
  return hull.length ? hull : pts.slice(0, 1)
}

// Territory geometry: circle-sample HULL_SAMPLES points around each member
// cell (padding baked into the sample radius so the outline never spikes),
// convex-hull them, then smooth with a closed midpoint-quadratic path.
export function hullGeometry(cellCenters) {
  const rad = Math.hypot(CELL_W, CELL_H) / 2 + HULL_PAD
  const samples = []
  for (const c of cellCenters) {
    for (let j = 0; j < HULL_SAMPLES; j += 1) {
      const ang = (j * 2 * Math.PI) / HULL_SAMPLES
      samples.push({ x: c.x + rad * Math.cos(ang), y: c.y + rad * Math.sin(ang) })
    }
  }
  const hull = convexHull(samples)
  if (hull.length === 0) return null
  if (hull.length === 1) {
    const p = hull[0]
    const d = `M ${round2(p.x - rad)} ${round2(p.y)} A ${rad} ${rad} 0 1 0 ${round2(p.x + rad)} ${round2(p.y)} A ${rad} ${rad} 0 1 0 ${round2(p.x - rad)} ${round2(p.y)} Z`
    return finishHull(d, [{ x: p.x - rad, y: p.y - rad }, { x: p.x + rad, y: p.y + rad }])
  }
  // Closed path through edge midpoints, hull vertices as quadratic controls.
  const mids = hull.map((p, i) => {
    const q = hull[(i + 1) % hull.length]
    return { x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 }
  })
  let d = `M ${round2(mids[0].x)} ${round2(mids[0].y)}`
  for (let i = 1; i <= hull.length; i += 1) {
    const ctrl = hull[i % hull.length]
    const mid = mids[i % hull.length]
    d += ` Q ${round2(ctrl.x)} ${round2(ctrl.y)} ${round2(mid.x)} ${round2(mid.y)}`
  }
  d += ' Z'
  return finishHull(d, hull)
}

function finishHull(d, boundaryPoints) {
  const xs = boundaryPoints.map((p) => p.x)
  const ys = boundaryPoints.map((p) => p.y)
  const bbox = { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) }
  return {
    d,
    points: boundaryPoints.map((p) => ({ x: round2(p.x), y: round2(p.y) })),
    bbox,
    centroidX: round2((bbox.minX + bbox.maxX) / 2),
    minY: round2(bbox.minY),
    maxY: round2(bbox.maxY),
  }
}

// Cubic Bézier point at parameter t.
export function cubicPointAt(p0, c1, c2, p3, t) {
  const u = 1 - t
  const x = u * u * u * p0.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t * t * t * p3.x
  const y = u * u * u * p0.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t * t * t * p3.y
  return { x, y }
}

// Intersection of the ray from a card's center toward `toward` with the card's
// w×h rect boundary, pushed `push` px outward along the ray.
export function rectBoundaryPoint(center, toward, w = CELL_W, h = CELL_H, push = 4) {
  const dx = toward.x - center.x
  const dy = toward.y - center.y
  const d = Math.hypot(dx, dy) || 1e-6
  const ux = dx / d
  const uy = dy / d
  const sx = Math.abs(ux) > 1e-9 ? (w / 2) / Math.abs(ux) : Infinity
  const sy = Math.abs(uy) > 1e-9 ? (h / 2) / Math.abs(uy) : Infinity
  const s = Math.min(sx, sy) + push
  return { x: center.x + ux * s, y: center.y + uy * s }
}

// Curved route between two cells: cubic Bézier bowed away from the content
// center, fanned when several routes share the same unordered pair.
export function routeEdge(a, b, contentCenter, fanIndex = 0) {
  const p0 = rectBoundaryPoint(a, b)
  const p3 = rectBoundaryPoint(b, a)
  const dx = p3.x - p0.x
  const dy = p3.y - p0.y
  const len = Math.hypot(dx, dy) || 1e-6
  const nx = -dy / len
  const ny = dx / len
  const mid = { x: (p0.x + p3.x) / 2, y: (p0.y + p3.y) / 2 }
  const dot = nx * (mid.x - contentCenter.x) + ny * (mid.y - contentCenter.y)
  const side = dot >= 0 ? 1 : -1
  const bow = Math.min(80, Math.max(24, 0.18 * len)) + 18 * fanIndex
  const c1 = { x: p0.x + dx / 3 + nx * bow * side, y: p0.y + dy / 3 + ny * bow * side }
  const c2 = { x: p0.x + (2 * dx) / 3 + nx * bow * side, y: p0.y + (2 * dy) / 3 + ny * bow * side }
  const label = cubicPointAt(p0, c1, c2, p3, 0.5)
  return {
    d: `M ${round2(p0.x)} ${round2(p0.y)} C ${round2(c1.x)} ${round2(c1.y)} ${round2(c2.x)} ${round2(c2.y)} ${round2(p3.x)} ${round2(p3.y)}`,
    x1: round2(p0.x), y1: round2(p0.y), x2: round2(p3.x), y2: round2(p3.y),
    labelX: round2(label.x), labelY: round2(label.y),
  }
}

// Character-count word wrap (no text measurement → deterministic).
export function wrapLabel(label, maxChars = 18, maxLines = 2) {
  const words = String(label || '').split(/\s+/).filter(Boolean)
  const lines = []
  let cur = ''
  for (const w of words) {
    const next = cur ? `${cur} ${w}` : w
    if (next.length <= maxChars) {
      cur = next
    } else {
      if (cur) lines.push(cur)
      cur = w.length > maxChars ? `${w.slice(0, maxChars - 1)}…` : w
      if (lines.length === maxLines - 1) break
    }
  }
  if (cur && lines.length < maxLines) lines.push(cur)
  // Ellipsis when content was truncated.
  const joined = lines.join(' ')
  if (joined.length < String(label || '').trim().replace(/\s+/g, ' ').length && !lines[lines.length - 1].endsWith('…')) {
    const last = lines[lines.length - 1]
    lines[lines.length - 1] = last.length >= maxChars ? `${last.slice(0, maxChars - 1)}…` : `${last}…`
  }
  return lines
}

// Kind → silhouette, centered at local (0,0). `inflate` grows the shape for
// the selection ring so the outcome/claim/replication border is never touched.
export function shapeGeom(kind, w = CELL_W, h = CELL_H, inflate = 0) {
  const W = w + inflate * 2
  const H = h + inflate * 2
  if (kind === 'quantum_only') {
    const cut = 16
    const pts = [
      [-W / 2 + cut, -H / 2], [W / 2 - cut, -H / 2], [W / 2, 0],
      [W / 2 - cut, H / 2], [-W / 2 + cut, H / 2], [-W / 2, 0],
    ]
    return { tag: 'polygon', attrs: { points: pts.map((p) => `${round2(p[0])},${round2(p[1])}`).join(' ') } }
  }
  if (kind === 'suggested') {
    const pts = [[0, -H / 2], [W / 2, 0], [0, H / 2], [-W / 2, 0]]
    return { tag: 'polygon', attrs: { points: pts.map((p) => `${round2(p[0])},${round2(p[1])}`).join(' ') } }
  }
  if (kind === 'unexplored') {
    return { tag: 'ellipse', attrs: { cx: 0, cy: 0, rx: round2(W / 2), ry: round2(H / 2) } }
  }
  return { tag: 'rect', attrs: { x: round2(-W / 2), y: round2(-H / 2), width: round2(W), height: round2(H), rx: 10 } }
}

// ---- pan/zoom transform math (pure numbers in/out) -------------------------

export function clampTransform(t, contentBounds, viewport) {
  const { width: vw, height: vh } = viewport
  const minOverlap = 120
  // Scaled content bbox in view units.
  const x1 = contentBounds.minX * t.k + t.tx
  const x2 = contentBounds.maxX * t.k + t.tx
  const y1 = contentBounds.minY * t.k + t.ty
  const y2 = contentBounds.maxY * t.k + t.ty
  let { tx, ty } = t
  if (x2 < minOverlap) tx += minOverlap - x2
  if (x1 > vw - minOverlap) tx -= x1 - (vw - minOverlap)
  if (y2 < minOverlap) ty += minOverlap - y2
  if (y1 > vh - minOverlap) ty -= y1 - (vh - minOverlap)
  return { k: t.k, tx, ty }
}

export function zoomAtPoint(t, ux, uy, factor, contentBounds, viewport) {
  const k2 = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, t.k * factor))
  const tx = ux - ((ux - t.tx) * k2) / t.k
  const ty = uy - ((uy - t.ty) * k2) / t.k
  const next = { k: k2, tx, ty }
  return contentBounds && viewport ? clampTransform(next, contentBounds, viewport) : next
}

export function focusTransform(node, viewport, t, contentBounds) {
  const next = { k: t.k, tx: viewport.width / 2 - node.x * t.k, ty: viewport.height / 2 - node.y * t.k }
  return contentBounds ? clampTransform(next, contentBounds, viewport) : next
}

// Screen px → user (viewBox) units for a scaled SVG.
export function toUser(clientX, clientY, svgRect, viewW, viewH) {
  return {
    x: (clientX - svgRect.left) * (viewW / svgRect.width),
    y: (clientY - svgRect.top) * (viewH / svgRect.height),
  }
}

// ---- main entry -------------------------------------------------------------

export function layoutMap(resolved, { expanded } = {}) {
  const domains = resolved?.domains || []
  const exp = expanded instanceof Set ? expanded : new Set(domains.map((d) => d.id))
  const anchors = anchorPositions(domains, exp)
  const anchorById = new Map(anchors.map((a) => [a.domainId, a]))

  // Seed cells with a phyllotaxis spiral inside their cluster (no RNG).
  const cells = []
  for (const d of domains) {
    if (!exp.has(d.id)) continue
    const a = anchorById.get(d.id)
    const members = d.cells || []
    members.forEach((c, k) => {
      const theta = k * 2.39996113 + 2 * Math.PI * hash01(d.id)
      const rho = Math.min(46 * Math.sqrt(k + 0.5), 0.9 * a.r)
      let x = a.x + rho * Math.cos(theta)
      let y = a.y + rho * Math.sin(theta)
      // coincident-seed guard
      for (const prev of cells) {
        if (Math.hypot(prev.x - x, prev.y - y) < 0.5) {
          x += hash01(c.id) * 4 + 1
          y += hash01(`${c.id}y`) * 4 + 1
        }
      }
      cells.push({
        id: c.id, domainId: d.id, x, y,
        stage: c.pipeline_stage || null,
        kind: c.kind, outcome: c.outcome_class,
        claimRank: claimRank(c.claim_level), replicationRank: replicationRank(c.replication_status),
        label: c.label, cell: c,
      })
    })
  }

  simulate(cells, anchors, resolved?.relations || [], { ticks: TICKS })
  resolveCollisions(cells, anchors, { iterations: SETTLE_ITERS })

  // Translate so the padded content bbox min corner sits at (MARGIN, MARGIN).
  const hullRad = Math.hypot(CELL_W, CELL_H) / 2 + HULL_PAD
  const sealPad = Math.max(SEAL_W, SEAL_H) / 2 + 8
  const boxes = []
  for (const c of cells) boxes.push({ x: c.x, y: c.y, pad: hullRad })
  for (const a of anchors) {
    if (!exp.has(a.domainId)) boxes.push({ x: a.x, y: a.y, pad: sealPad })
  }
  if (!boxes.length) boxes.push({ x: 0, y: 0, pad: hullRad })
  const minX = Math.min(...boxes.map((b) => b.x - b.pad))
  const maxX = Math.max(...boxes.map((b) => b.x + b.pad))
  const minY = Math.min(...boxes.map((b) => b.y - b.pad))
  const maxY = Math.max(...boxes.map((b) => b.y + b.pad))
  const shiftX = MARGIN - minX
  const shiftY = MARGIN - minY
  for (const c of cells) { c.x += shiftX; c.y += shiftY }
  const shiftedAnchors = anchors.map((a) => ({ ...a, x: a.x + shiftX, y: a.y + shiftY }))
  const width = round2(maxX - minX + 2 * MARGIN)
  const height = round2(Math.max(maxY - minY + 2 * MARGIN, 240))

  // Hulls + labels (expanded), seals (collapsed).
  const hulls = []
  const seals = []
  const placedLabels = []
  for (const d of domains) {
    const a = shiftedAnchors.find((x) => x.domainId === d.id)
    if (!exp.has(d.id)) {
      seals.push({ domainId: d.id, label: d.label, count: (d.cells || []).length, x: round2(a.x), y: round2(a.y) })
      continue
    }
    const members = cells.filter((c) => c.domainId === d.id)
    if (!members.length) continue
    const geom = hullGeometry(members.map((c) => ({ x: c.x, y: c.y })))
    if (!geom) continue
    let labelY = geom.minY - 12
    let labelBelow = false
    for (const pl of placedLabels) {
      if (Math.abs(pl.x - geom.centroidX) < 160 && Math.abs(pl.y - labelY) < 24) {
        labelY = geom.maxY + 16
        labelBelow = true
        break
      }
    }
    placedLabels.push({ x: geom.centroidX, y: labelY })
    hulls.push({
      domainId: d.id, label: d.label, count: members.length,
      d: geom.d, points: geom.points,
      bbox: geom.bbox, labelX: geom.centroidX, labelY: round2(labelY), labelBelow,
    })
  }

  // Typed relation routes between present cells only.
  const byId = new Map(cells.map((c) => [c.id, c]))
  const contentCenter = { x: width / 2, y: height / 2 }
  const pairCount = new Map()
  const edges = []
  for (const r of resolved?.relations || []) {
    const a = byId.get(r.from_cell)
    const b = byId.get(r.to_cell)
    if (!a || !b) continue
    const key = [r.from_cell, r.to_cell].sort().join('~')
    const fan = pairCount.get(key) || 0
    pairCount.set(key, fan + 1)
    const style = relationStyle(r.type)
    edges.push({
      from: r.from_cell, to: r.to_cell, kind: 'relation', relation: r.type,
      dash: style.dash, marker: style.marker,
      ...routeEdge(a, b, contentCenter, fan),
    })
  }

  // Emit nodes with rounded centers + wrapped labels + a lib-computed aria label.
  const nodes = cells.map((c) => ({
    id: c.id, type: 'cell',
    x: round2(c.x), y: round2(c.y), w: CELL_W, h: CELL_H,
    kind: c.kind, outcome: c.outcome, claimRank: c.claimRank, replicationRank: c.replicationRank,
    stage: c.stage, domainId: c.domainId, label: c.label,
    lines: wrapLabel(c.label),
    ariaLabel: `${c.label} — ${OUTCOME_LABELS[c.outcome] || c.outcome}; claim: ${c.cell.claim_level || 'untested'}; replication: ${c.cell.replication_status || 'none'}; ${c.kind} cell`,
    cell: c.cell,
  }))

  return {
    nodes,
    hulls,
    seals,
    edges,
    width,
    height,
    bounds: { minX: 0, minY: 0, maxX: width, maxY: height },
  }
}
