import assert from 'node:assert/strict'
import test from 'node:test'
import {
  layoutMap, hash01, clusterRadius, convexHull, hullGeometry, wrapLabel, shapeGeom,
  zoomAtPoint, clampTransform, focusTransform, toUser, round2, relationStyle,
  R_COLLIDE, ZOOM_MIN, ZOOM_MAX, CELL_W, CELL_H,
} from './atlasMapLayout.js'
import { resolveOntology } from './atlasModel.js'
import { ATLAS_SEED } from './atlasOntology.seed.js'

const resolved = resolveOntology(ATLAS_SEED, [], null)
const allExpanded = new Set(resolved.domains.map((d) => d.id))

// The graph consumes the same regrouped shape Atlas.jsx passes (flat cells per domain).
function asGraphInput(res) {
  return res
}

test('layoutMap is deterministic (repeat calls deep-equal, no hidden state)', () => {
  const a = layoutMap(asGraphInput(resolved), { expanded: allExpanded })
  const b = layoutMap(asGraphInput(resolved), { expanded: allExpanded })
  hash01('interleaved unrelated work')
  clusterRadius(5)
  const c = layoutMap(asGraphInput(resolved), { expanded: allExpanded })
  assert.deepStrictEqual(a, b)
  assert.deepStrictEqual(a, c)
})

test('every emitted coordinate is rounded to 2 decimals and finite', () => {
  const { nodes, hulls, seals, edges, width, height } = layoutMap(resolved, { expanded: allExpanded })
  const check = (v) => { assert.ok(Number.isFinite(v)); assert.equal(v, round2(v)) }
  for (const n of nodes) { check(n.x); check(n.y) }
  for (const h of hulls) { check(h.labelX); check(h.labelY) }
  for (const s of seals) { check(s.x); check(s.y) }
  for (const e of edges) { check(e.x1); check(e.y1); check(e.x2); check(e.y2); check(e.labelX); check(e.labelY) }
  check(width); check(height)
})

test('zero card overlap: every cell pair at least 2*R_COLLIDE apart', () => {
  const { nodes } = layoutMap(resolved, { expanded: allExpanded })
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const d = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y)
      assert.ok(d >= 2 * R_COLLIDE - 0.01, `${nodes[i].id} vs ${nodes[j].id}: ${d}`)
    }
  }
})

test('containment: every cell sits inside its own domain hull; hull bboxes pairwise disjoint', () => {
  const { nodes, hulls } = layoutMap(resolved, { expanded: allExpanded })
  const hullByDomain = new Map(hulls.map((h) => [h.domainId, h]))
  const inPoly = (pt, poly) => {
    let inside = false
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
      const a = poly[i]; const b = poly[j]
      if ((a.y > pt.y) !== (b.y > pt.y) && pt.x < ((b.x - a.x) * (pt.y - a.y)) / (b.y - a.y) + a.x) inside = !inside
    }
    return inside
  }
  for (const n of nodes) {
    const h = hullByDomain.get(n.domainId)
    assert.ok(h, `hull for ${n.domainId}`)
    for (const [cx, cy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const corner = { x: n.x + (cx * CELL_W) / 2, y: n.y + (cy * CELL_H) / 2 }
      assert.ok(inPoly(corner, h.points), `${n.id} corner outside ${n.domainId} hull`)
    }
  }
  // Territory disjointness: the smoothed outline is inscribed in the convex
  // hull, so convex-convex separation (SAT) proves the drawn shapes never
  // intersect. (Plain bbox tests false-positive on diagonal neighbours.)
  const satDisjoint = (P, Q) => {
    const axes = (poly) => poly.map((p, i) => {
      const q = poly[(i + 1) % poly.length]
      return { x: -(q.y - p.y), y: q.x - p.x }
    })
    for (const ax of [...axes(P), ...axes(Q)]) {
      const proj = (poly) => {
        let lo = Infinity; let hi = -Infinity
        for (const p of poly) {
          const v = p.x * ax.x + p.y * ax.y
          lo = Math.min(lo, v); hi = Math.max(hi, v)
        }
        return [lo, hi]
      }
      const [aLo, aHi] = proj(P)
      const [bLo, bHi] = proj(Q)
      if (aHi < bLo || bHi < aLo) return true // separating axis found
    }
    return false
  }
  for (let i = 0; i < hulls.length; i += 1) {
    for (let j = i + 1; j < hulls.length; j += 1) {
      assert.ok(
        satDisjoint(hulls[i].points, hulls[j].points),
        `hulls ${hulls[i].domainId} / ${hulls[j].domainId} intersect`,
      )
    }
  }
})

test('clusterRadius always satisfies the ring-packing bound', () => {
  for (let n = 2; n <= 8; n += 1) {
    assert.ok(clusterRadius(n) >= R_COLLIDE / Math.sin(Math.PI / n), `n=${n}`)
  }
  assert.ok(clusterRadius(1) > 0)
})

test('collapsed domain → one inert seal, no member cells, no hull; relations only between present cells', () => {
  const first = resolved.domains[0]
  const expanded = new Set(allExpanded)
  expanded.delete(first.id)
  const { nodes, hulls, seals, edges } = layoutMap(resolved, { expanded })
  assert.ok(!nodes.some((n) => n.domainId === first.id))
  assert.ok(!hulls.some((h) => h.domainId === first.id))
  assert.equal(seals.length, 1)
  assert.equal(seals[0].domainId, first.id)
  assert.equal(seals[0].count, first.cells.length)
  const present = new Set(nodes.map((n) => n.id))
  assert.ok(edges.every((e) => present.has(e.from) && present.has(e.to)))
})

test('nodes carry all four channels, stage, wrapped lines, and a spoken aria label', () => {
  const { nodes } = layoutMap(resolved, { expanded: allExpanded })
  assert.equal(nodes.length, 19)
  for (const n of nodes) {
    assert.ok('outcome' in n && 'claimRank' in n && 'replicationRank' in n && 'kind' in n && 'stage' in n)
    assert.ok(Array.isArray(n.lines) && n.lines.length >= 1 && n.lines.length <= 2)
    assert.match(n.ariaLabel, /claim: /)
    assert.match(n.ariaLabel, /replication: /)
  }
})

test('routes carry typed dash/marker and endpoints on the card boundary', () => {
  const { nodes, edges } = layoutMap(resolved, { expanded: allExpanded })
  assert.ok(edges.length >= 1)
  const byId = new Map(nodes.map((n) => [n.id, n]))
  for (const e of edges) {
    assert.ok(e.relation)
    assert.ok(['arrow', 'bar'].includes(e.marker))
    const src = byId.get(e.from)
    // endpoint sits just outside the source card boundary (pushed 4px along the ray)
    const ddx = Math.abs(e.x1 - src.x)
    const ddy = Math.abs(e.y1 - src.y)
    const onBoundary = Math.abs(ddx - CELL_W / 2) < 6 || Math.abs(ddy - CELL_H / 2) < 6
    assert.ok(onBoundary, `${e.from}→${e.to} start not near boundary (dx=${ddx}, dy=${ddy})`)
  }
  assert.deepEqual(relationStyle('nonexistent_type'), { dash: '6 4', marker: 'arrow' })
})

test('wrapLabel: ≤2 lines, ≤19 chars, ellipsis on truncation, short labels untouched', () => {
  assert.deepEqual(wrapLabel('short'), ['short'])
  const long = wrapLabel('Entanglement-assisted distributed and communication-limited learning')
  assert.equal(long.length, 2)
  assert.ok(long.every((l) => l.length <= 19))
  assert.ok(long[1].endsWith('…'))
  const swaps = wrapLabel('Variational quantum embedding, attention, FFN, and full-block swaps')
  assert.equal(swaps.length, 2)
  assert.ok(swaps.every((l) => l.length > 0))
})

test('hull geometry degrades cleanly for 1-cell and collinear 2-cell domains', () => {
  const one = hullGeometry([{ x: 100, y: 100 }])
  assert.ok(one.d.length > 0 && Number.isFinite(one.minY))
  const two = hullGeometry([{ x: 0, y: 0 }, { x: 200, y: 0 }])
  assert.ok(two.d.startsWith('M') && two.d.endsWith('Z'))
  assert.ok(two.points.length >= 3) // circle samples make even collinear input 2-D
  const hull = convexHull([{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }])
  assert.ok(hull.length >= 1) // collinear input never throws
})

test('shapeGeom: silhouettes per kind, centered at origin, inflatable for the selection ring', () => {
  assert.equal(shapeGeom('head_to_head').tag, 'rect')
  assert.equal(shapeGeom('quantum_only').tag, 'polygon')
  assert.equal(shapeGeom('suggested').tag, 'polygon')
  assert.equal(shapeGeom('unexplored').tag, 'ellipse')
  const base = shapeGeom('head_to_head')
  const ring = shapeGeom('head_to_head', CELL_W, CELL_H, 4)
  assert.ok(ring.attrs.width > base.attrs.width)
})

test('transform math: zoom clamps and keeps the anchor invariant; clamp keeps overlap; focus centers', () => {
  const bounds = { minX: 0, minY: 0, maxX: 1000, maxY: 800 }
  const view = { width: 1000, height: 800 }
  const t0 = { k: 1, tx: 0, ty: 0 }
  const z = zoomAtPoint(t0, 400, 300, 1.25, bounds, view)
  // anchor point invariant: user point (400,300) maps to the same view point
  const before = { x: 400 * t0.k + t0.tx, y: 300 * t0.k + t0.ty }
  const after = { x: 400 * z.k + z.tx, y: 300 * z.k + z.ty }
  assert.ok(Math.abs(before.x - after.x) < 1e-6 && Math.abs(before.y - after.y) < 1e-6)
  // clamping to limits
  let t = t0
  for (let i = 0; i < 20; i += 1) t = zoomAtPoint(t, 400, 300, 1.5, bounds, view)
  assert.ok(t.k <= ZOOM_MAX + 1e-9)
  for (let i = 0; i < 40; i += 1) t = zoomAtPoint(t, 400, 300, 0.5, bounds, view)
  assert.ok(t.k >= ZOOM_MIN - 1e-9)
  // pan clamp: content can never fully leave the viewport
  const far = clampTransform({ k: 1, tx: 5000, ty: 5000 }, bounds, view)
  assert.ok(far.tx < 5000 && far.ty < 5000)
  // focus centers the node at unchanged k
  const f = focusTransform({ x: 500, y: 400 }, view, { k: 1, tx: -999, ty: -999 }, bounds)
  assert.equal(f.k, 1)
  assert.ok(Math.abs(500 * f.k + f.tx - view.width / 2) < 121) // within clamp tolerance
  // toUser round-trips a known rect
  const u = toUser(150, 100, { left: 50, top: 0, width: 500, height: 400 }, 1000, 800)
  assert.deepEqual(u, { x: 200, y: 200 })
})
