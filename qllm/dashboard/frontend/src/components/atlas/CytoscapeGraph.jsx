import { useEffect, useRef } from 'react'
import cytoscape from 'cytoscape'
import { toElementArray } from '../../lib/atlasGraph.js'
import { toStylesheet, ATLAS_TOKEN_KEYS } from '../../lib/atlasStyle.js'

// The ONLY module that imports cytoscape. Kept out of the pure test path (no
// *.test.js imports it) and lazy-loaded by Atlas.jsx so the heavy lib lands in a
// route-split chunk. Element/style computation is done by pure libs upstream;
// this wrapper owns only the imperative cytoscape lifecycle.

function readTokens() {
  const cs = getComputedStyle(document.documentElement)
  const map = {}
  for (const k of ATLAS_TOKEN_KEYS) map[k] = cs.getPropertyValue(k).trim()
  return map
}

// breadthfirst is deterministic and edge-driven — the ontology emits explicit
// domain→component→cell hierarchy edges, so this never diverges the way a
// force layout does on a mostly-edgeless graph.
const LAYOUT = { name: 'breadthfirst', directed: true, padding: 24, spacingFactor: 1.15, fit: true }

function relayout(cy) {
  if (!cy || cy.destroyed()) return
  cy.resize()
  const l = cy.layout(LAYOUT)
  l.one('layoutstop', () => { if (!cy.destroyed()) cy.fit(undefined, 28) })
  l.run()
}

export default function CytoscapeGraph({ resolved, expanded, onSelect }) {
  const containerRef = useRef(null)
  const cyRef = useRef(null)

  // Mount once.
  useEffect(() => {
    const container = containerRef.current
    if (!container) return undefined
    const cy = cytoscape({
      container,
      elements: toElementArray(resolved, { expanded }),
      style: toStylesheet(readTokens()),
      wheelSensitivity: 0.2,
    })
    cy.on('tap', 'node[type = "cell"]', (e) => onSelect?.(e.target.id()))
    cyRef.current = cy

    // Lay out only once the container actually has size (handles the lazy /
    // flex 0-size-at-init case where cytoscape would otherwise render blank).
    let laidOut = false
    const tryLayout = () => {
      if (laidOut || cy.destroyed()) return
      if (container.clientWidth > 0 && container.clientHeight > 0) {
        laidOut = true
        relayout(cy)
      }
    }
    const ro = new ResizeObserver(tryLayout)
    ro.observe(container)
    tryLayout()

    const themeObserver = new MutationObserver(() => { if (!cy.destroyed()) cy.style(toStylesheet(readTokens())) })
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

    return () => {
      ro.disconnect()
      themeObserver.disconnect()
      cy.destroy()
      cyRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Update elements + relayout when the resolved model or expansion changes.
  useEffect(() => {
    const cy = cyRef.current
    if (!cy || cy.destroyed()) return
    cy.json({ elements: toElementArray(resolved, { expanded }) })
    relayout(cy)
  }, [resolved, expanded])

  return (
    <div
      ref={containerRef}
      className="atlas-graph"
      role="img"
      aria-label="Atlas graph of ML domains, pipeline components, and quantum-versus-classical outcomes. A list view with the same data is available via the List toggle."
    />
  )
}
