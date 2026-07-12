// Pure Cytoscape stylesheet builder. Takes a tokenMap of resolved CSS custom
// property values ({ '--q': '#c05fc0', ... }) — resolved at mount by the React
// wrapper via getComputedStyle — and returns a plain array of {selector, style}
// objects. No cytoscape import, no DOM: fully unit-testable.
//
// Four ORTHOGONAL visual channels so no two integrity dimensions collapse:
//   fill  = outcome_class (color bucket; "classical holds / no advantage" is
//           classical-blue at full prominence, never green, never dimmed)
//   border-width = claim_level rank (evidence strength)
//   border-style = replication (dashed = none, solid = any) — a DIFFERENT channel
//                  from claim, so replication can never be read as claim strength
//   shape = kind (head-to-head / quantum-only / suggested / unexplored)

export function toStylesheet(tokenMap = {}) {
  const t = (key, fallback) => tokenMap[key] || fallback
  const outcomeFill = {
    quantum_candidate: t('--q', '#c05fc0'),
    quantum_only: t('--q', '#c05fc0'),
    classical_holds: t('--c', '#3987e5'),
    open: t('--warn', '#e0a52e'),
    suggested: t('--accent', '#8f8af0'),
    unexplored: t('--null', '#8f8ea0'),
  }
  const ink = t('--ink', '#f1f0f5')
  const ink2 = t('--ink2', '#b2b1bc')
  const hair = t('--hair', 'rgba(255,255,255,.10)')
  const axis = t('--axis', '#3a3a42')
  const surface2 = t('--surface2', '#1e1e25')

  const sheet = [
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        color: ink,
        'font-size': 11,
        'text-wrap': 'wrap',
        'text-max-width': 130,
        'text-valign': 'center',
      },
    },
    {
      selector: '.domain',
      style: {
        shape: 'round-rectangle',
        'background-opacity': 0.06,
        'background-color': ink2,
        'border-width': 1,
        'border-color': hair,
        'text-valign': 'top',
        'font-weight': 700,
        'font-size': 12,
        padding: 12,
      },
    },
    {
      selector: '.component',
      style: {
        shape: 'round-rectangle',
        'background-opacity': 0.04,
        'background-color': surface2,
        'border-width': 1,
        'border-color': hair,
        'text-valign': 'top',
        'font-size': 10,
        color: ink2,
        padding: 8,
      },
    },
    {
      selector: '.cell',
      style: {
        width: 128,
        height: 40,
        'border-color': ink,
        // claim strength → border WIDTH (evidence strength channel)
        'border-width': 'mapData(claimRank, 0, 8, 1.5, 6)',
        'text-max-width': 118,
        color: '#ffffff',
        'text-outline-width': 0,
      },
    },
  ]

  for (const [outcome, color] of Object.entries(outcomeFill)) {
    sheet.push({ selector: `.outcome-${outcome}`, style: { 'background-color': color } })
  }
  // unexplored/suggested fills are light — use dark ink for legibility.
  sheet.push({ selector: '.outcome-unexplored, .outcome-open', style: { color: ink } })

  // shape → kind
  sheet.push({ selector: '.kind-head_to_head', style: { shape: 'round-rectangle' } })
  sheet.push({ selector: '.kind-quantum_only', style: { shape: 'hexagon' } })
  sheet.push({ selector: '.kind-suggested', style: { shape: 'diamond' } })
  sheet.push({ selector: '.kind-unexplored', style: { shape: 'ellipse' } })

  // replication → border STYLE (distinct channel from claim width)
  sheet.push({ selector: '.cell[replicationRank = 0]', style: { 'border-style': 'dashed' } })
  sheet.push({ selector: '.cell[replicationRank > 0]', style: { 'border-style': 'solid' } })

  // selection + edges
  sheet.push({ selector: 'node:selected', style: { 'border-color': t('--accent', '#8f8af0'), 'border-width': 5 } })
  sheet.push({
    selector: 'edge',
    style: {
      width: 1.5,
      'line-color': axis,
      'curve-style': 'bezier',
      'target-arrow-shape': 'none',
      opacity: 0.6,
    },
  })
  sheet.push({
    selector: 'edge.relation',
    style: { 'line-style': 'dashed', 'line-color': t('--accent', '#8f8af0'), label: 'data(relation)', 'font-size': 8, color: ink2 },
  })

  return sheet
}

// The token keys the wrapper must resolve from the live theme and pass in.
export const ATLAS_TOKEN_KEYS = Object.freeze([
  '--q', '--c', '--warn', '--accent', '--null', '--ink', '--ink2', '--hair', '--axis', '--surface2',
])
