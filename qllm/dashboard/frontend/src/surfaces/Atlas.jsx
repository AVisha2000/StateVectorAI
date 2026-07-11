import { PageHeader, Scaffold } from '../lib/ui.jsx'

export default function Atlas() {
  return (
    <>
      <PageHeader
        title="Atlas"
        sub="The explorable map of classical-vs-quantum ML — the public, read-only face. Null results render with equal prominence to positive ones."
      />
      <Scaffold
        phase="Phase 3 — Atlas"
        blurb="Domains → pipeline components → head-to-head outcomes, plus quantum-only branches. Built on Cytoscape.js from a new curated ontology joined to derived verdicts."
        bullets={[
          'New curated domain→component ontology (not yet in RESEARCH_MAP.yaml)',
          'Cytoscape.js graph with claim-level and replication kept distinct',
          '"No advantage found" cells as a first-class, visible state',
          'Literature-suggested nodes fed from Discover',
          'Separate static export for public hosting (no filesystem/queue routes)',
        ]}
      />
    </>
  )
}
