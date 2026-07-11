import { PageHeader, Scaffold } from '../lib/ui.jsx'

export default function Library() {
  return (
    <>
      <PageHeader
        title="Library"
        sub="The research archive and knowledge vault — everything the lab reads, auto-scanned and manually dumped, synthesized into a knowledge graph."
      />
      <Scaffold
        phase="Phase 4 — research loop (greenfield)"
        blurb="A paper archive with a knowledge graph linking papers ↔ concepts ↔ experiments ↔ Atlas nodes. Scanners are bounded (quant-ph + a QML-filtered cs.LG slice, per-day cap) with tiered, cheap scoring."
        bullets={[
          'Filterable archive: inbox → reviewing → linked → feature-candidate → archived',
          'Feature-potential score and bidirectional experiment links',
          'Knowledge graph (Cytoscape.js) over the vault',
          'Scanner config + synthesis status',
        ]}
      />
    </>
  )
}
