import { useMemo, useState } from 'react'
import { useAtlasOntology, useVerdicts } from '../lib/hooks.js'
import { PageHeader, Loading } from '../lib/ui.jsx'
import AtlasGraphSvg from '../components/atlas/AtlasGraphSvg.jsx'
import { ATLAS_SEED } from '../lib/atlasOntology.seed.js'
import {
  resolveOntology,
  atlasSummary,
  filteredCells,
  OUTCOME_ORDER,
  OUTCOME_LABELS,
  CLAIM_LEVELS,
  REPLICATION_STATUSES,
} from '../lib/atlasModel.js'
import AtlasList from '../components/atlas/AtlasList.jsx'
import AtlasNodeDetail from '../components/atlas/AtlasNodeDetail.jsx'
import AtlasLegend from '../components/atlas/AtlasLegend.jsx'

export default function Atlas() {
  const ontologyQuery = useAtlasOntology()
  const { data: verdictData } = useVerdicts()

  const [view, setView] = useState('list')
  const [outcome, setOutcome] = useState('all')
  const [claim, setClaim] = useState('all')
  const [replication, setReplication] = useState('all')
  const [selectedId, setSelectedId] = useState(null)

  // Seed fallback until GET /api/atlas/ontology ships; verdict snapshots refine
  // per-cell claim/replication where they match.
  const ontology = ontologyQuery.data || ATLAS_SEED
  const usingSeed = !ontologyQuery.data
  const snapshots = Array.isArray(verdictData?.snapshots) ? verdictData.snapshots : []

  const resolved = useMemo(() => resolveOntology(ontology, snapshots, null), [ontology, snapshots])
  const summary = useMemo(() => atlasSummary(resolved), [resolved])
  const allCells = useMemo(() => filteredCells(resolved, {}), [resolved])
  const filtered = useMemo(
    () => filteredCells(resolved, { outcome, claim, replication }),
    [resolved, outcome, claim, replication],
  )
  const selected = useMemo(() => allCells.find((c) => c.id === selectedId) || null, [allCells, selectedId])

  // Regroup filtered cells back under their domains (with pipeline-stage
  // components) so the list and the graph render the same filtered structure.
  const filteredDomains = useMemo(() => {
    const byDomain = new Map()
    for (const c of filtered) {
      if (!byDomain.has(c.domain_id)) byDomain.set(c.domain_id, { id: c.domain_id, label: c.domain_label, cells: [] })
      byDomain.get(c.domain_id).cells.push(c)
    }
    return [...byDomain.values()].map((d) => {
      const byStage = new Map()
      for (const c of d.cells) {
        const stage = c.pipeline_stage || 'other'
        if (!byStage.has(stage)) byStage.set(stage, [])
        byStage.get(stage).push(c)
      }
      const components = [...byStage.entries()].map(([stage, cs]) => ({ id: `${d.id}::${stage}`, label: stage, pipeline_stage: stage, cells: cs }))
      return { ...d, components }
    })
  }, [filtered])

  if (ontologyQuery.isLoading) return <Loading label="Loading the Atlas…" />

  return (
    <>
      <PageHeader
        title="Atlas — the map of quantum vs classical ML"
        sub="Classical ML as the base map, quantum overlaid. Rendered from the research map + the verdict store — honest by construction: null / 'no advantage found' cells are as prominent as positive ones."
        actions={
          <div className="seg" role="tablist">
            <button className={view === 'graph' ? 'on' : undefined} onClick={() => setView('graph')}>Graph</button>
            <button className={view === 'list' ? 'on' : undefined} onClick={() => setView('list')}>List</button>
          </div>
        }
      />

      {usingSeed ? (
        <div className="notice" style={{ marginTop: 14 }}>
          Rendering the <b>frontend seed ontology</b> (placeholder) — the curated domain→component map is not yet served at
          <span className="mono"> /api/atlas/ontology</span>. Per-cell claim level and replication still come from the live
          <span className="mono"> /verdicts</span> store when a snapshot matches.
        </div>
      ) : null}

      {/* Outcome summary — every bucket visible, nulls included. */}
      <div className="atlas-summary" style={{ marginTop: 14 }}>
        {OUTCOME_ORDER.map((o) => (
          <button
            key={o}
            className={`atlas-summary-tile ${outcome === o ? 'on' : ''}`}
            onClick={() => setOutcome(outcome === o ? 'all' : o)}
            title={`Filter to ${OUTCOME_LABELS[o]}`}
          >
            <span className={`atlas-dot atlas-oc-${o}`} />
            <span className="num v">{summary[o] || 0}</span>
            <span className="s">{OUTCOME_LABELS[o]}</span>
          </button>
        ))}
      </div>

      <div className="row" style={{ margin: '14px 0 12px', gap: 10 }}>
        <label className="row" style={{ gap: 6 }}>
          <span className="microlabel">Claim</span>
          <select className="mini" value={claim} onChange={(e) => setClaim(e.target.value)}>
            <option value="all">all</option>
            {CLAIM_LEVELS.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </label>
        <label className="row" style={{ gap: 6 }}>
          <span className="microlabel">Replication</span>
          <select className="mini" value={replication} onChange={(e) => setReplication(e.target.value)}>
            <option value="all">all</option>
            {REPLICATION_STATUSES.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </label>
        <span className="hint spacer">{filtered.length} of {summary.total} cells</span>
      </div>

      <AtlasLegend />

      <div className="atlas-shell" style={{ marginTop: 14 }}>
        <div>
          {view === 'graph' ? (
            filteredDomains.length === 0 ? (
              <div className="state">No cells match these filters.</div>
            ) : (
              <AtlasGraphSvg resolved={{ ...resolved, domains: filteredDomains }} onSelect={setSelectedId} selectedId={selectedId} />
            )
          ) : filteredDomains.length === 0 ? (
            <div className="state">No cells match these filters.</div>
          ) : (
            <AtlasList domains={filteredDomains} onSelect={setSelectedId} selectedId={selectedId} />
          )}
        </div>
        <div className="atlas-side">
          <AtlasNodeDetail cell={selected} />
        </div>
      </div>
    </>
  )
}
