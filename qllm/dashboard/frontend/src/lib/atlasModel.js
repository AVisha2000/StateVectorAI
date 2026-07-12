// Pure Atlas model: join the (seed or canonical) ontology to derived /verdicts
// snapshots and shape it for the graph and list. Framework-free for node --test.
//
// Integrity, encoded here so it is testable:
//   * `outcome_class` is a COARSE COLOR BUCKET derived only from the RESEARCH_MAP
//     `status` (a planning index) + cell `kind`. It is NOT a verdict, NOT a
//     composite/averaged advantage score, and "classical holds / no advantage
//     found" is a first-class bucket rendered at full prominence (never green,
//     never hidden).
//   * `claim_level` and `replication_status` come from the canonical /verdicts
//     snapshot when one matches, else the labeled seed fallback — and they are
//     ALWAYS kept as separate fields, never merged into one another or into
//     `outcome_class`.
import { ATLAS_SEED } from './atlasOntology.seed.js'

// RESEARCH_MAP status → outcome bucket (color only).
const STATUS_OUTCOME = Object.freeze({
  negative: 'classical_holds', // tested, quantum did not beat the matched control
  quantum_inspired: 'classical_holds', // useful result is classically deployable
  methodology_only: 'classical_holds', // a control/method, not an application advantage
  partial: 'quantum_candidate', // a mechanism works; end-to-end not established
  blocked: 'open', // gated — do not interpret until named gates pass
  infrastructure: 'open', // research-enabling, not an advantage hypothesis
  open: 'open',
  unexplored: 'unexplored',
})

export const OUTCOME_ORDER = Object.freeze([
  'quantum_candidate', 'quantum_only', 'classical_holds', 'open', 'suggested', 'unexplored',
])

export const OUTCOME_LABELS = Object.freeze({
  quantum_candidate: 'Quantum candidate',
  quantum_only: 'Quantum-only paradigm',
  classical_holds: 'Classical holds · no advantage',
  open: 'Open · gated',
  suggested: 'Literature-suggested',
  unexplored: 'Unexplored',
})

// The two DISTINCT ordered vocabularies from RESEARCH_MAP.yaml. Kept separate so
// claim strength (border width) and replication (border style) never collapse
// into one visual channel.
export const CLAIM_LEVELS = Object.freeze([
  'untested', 'correctness', 'diagnostic', 'mechanism', 'paired_empirical', 'scaling', 'hardware', 'practical', 'formal',
])
export const REPLICATION_STATUSES = Object.freeze([
  'none', 'within_study_resampling', 'single_task_instance', 'multi_seed_single_instance', 'multi_instance', 'multi_hardware_calibration',
])

export function claimRank(level) {
  const i = CLAIM_LEVELS.indexOf(level)
  return i < 0 ? 0 : i
}
export function replicationRank(rep) {
  const i = REPLICATION_STATUSES.indexOf(rep)
  return i < 0 ? 0 : i
}

export function outcomeClass(cell) {
  if (!cell) return 'unexplored'
  if (cell.kind === 'quantum_only') return 'quantum_only'
  if (cell.kind === 'suggested') return 'suggested'
  if (cell.kind === 'unexplored') return 'unexplored'
  return STATUS_OUTCOME[cell.status] || 'open'
}

// Index verdict snapshots by both join keys a cell's verdict_ref may use.
export function indexVerdicts(snapshots) {
  const idx = new Map()
  for (const s of Array.isArray(snapshots) ? snapshots : []) {
    if (!s) continue
    if (s.verdict_key) idx.set(`key:${s.verdict_key}`, s)
    if (s.source_kind && s.source_id != null) idx.set(`src:${s.source_kind}:${s.source_id}`, s)
  }
  return idx
}

// Resolve one cell: canonical verdict wins for claim/replication; seed is the
// labeled fallback. `status` (and thus color) stays from the curated ontology.
export function joinCellVerdict(cell, verdictIndex) {
  const ref = cell.verdict_ref || null
  let snap = null
  if (ref && verdictIndex) {
    if (ref.verdict_key) snap = verdictIndex.get(`key:${ref.verdict_key}`) || null
    if (!snap && ref.source_kind && ref.source_id != null) {
      snap = verdictIndex.get(`src:${ref.source_kind}:${ref.source_id}`) || null
    }
  }
  const resolved = {
    ...cell,
    status: cell.seed_status ?? null,
    // The map's claim axis stays on the RESEARCH_MAP ladder (untested→formal).
    // A verdict's claim_level uses the DIFFERENT classify_claim vocabulary
    // (empirical/smoke/…) which does not map onto this ladder, so it is surfaced
    // SEPARATELY as verdict_claim_level rather than overriding the map axis.
    claim_level: cell.seed_claim_level ?? null,
    // replication_status shares the RESEARCH_MAP vocabulary, so the canonical
    // ledger value (via the snapshot) does win here.
    replication_status: snap?.replication_status ?? cell.seed_replication_status ?? null,
    verdict_claim_level: snap?.claim_level ?? null,
    verdict_claim_status: snap?.claim_status ?? null,
    provenance: snap ? 'derived_verdict' : 'seed',
    verdict_id: snap?.id ?? null,
  }
  resolved.outcome_class = outcomeClass(resolved)
  return resolved
}

// Resolve the whole ontology; group each domain's cells into pipeline-stage
// components (the middle graph tier) while also keeping the flat cell list.
export function resolveOntology(ontology, verdicts, explore) {
  const onto = ontology || ATLAS_SEED
  const idx = indexVerdicts(verdicts)
  const domains = (onto.domains || []).map((d) => {
    const cells = (d.cells || []).map((c) => joinCellVerdict(c, idx))
    const byStage = new Map()
    for (const c of cells) {
      const stage = c.pipeline_stage || 'other'
      if (!byStage.has(stage)) byStage.set(stage, [])
      byStage.get(stage).push(c)
    }
    const components = [...byStage.entries()].map(([stage, cs]) => ({
      id: `${d.id}::${stage}`,
      label: stage,
      pipeline_stage: stage,
      cells: cs,
    }))
    return { ...d, cells, components }
  })
  return { ...onto, domains, relations: onto.relations || [], explore: explore || null }
}

// Per-outcome counts, including an explicit null/classical count so the omission
// of "no advantage found" can never be silent.
export function atlasSummary(resolved) {
  const counts = { total: 0 }
  for (const o of OUTCOME_ORDER) counts[o] = 0
  for (const d of resolved?.domains || []) {
    for (const c of d.cells || []) {
      counts.total += 1
      counts[c.outcome_class] = (counts[c.outcome_class] || 0) + 1
    }
  }
  return counts
}

export function matchesOutcome(cell, outcome) {
  return !outcome || outcome === 'all' || cell.outcome_class === outcome
}
export function matchesClaim(cell, level) {
  return !level || level === 'all' || cell.claim_level === level
}
export function matchesReplication(cell, rep) {
  return !rep || rep === 'all' || cell.replication_status === rep
}

// Flatten resolved cells with their domain context, applying filters. Used by
// the list view and (indirectly) the graph.
export function filteredCells(resolved, { outcome, claim, replication } = {}) {
  const out = []
  for (const d of resolved?.domains || []) {
    for (const c of d.cells || []) {
      if (matchesOutcome(c, outcome) && matchesClaim(c, claim) && matchesReplication(c, replication)) {
        out.push({ ...c, domain_id: d.id, domain_label: d.label })
      }
    }
  }
  return out
}
