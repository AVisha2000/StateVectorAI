import assert from 'node:assert/strict'
import test from 'node:test'
import {
  outcomeClass,
  indexVerdicts,
  joinCellVerdict,
  resolveOntology,
  atlasSummary,
  filteredCells,
  claimRank,
  replicationRank,
} from './atlasModel.js'
import { ATLAS_SEED } from './atlasOntology.seed.js'

test('outcomeClass buckets status/kind — "no advantage" is classical_holds, not green or hidden', () => {
  assert.equal(outcomeClass({ status: 'negative' }), 'classical_holds')
  assert.equal(outcomeClass({ status: 'quantum_inspired' }), 'classical_holds')
  assert.equal(outcomeClass({ status: 'partial' }), 'quantum_candidate')
  assert.equal(outcomeClass({ status: 'blocked' }), 'open')
  assert.equal(outcomeClass({ kind: 'quantum_only', status: 'unexplored' }), 'quantum_only')
  assert.equal(outcomeClass({ kind: 'unexplored', status: 'unexplored' }), 'unexplored')
})

test('joinCellVerdict: canonical verdict wins for claim/replication, seed is fallback, fields stay separate', () => {
  const cell = {
    id: 'x', area_id: 'a1', kind: 'head_to_head', pipeline_stage: 'model',
    seed_status: 'partial', seed_claim_level: 'diagnostic', seed_replication_status: 'none',
    verdict_ref: { verdict_key: 'k1' },
  }
  const idx = indexVerdicts([{ id: 9, verdict_key: 'k1', claim_level: 'empirical', claim_status: 'candidate', replication_status: 'multi_seed_single_instance' }])
  const r = joinCellVerdict(cell, idx)
  assert.equal(r.provenance, 'derived_verdict')
  // Map claim axis stays on the RESEARCH_MAP ladder (seed); the verdict's own
  // (different) claim vocabulary is surfaced separately, never conflated.
  assert.equal(r.claim_level, 'diagnostic')
  assert.equal(r.verdict_claim_level, 'empirical')
  assert.equal(r.verdict_claim_status, 'candidate')
  assert.equal(r.replication_status, 'multi_seed_single_instance') // canonical ledger value wins (shared vocab)
  assert.equal(r.status, 'partial') // color status stays curated
  assert.equal(r.outcome_class, 'quantum_candidate')
  assert.equal(r.verdict_id, 9)
  // claim and replication are never merged into one another
  assert.notEqual(r.claim_level, r.replication_status)
})

test('joinCellVerdict falls back to labeled seed when no snapshot matches', () => {
  const cell = { id: 'x', seed_status: 'negative', seed_claim_level: 'diagnostic', seed_replication_status: 'multi_seed_single_instance', verdict_ref: null }
  const r = joinCellVerdict(cell, indexVerdicts([]))
  assert.equal(r.provenance, 'seed')
  assert.equal(r.claim_level, 'diagnostic')
  assert.equal(r.replication_status, 'multi_seed_single_instance')
  assert.equal(r.outcome_class, 'classical_holds')
})

test('resolveOntology groups cells into pipeline-stage components and keeps flat cells', () => {
  const resolved = resolveOntology(ATLAS_SEED, [], null)
  assert.ok(resolved.domains.length >= 4)
  const d0 = resolved.domains[0]
  assert.ok(Array.isArray(d0.components) && d0.components.length >= 1)
  assert.ok(d0.components.every((comp) => comp.id.includes('::')))
  assert.ok(d0.cells.every((c) => c.outcome_class && c.claim_level != null))
})

test('atlasSummary counts every outcome including classical/null — never silently drops them', () => {
  const resolved = resolveOntology(ATLAS_SEED, [], null)
  const s = atlasSummary(resolved)
  const sumBuckets = s.quantum_candidate + s.quantum_only + s.classical_holds + s.open + s.suggested + s.unexplored
  assert.equal(sumBuckets, s.total)
  assert.ok(s.classical_holds >= 1) // the seed has tested-null areas and they are counted
  assert.ok(s.total >= 15)
})

test('filters select by outcome/claim/replication independently', () => {
  const resolved = resolveOntology(ATLAS_SEED, [], null)
  const holds = filteredCells(resolved, { outcome: 'classical_holds' })
  assert.ok(holds.length >= 1 && holds.every((c) => c.outcome_class === 'classical_holds'))
  const untested = filteredCells(resolved, { claim: 'untested' })
  assert.ok(untested.every((c) => c.claim_level === 'untested'))
  assert.ok(filteredCells(resolved, {}).length === atlasSummary(resolved).total)
})

test('rank helpers order the two distinct vocabularies', () => {
  assert.equal(claimRank('untested'), 0)
  assert.ok(claimRank('empirical') === 0) // not in RESEARCH_MAP claim_levels → 0 (unknown)
  assert.ok(claimRank('formal') > claimRank('diagnostic'))
  assert.equal(replicationRank('none'), 0)
  assert.ok(replicationRank('multi_hardware_calibration') > replicationRank('single_task_instance'))
})
