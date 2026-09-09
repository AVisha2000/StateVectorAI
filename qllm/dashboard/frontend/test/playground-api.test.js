import assert from 'node:assert/strict'
import test from 'node:test'

import { playgroundResponse } from '../api/playground.js'
import { RESEARCH_WORLD_SNAPSHOT } from '../src/lib/researchWorld.js'

const HUMAN = { 'X-StateVector-Test-Actor': 'human-reviewer' }

test('serves the canonical deterministic Research World fixture over GET only', () => {
  const result = playgroundResponse('GET', 'research-world')
  assert.equal(result.status, 200)
  assert.strictEqual(result.body, RESEARCH_WORLD_SNAPSHOT)
  assert.equal(result.body.schema_version, '1')
  assert.equal(result.body.mode, 'fixture')
  assert.equal(result.body.regions.length, 3)
  assert.equal(result.body.agents.length, 6)
  assert.deepEqual(result.body.allocation.by_region, { physics: 2, mathematics: 2, 'machine-learning': 2 })
  assert.equal(result.body.cost.coverage.total_agents, 6)
  assert.deepEqual(playgroundResponse('POST', 'research-world'), {
    status: 404,
    body: { detail: 'not found in the UAT playground' },
  })
})

test('serves deterministic playground state without a worker', () => {
  assert.deepEqual(playgroundResponse('GET', 'health'), {
    status: 200,
    body: { ok: true, mode: 'playground', persistent: false, worker: false },
  })
  assert.equal(playgroundResponse('GET', 'jobs').body.length, 4)
  assert.equal(playgroundResponse('GET', 'verdicts/102').body.snapshot.id, 102)
  const work = playgroundResponse('GET', 'work')
  assert.equal(work.status, 200)
  assert.equal(work.body[0].id, 'work-quantum-controls')
  assert.equal('persistent' in work.body[0], false)
  const decisions = playgroundResponse('GET', 'decisions')
  assert.equal(decisions.status, 200)
  assert.equal(decisions.body[0].request_payload.recommendation, 'Pause for a narrower question')
  assert.equal(playgroundResponse('GET', 'decisions/decision-control-scope').body.allowed_actions[0], 'resolve')
  const detail = playgroundResponse('GET', 'work/work-quantum-controls').body
  assert.deepEqual(detail.events.map(({ revision, event_type }) => [revision, event_type]), [
    [1, 'proposed'], [2, 'claimed'], [3, 'started'], [4, 'decision_requested'], [5, 'claim_proposed'], [6, 'claim_reviewed'],
  ])
  assert.deepEqual(detail.revisions.map(({ revision, state, owner_actor_id }) => [revision, state, owner_actor_id]), [
    [1, 'available', null], [2, 'claimed', 'actor.agent.alpha'], [3, 'running', 'actor.agent.alpha'],
    [4, 'running', 'actor.agent.alpha'], [5, 'submitted', null], [6, 'ready_for_human', null],
  ])
  assert.deepEqual(detail.revisions.map(({ created_at }) => created_at), detail.events.map(({ created_at }) => created_at))
  assert.equal(detail.revisions[1].lease_expires_at, detail.revisions[2].lease_expires_at)
  assert.equal(detail.revisions[2].lease_expires_at, detail.revisions[3].lease_expires_at)
  assert.equal(detail.revisions[1].lease_expires_at, detail.events[1].created_at + detail.events[1].payload.lease_seconds)
  assert.equal(detail.events.at(-1).event_type, 'claim_reviewed')
  assert.equal(detail.revision, 6)
  const lineage = playgroundResponse('GET', 'objects/work-quantum-controls/lineage')
  assert.equal(lineage.body.authority, 'coordination_provisional')
  assert.equal(lineage.body.scientific_acceptance, false)
  assert.equal(lineage.body.canonical_claim_updated, false)
  assert.ok(lineage.body.nodes.some((node) => node.type === 'classification' && node.revision === 1))
  assert.ok(lineage.body.nodes.some((node) => node.type === 'decision' && node.revision === 1))
  assert.ok(lineage.body.edges.filter((edge) => edge.relation === 'classifies').length === 2)
  assert.ok(lineage.body.edges.some((edge) => edge.relation === 'decides'))
  const submission = playgroundResponse('GET', 'evidence/submission-control-scope').body
  assert.equal(submission.outcome, 'null')
  assert.equal(submission.authority, 'coordination_provisional')
  assert.equal(submission.scientific_acceptance, false)
  assert.deepEqual(submission.items.map(({ kind }) => kind), ['supporting', 'contradicting', 'null', 'failed', 'excluded'])
  assert.ok(submission.items.every(({ anchor_type, anchor_id }) => anchor_type && anchor_id))
  for (const { anchor_type, anchor_id } of submission.items) {
    assert.ok(lineage.body.nodes.some((node) => node.id === anchor_id && node.type === anchor_type))
    assert.ok(lineage.body.edges.some((edge) => (
      edge.from === submission.id && edge.to === anchor_id && edge.relation === 'anchors'
    )))
  }
  const claim = playgroundResponse('GET', 'claim-proposals/claim-control-scope').body
  const review = playgroundResponse('GET', 'reviews/review-control-scope').body
  assert.equal(claim.canonical_claim_updated, false)
  assert.equal(review.canonical_claim_updated, false)
  const overlap = playgroundResponse('GET', 'work/work-quantum-controls/overlaps').body
  assert.equal(overlap.candidates[0].classification, 'replication')
})

test('simulates mutations without claiming persistence', () => {
  const result = playgroundResponse('POST', 'jobs', { run_name: 'uat-check' })
  assert.equal(result.status, 200)
  assert.equal(result.body.run_name, 'uat-check')
  assert.equal(result.body.playground, true)
  assert.equal(result.body.persistent, false)
  const decision = playgroundResponse('POST', 'decisions/decision-control-scope/resolve', {
    expected_revision: 1,
    idempotency_key: 'synthetic-resolution',
    answer: 'Linear readout',
    rationale: 'Keep this UAT correction bounded.',
  }, HUMAN)
  assert.equal(decision.status, 200)
  assert.equal(decision.body.status, 'resolved')
  assert.equal(decision.body.resolver_actor_id, 'actor.human.reviewer')
  assert.deepEqual(Object.keys(decision.body), [
    'id', 'work_id', 'revision', 'status', 'question', 'options', 'required_human_capability',
    'request_payload', 'answer', 'rationale', 'resolver_actor_id', 'allowed_actions',
    'blocking_decision_ids', 'actionable_next_state', 'provenance_links',
  ])
  assert.equal('persistent' in decision.body, false)
  assert.equal('resolution_event' in decision.body, false)
  assert.equal(playgroundResponse('GET', 'decisions/decision-control-scope').body.status, 'open')
})

test('requires the exact synthetic human selector', () => {
  const payload = { expected_revision: 1, idempotency_key: 'synthetic-resolution', answer: 'Linear readout', rationale: 'Bounded synthetic answer.' }
  assert.equal(playgroundResponse('POST', 'decisions/decision-control-scope/resolve', payload).status, 403)
  assert.equal(playgroundResponse('POST', 'decisions/decision-control-scope/resolve', payload, { 'x-statevector-test-actor': 'agent-alpha' }).status, 403)
  assert.equal(playgroundResponse('POST', 'decisions/decision-control-scope/resolve', payload, { 'X-StateVector-Test-Actor': 'human-reviewer ' }).status, 403)
})

test('returns 422 for decision-resolution schema and bounds failures', () => {
  const payload = { expected_revision: 1, idempotency_key: 'synthetic-resolution', answer: 'Linear readout', rationale: 'Bounded synthetic answer.' }
  for (const invalid of [
    null,
    [],
    { ...payload, extra: 'not allowed' },
    { idempotency_key: payload.idempotency_key, answer: payload.answer, rationale: payload.rationale },
    { ...payload, expected_revision: 0 },
    { ...payload, expected_revision: 1.5 },
    { ...payload, idempotency_key: '' },
    { ...payload, idempotency_key: 'x'.repeat(129) },
    { ...payload, answer: '' },
    { ...payload, answer: 'x'.repeat(2001) },
    { ...payload, rationale: '' },
    { ...payload, rationale: 'x'.repeat(4001) },
  ]) {
    assert.equal(playgroundResponse('POST', 'decisions/decision-control-scope/resolve', invalid, HUMAN).status, 422)
  }
  assert.equal(playgroundResponse('POST', 'decisions/decision-control-scope/resolve', { ...payload, idempotency_key: 'x'.repeat(128) }, HUMAN).status, 200)
  assert.equal(playgroundResponse('POST', 'decisions/decision-control-scope/resolve', { ...payload, answer: 'x'.repeat(2000) }, HUMAN).status, 409)
  assert.equal(playgroundResponse('POST', 'decisions/decision-control-scope/resolve', { ...payload, rationale: 'x'.repeat(4000) }, HUMAN).status, 200)
})

test('returns 409 for a stale revision or answer outside the available options', () => {
  const payload = { expected_revision: 1, idempotency_key: 'synthetic-resolution', answer: 'Linear readout', rationale: 'Bounded synthetic answer.' }
  assert.equal(playgroundResponse('POST', 'decisions/decision-control-scope/resolve', { ...payload, expected_revision: 2 }, HUMAN).status, 409)
  assert.equal(playgroundResponse('POST', 'decisions/decision-control-scope/resolve', { ...payload, answer: 'Invent a new option' }, HUMAN).status, 409)
})

test('rejects credential-bearing decision values without reflection or persistence', () => {
  const payload = { expected_revision: 1, idempotency_key: 'synthetic-resolution', answer: 'Linear readout', rationale: 'Bounded synthetic answer.' }
  for (const [field, probe] of [
    ['rationale', 'api_key=synthetic-credential-probe'],
    ['idempotency_key', 'token=synthetic-credential-probe'],
  ]) {
    const result = playgroundResponse('POST', 'decisions/decision-control-scope/resolve', { ...payload, [field]: probe }, HUMAN)
    assert.deepEqual(result, { status: 409, body: { detail: 'credential-bearing values are not accepted' } })
    assert.equal(JSON.stringify(result).includes(probe), false)
    const decision = playgroundResponse('GET', 'decisions/decision-control-scope').body
    assert.equal(decision.status, 'open')
    assert.equal(decision.answer, null)
    assert.equal(decision.rationale, null)
  }
})

test('fails closed for routes outside the playground contract', () => {
  assert.deepEqual(playgroundResponse('GET', 'private-artifacts'), {
    status: 404,
    body: { detail: 'not found in the UAT playground' },
  })
})
