import {
  ARXIV_SCAN,
  ATLAS_ONTOLOGY,
  CAPABILITIES,
  DATASETS,
  DESIGNER_CAPABILITIES,
  DESIGNER_VALIDATION,
  DIAGNOSTICS_7,
  JOBS,
  MODEL_GRAPH_7,
  MODEL_TESTS_7,
  OVERVIEW,
  PRESETS,
  SCALING_GRP,
  STATUS,
  STUDIES,
  STUDY_1,
  STUDY_1_WORKSPACES,
  VERDICTS,
  VERDICT_DETAIL_101,
  WORKSPACE_7,
} from '../e2e/fixtures.js'
import { RESEARCH_WORLD_SNAPSHOT } from '../src/lib/researchWorld.js'

const TEST_ACTOR_HEADER = 'X-StateVector-Test-Actor'
const HUMAN_REVIEWER = 'human-reviewer'
const CREDENTIAL_BEARING_PATTERN = /(?:\bbearer\s+[a-z0-9._~+\/=-]{8,}|api[_ -]?key\s*[:=]\s*\S+|-----begin[^\n]{0,40}private key-----|\bsk-[a-z0-9_-]{8,}|\bAIza[a-z0-9_-]{16,}|\bgh[pousr]_[a-z0-9]{20,}|\bxox[baprs]-[a-z0-9-]{10,}|\bAKIA[0-9A-Z]{16}\b|password\s*[:=]\s*\S+|https?:\/\/[^/\s:@]+:[^/@\s]+@|(?:^|[\s?&#])(?:access[_-]?token|api[_-]?key|auth[_-]?token|client[_-]?secret|credential|password|private[_-]?key|refresh[_-]?token|secret|token)=[^&#\s]+)/i

const PLAYGROUND_WORK = [
  {
    id: 'work-quantum-controls', revision: 6, state: 'ready_for_human', owner_actor_id: null, lease_expires_at: null,
    payload: {
      title: 'Map matched controls for variational readout',
      question: 'Which classical readout controls match the current quantum circuit comparison?',
      objective: 'Produce a bounded control inventory for human review.',
      scope: 'Quantum readout comparisons only.', stop_rule: 'Stop after the curated control set is mapped.',
      acceptance_evidence: ['Control inventory with provenance links.'], domain: 'quantum', requested_capabilities: ['literature:read'],
      research_signature: {
        question: 'which classical readout controls match the current quantum circuit comparison?',
        system: 'statevector simulator', method: 'matched control map', dataset: 'synthetic uat fixture',
        metric: 'control coverage', comparator: 'matched classical readout', regime: 'small cpu-only',
        intended_evidence_level: 'diagnostic',
      },
    },
    allowed_actions: [], blocking_overlap_ids: [], blocking_decision_ids: ['decision-control-scope'], actionable_next_state: null, provenance_links: [],
  },
  {
    id: 'work-noise-survey', revision: 1, state: 'available', owner_actor_id: null, lease_expires_at: null,
    payload: {
      title: 'Survey noise-model assumptions',
      question: 'Which noise assumptions are explicit in the simulator studies already catalogued?',
      objective: 'Summarise assumptions without making a performance claim.',
      scope: 'Simulator studies in the shared quantum catalogue.', stop_rule: 'Stop after each included study has an assumption note.',
      acceptance_evidence: ['Assumption notes linked to source records.'], domain: 'quantum', requested_capabilities: ['catalogue:read'],
      research_signature: {
        question: 'which noise assumptions are explicit in the simulator studies already catalogued?',
        system: 'statevector simulator', method: 'literature survey', dataset: 'synthetic uat fixture',
        metric: 'assumption coverage', comparator: 'not applicable', regime: 'small cpu-only',
        intended_evidence_level: 'untested',
      },
    },
    allowed_actions: [], blocking_overlap_ids: [], blocking_decision_ids: [], actionable_next_state: null, provenance_links: [],
  },
]

const PLAYGROUND_DECISIONS = [
  {
    id: 'decision-control-scope', work_id: 'work-quantum-controls', revision: 1, status: 'open',
    question: 'Which matched control family should the next synthesis prioritise?',
    options: ['Linear readout', 'Classical nonlinear readout', 'Pause for a narrower question'],
    required_human_capability: 'decision:resolve',
    request_payload: {
      question: 'Which matched control family should the next synthesis prioritise?',
      options: ['Linear readout', 'Classical nonlinear readout', 'Pause for a narrower question'],
      recommendation: 'Pause for a narrower question', supporting_evidence_ids: [],
      uncertainty: 'Existing records mix model families.', consequences: ['The next work package stays bounded.'],
      reversible: true, expires_at: '2100-01-01T00:00:00Z', required_human_capability: 'decision:resolve',
    },
    answer: null, rationale: null, resolver_actor_id: null, allowed_actions: ['resolve'], blocking_decision_ids: [], actionable_next_state: 'resolved',
    provenance_links: [{ relation: 'attached_to', object_type: 'work', object_id: 'work-quantum-controls' }],
  },
]

const PLAYGROUND_WORK_DETAIL = {
  ...PLAYGROUND_WORK[0],
  events: [
    { id: 1, work_id: 'work-quantum-controls', revision: 1, event_type: 'proposed', actor_id: 'actor.agent.alpha', capability: 'work:propose', previous_state: null, next_state: 'available', payload: PLAYGROUND_WORK[0].payload, created_at: 1788343200 },
    { id: 2, work_id: 'work-quantum-controls', revision: 2, event_type: 'claimed', actor_id: 'actor.agent.alpha', capability: 'work:claim', previous_state: 'available', next_state: 'claimed', payload: { lease_seconds: 3600 }, created_at: 1788343260 },
    { id: 3, work_id: 'work-quantum-controls', revision: 3, event_type: 'started', actor_id: 'actor.agent.alpha', capability: 'work:start', previous_state: 'claimed', next_state: 'running', payload: {}, created_at: 1788343320 },
    { id: 4, work_id: 'work-quantum-controls', revision: 4, event_type: 'decision_requested', actor_id: 'actor.agent.alpha', capability: 'decision:request', previous_state: 'running', next_state: 'running', payload: { decision_id: 'decision-control-scope' }, created_at: 1788343380 },
    { id: 5, work_id: 'work-quantum-controls', revision: 5, event_type: 'claim_proposed', actor_id: 'actor.agent.alpha', capability: 'claim:propose', previous_state: 'running', next_state: 'submitted', payload: { claim_id: 'claim-control-scope', submission_id: 'submission-control-scope', claim_hash: '3'.repeat(64) }, created_at: 1788343440 },
    { id: 6, work_id: 'work-quantum-controls', revision: 6, event_type: 'claim_reviewed', actor_id: 'actor.agent.beta', capability: 'review:submit', previous_state: 'submitted', next_state: 'ready_for_human', payload: { review_id: 'review-control-scope', claim_id: 'claim-control-scope', outcome: 'verified_for_handoff' }, created_at: 1788343500 },
  ],
  revisions: [
    { work_id: 'work-quantum-controls', revision: 1, state: 'available', payload: PLAYGROUND_WORK[0].payload, owner_actor_id: null, lease_expires_at: null, created_at: 1788343200 },
    { work_id: 'work-quantum-controls', revision: 2, state: 'claimed', payload: PLAYGROUND_WORK[0].payload, owner_actor_id: 'actor.agent.alpha', lease_expires_at: 1788346860, created_at: 1788343260 },
    { work_id: 'work-quantum-controls', revision: 3, state: 'running', payload: PLAYGROUND_WORK[0].payload, owner_actor_id: 'actor.agent.alpha', lease_expires_at: 1788346860, created_at: 1788343320 },
    { work_id: 'work-quantum-controls', revision: 4, state: 'running', payload: PLAYGROUND_WORK[0].payload, owner_actor_id: 'actor.agent.alpha', lease_expires_at: 1788346860, created_at: 1788343380 },
    { work_id: 'work-quantum-controls', revision: 5, state: 'submitted', payload: PLAYGROUND_WORK[0].payload, owner_actor_id: null, lease_expires_at: null, created_at: 1788343440 },
    { work_id: 'work-quantum-controls', revision: 6, state: 'ready_for_human', payload: PLAYGROUND_WORK[0].payload, owner_actor_id: null, lease_expires_at: null, created_at: 1788343500 },
  ],
}

const PLAYGROUND_OVERLAPS = {
  work_id: 'work-quantum-controls', exact_match_requires_classification: false,
  authority: 'coordination_provisional',
  candidates: [{ id: 'prior-control-scope', candidate_type: 'prior_work', title: 'Synthetic matched-control reference', reasons: ['exact_identifier'], exact_match: true, classified: true, classification: 'replication' }],
}

const PLAYGROUND_SUBMISSION = {
  id: 'submission-control-scope', work_id: 'work-quantum-controls', revision: 1, work_revision: 3,
  status: 'submitted', authority: 'coordination_provisional', scientific_acceptance: false, outcome: 'null',
  summary: 'The synthetic comparison did not separate the readout families.', methods: 'Matched synthetic comparison.',
  controls: 'Linear classical readout.', regime: 'Small CPU-only UAT fixture.', uncertainty: 'No empirical finding is claimed.',
  limitations: ['Synthetic fixture only.'], gaps: ['Human taste is needed to choose the next family.'],
  items: [
    { kind: 'supporting', statement: 'Synthetic fixture metadata records the matched linear control.', anchor_type: 'resultsdb_job', anchor_id: '7' },
    { kind: 'contradicting', statement: 'Synthetic analogue metadata retains a different readout family.', anchor_type: 'resultsdb_job', anchor_id: '8' },
    { kind: 'null', statement: 'No synthetic separation was recorded.', anchor_type: 'prior_work', anchor_id: 'prior-control-scope' },
    { kind: 'failed', statement: 'A synthetic comparison job is marked failed and is not evidence of separation.', anchor_type: 'resultsdb_job', anchor_id: '9' },
    { kind: 'excluded', statement: 'A queued synthetic job is excluded because it has no completed result.', anchor_type: 'resultsdb_job', anchor_id: '10' },
  ],
  content_hash: '2'.repeat(64), predecessor_content_hash: null, correction_reason: null,
  submitted_by_actor_id: 'actor.agent.alpha', created_at: '2026-09-02T10:03:00Z',
}

const PLAYGROUND_CLAIM = {
  id: 'claim-control-scope', submission_id: PLAYGROUND_SUBMISSION.id, submission_revision: 1,
  work_id: 'work-quantum-controls', revision: 1, status: 'proposed', authority: 'coordination_provisional',
  scientific_acceptance: false, canonical_claim_updated: false, submission_hash: PLAYGROUND_SUBMISSION.content_hash,
  content_hash: '3'.repeat(64), statement: 'This UAT fixture is an unreviewed null result only.',
  scope: 'Synthetic fixture and matched control only.', comparator: 'Matched linear readout.', regime: 'Small CPU-only UAT fixture.',
  intended_evidence_level: 'diagnostic', uncertainty: 'No empirical finding is claimed.', limitations: ['Synthetic fixture only.'],
  predecessor_claim_content_hash: null, correction_reason: null, proposed_by_actor_id: 'actor.agent.alpha',
  created_at: '2026-09-02T10:04:00Z', work_revision: 5,
}

const PLAYGROUND_REVIEW = {
  id: 'review-control-scope', submission_id: PLAYGROUND_SUBMISSION.id, claim_id: PLAYGROUND_CLAIM.id,
  work_id: 'work-quantum-controls', revision: 1, authority: 'coordination_provisional', scientific_acceptance: false,
  canonical_claim_updated: false, submission_hash: PLAYGROUND_SUBMISSION.content_hash, claim_hash: PLAYGROUND_CLAIM.content_hash,
  content_hash: '4'.repeat(64), outcome: 'verified_for_handoff', rationale: 'The bounded synthetic provenance is internally consistent.',
  checks: ['Matched-control scope retained.'], limitations: ['This is not scientific acceptance.'],
  reviewed_by_actor_id: 'actor.agent.beta', created_at: '2026-09-02T10:05:00Z', work_revision: 6,
}

const PLAYGROUND_LINEAGE = {
  object_id: 'work-quantum-controls', work_id: 'work-quantum-controls', authority: 'coordination_provisional',
  scientific_acceptance: false, canonical_claim_updated: false,
  nodes: [
    { id: 'work-quantum-controls', type: 'work', revision: 6, hash: '1'.repeat(64) },
    { id: 'prior-control-scope', type: 'prior_work', revision: 1, hash: '5'.repeat(64) },
    { id: '7', type: 'resultsdb_job' },
    { id: '8', type: 'resultsdb_job' },
    { id: '9', type: 'resultsdb_job' },
    { id: '10', type: 'resultsdb_job' },
    { id: 'classification-control-scope', type: 'classification', revision: 1, hash: '6'.repeat(64) },
    { id: PLAYGROUND_SUBMISSION.id, type: 'submission', revision: 1, hash: PLAYGROUND_SUBMISSION.content_hash },
    { id: PLAYGROUND_CLAIM.id, type: 'claim', revision: 1, hash: PLAYGROUND_CLAIM.content_hash },
    { id: PLAYGROUND_REVIEW.id, type: 'review', revision: 1, hash: PLAYGROUND_REVIEW.content_hash },
    { id: PLAYGROUND_DECISIONS[0].id, type: 'decision', revision: 1, hash: '7'.repeat(64) },
  ],
  edges: [
    { from: 'prior-control-scope', to: 'work-quantum-controls', relation: 'informs' },
    { from: 'classification-control-scope', to: 'prior-control-scope', relation: 'classifies' },
    { from: 'classification-control-scope', to: 'work-quantum-controls', relation: 'classifies' },
    { from: PLAYGROUND_SUBMISSION.id, to: 'work-quantum-controls', relation: 'submitted_for' },
    { from: PLAYGROUND_SUBMISSION.id, to: '7', relation: 'anchors' },
    { from: PLAYGROUND_SUBMISSION.id, to: '8', relation: 'anchors' },
    { from: PLAYGROUND_SUBMISSION.id, to: 'prior-control-scope', relation: 'anchors' },
    { from: PLAYGROUND_SUBMISSION.id, to: '9', relation: 'anchors' },
    { from: PLAYGROUND_SUBMISSION.id, to: '10', relation: 'anchors' },
    { from: PLAYGROUND_CLAIM.id, to: PLAYGROUND_SUBMISSION.id, relation: 'proposes_from' },
    { from: PLAYGROUND_REVIEW.id, to: PLAYGROUND_CLAIM.id, relation: 'reviews' },
    { from: PLAYGROUND_DECISIONS[0].id, to: 'work-quantum-controls', relation: 'decides' },
  ],
}

const CONFIG_CHOICES = {
  architecture: ['mlp', 'qrnn'],
  backend: ['pennylane', 'tensorcircuit', 'tensorcircuit_mps'],
  datasets: DATASETS.map(({ name }) => name),
  playground: true,
}

const GET_ROUTES = new Map([
  ['/health', { ok: true, mode: 'playground', persistent: false, worker: false }],
  ['/lab/overview', OVERVIEW],
  ['/status', { ...STATUS, worker: 'PLAYGROUND · simulated', gpu_available: false }],
  ['/config/choices', CONFIG_CHOICES],
  ['/presets', PRESETS],
  ['/datasets', DATASETS],
  ['/jobs', JOBS],
  ['/verdicts', VERDICTS],
  ['/verdicts/101', VERDICT_DETAIL_101],
  ['/verdicts/102', { snapshot: VERDICTS.snapshots[1], history: [VERDICTS.snapshots[1]] }],
  ['/research/capabilities', CAPABILITIES],
  ['/research-world', RESEARCH_WORLD_SNAPSHOT],
  ['/atlas/ontology', ATLAS_ONTOLOGY],
  ['/designer/circuit', DESIGNER_CAPABILITIES],
  ['/work', PLAYGROUND_WORK],
  ['/decisions', PLAYGROUND_DECISIONS],
  ['/decisions/decision-control-scope', PLAYGROUND_DECISIONS[0]],
  ['/work/work-quantum-controls', PLAYGROUND_WORK_DETAIL],
  ['/work/work-quantum-controls/overlaps', PLAYGROUND_OVERLAPS],
  ['/objects/work-quantum-controls/lineage', PLAYGROUND_LINEAGE],
  [`/evidence/${PLAYGROUND_SUBMISSION.id}`, PLAYGROUND_SUBMISSION],
  [`/claim-proposals/${PLAYGROUND_CLAIM.id}`, PLAYGROUND_CLAIM],
  [`/reviews/${PLAYGROUND_REVIEW.id}`, PLAYGROUND_REVIEW],
  ['/jobs/7/workspace', WORKSPACE_7],
  ['/jobs/7/comparison', WORKSPACE_7.comparison],
  ['/jobs/7/diagnostics', DIAGNOSTICS_7],
  ['/jobs/7/model-graph', MODEL_GRAPH_7],
  ['/jobs/7/model-tests', MODEL_TESTS_7],
  ['/scaling-tests', [{ group_id: 'scale-grp', complete_count: 3, total_count: 3 }]],
  ['/scaling-tests/scale-grp', SCALING_GRP],
  ['/studies', STUDIES],
  ['/studies/1', STUDY_1],
  ...Object.entries(STUDY_1_WORKSPACES),
])

function cleanPath(value) {
  const path = Array.isArray(value) ? value.join('/') : String(value || '')
  return `/${path}`.replace(/\/+/g, '/').replace(/\/$/, '') || '/'
}

function syntheticActor(headers = {}) {
  const entries = Object.entries(headers || {})
  const match = entries.find(([key]) => key.toLowerCase() === TEST_ACTOR_HEADER.toLowerCase())
  const value = match?.[1]
  return Array.isArray(value) ? value[0] : value
}

function playgroundHeaders(headers) {
  const actor = syntheticActor(headers)
  return actor ? { [TEST_ACTOR_HEADER]: String(actor) } : {}
}

function decisionResolutionSchemaError(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return 'request body must be an object'
  const required = ['expected_revision', 'idempotency_key', 'answer', 'rationale']
  if (Object.keys(body).length !== required.length || required.some((key) => !(key in body))) {
    return 'request body must contain exactly expected_revision, idempotency_key, answer, and rationale'
  }
  if (!Number.isInteger(body.expected_revision) || body.expected_revision < 1) return 'expected_revision must be an integer greater than or equal to 1'
  if (typeof body.idempotency_key !== 'string' || body.idempotency_key.length < 1 || body.idempotency_key.length > 128) {
    return 'idempotency_key must be a string with length 1 through 128'
  }
  if (typeof body.answer !== 'string' || body.answer.length < 1 || body.answer.length > 2000) {
    return 'answer must be a string with length 1 through 2000'
  }
  if (typeof body.rationale !== 'string' || body.rationale.length < 1 || body.rationale.length > 4000) {
    return 'rationale must be a string with length 1 through 4000'
  }
  return null
}

function hasCredentialBearingResolutionValue(body) {
  return CREDENTIAL_BEARING_PATTERN.test(body.idempotency_key) || CREDENTIAL_BEARING_PATTERN.test(body.rationale)
}

export function playgroundResponse(method, rawPath, body = {}, headers = {}) {
  const path = cleanPath(rawPath)
  if (method === 'GET') {
    if (GET_ROUTES.has(path)) return { status: 200, body: GET_ROUTES.get(path) }
    const job = path.match(/^\/jobs\/(\d+)$/)
    if (job) {
      const found = JOBS.find(({ id }) => id === Number(job[1]))
      return found ? { status: 200, body: found } : { status: 404, body: { detail: 'not found' } }
    }
  }

  if (method === 'POST' && (path === '/jobs' || path === '/jobs/sweep')) {
    return {
      status: 200,
      body: {
        id: 99,
        run_name: body.run_name || 'playground-run',
        status: 'queued',
        playground: true,
        persistent: false,
      },
    }
  }
  if (method === 'POST' && path === '/designer/circuit') {
    return { status: 200, body: DESIGNER_VALIDATION }
  }
  if (method === 'POST' && path === '/discover/arxiv/scan') {
    return { status: 200, body: ARXIV_SCAN }
  }
  if (method === 'POST' && path === '/decisions/decision-control-scope/resolve') {
    if (syntheticActor(headers) !== HUMAN_REVIEWER) {
      return { status: 403, body: { detail: `Synthetic decision resolution requires ${TEST_ACTOR_HEADER}: ${HUMAN_REVIEWER}.` } }
    }
    const schemaError = decisionResolutionSchemaError(body)
    if (schemaError) {
      return { status: 422, body: { detail: schemaError } }
    }
    if (hasCredentialBearingResolutionValue(body)) {
      return { status: 409, body: { detail: 'credential-bearing values are not accepted' } }
    }
    if (body.expected_revision !== 1) {
      return { status: 409, body: { detail: 'stale synthetic decision revision' } }
    }
    if (!PLAYGROUND_DECISIONS[0].options.includes(body.answer)) {
      return { status: 409, body: { detail: 'synthetic decision answer must be one of its options' } }
    }
    return {
      status: 200,
      body: {
        ...PLAYGROUND_DECISIONS[0], revision: 2, status: 'resolved', answer: body.answer,
        rationale: body.rationale, resolver_actor_id: 'actor.human.reviewer',
        allowed_actions: [], actionable_next_state: null,
      },
    }
  }
  if (method === 'POST' && /^\/jobs\/\d+\/cancel$/.test(path)) {
    return { status: 200, body: { status: 'cancelled', playground: true, persistent: false } }
  }

  return { status: 404, body: { detail: 'not found in the UAT playground' } }
}

export default function handler(req, res) {
  const result = playgroundResponse(req.method, req.query.path, req.body, playgroundHeaders(req.headers))
  res.setHeader('Cache-Control', 'no-store')
  res.setHeader('X-StateVector-Mode', 'playground')
  return res.status(result.status).json(result.body)
}
