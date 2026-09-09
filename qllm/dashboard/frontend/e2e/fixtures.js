// Deterministic mock backend for the E2E tests. The app fetches `/api/...`;
// each test installs these stubs via page.route so no FastAPI/GPU is needed and
// runs stay hermetic. Shapes mirror qllm/dashboard/openapi.json.
import { ATLAS_ONTOLOGY } from './atlasOntology.fixture.js'
import { RESEARCH_WORLD_SNAPSHOT } from '../src/lib/researchWorld.js'
export { ATLAS_ONTOLOGY }

export const JOBS = [
  { id: 7, run_name: 'qrnn-s42', status: 'running', comparison_role: 'candidate', preset_id: 'quantum-ffn-4q', dataset_name: 'monitored_ising', seed: 42, steps: 2000, model_family: 'qrnn', analogue_state: 'linked', device_target: 'cpu', comparison_state: 'available', uses_quantum: true, claim: { label: 'paired empirical' } },
  { id: 8, run_name: 'gru-s42', status: 'done', comparison_role: 'analogue', preset_id: 'classical-small', dataset_name: 'monitored_ising', seed: 42, steps: 2000, model_family: 'gru', analogue_state: 'none', device_target: 'cpu', comparison_state: 'none', uses_quantum: false },
  { id: 9, run_name: 'qattn-s77', status: 'error', comparison_role: 'candidate', preset_id: 'quantum-attn', dataset_name: 'contextual', seed: 77, steps: 2000, model_family: 'qattn', analogue_state: 'none', device_target: 'cpu', comparison_state: 'missing', uses_quantum: true },
  { id: 10, run_name: 'tsq-s11', status: 'queued', comparison_role: 'candidate', preset_id: 'quantum-ffn-4q', dataset_name: 'contextual', seed: 11, steps: 2000, model_family: 'qffn', analogue_state: 'none', device_target: 'cpu', comparison_state: 'none', uses_quantum: true },
]

export const STATUS = { worker: 'CPU · active', gpu_available: false, queued: 1, running: 1, runs: 312 }

export const RESULTSDB_RUN_IDS = ['11111111-1111-4111-8111-111111111111']

export const WORK = [
  {
    id: 'work-controls', revision: 7, state: 'ready_for_human', owner_actor_id: null,
    lease_expires_at: null, allowed_actions: [], blocking_overlap_ids: [],
    blocking_decision_ids: [], actionable_next_state: null, provenance_links: [],
    payload: {
      title: 'Map matched controls', question: 'Which controls match the circuit?',
      objective: 'Produce one bounded, reviewable control map.', scope: 'Readout comparison',
      stop_rule: 'Stop after the control map.', acceptance_evidence: ['A provisional map with exact sources.'],
      domain: 'quantum', requested_capabilities: ['literature:read'],
      research_signature: {
        question: 'which controls match the circuit?', system: 'statevector simulator',
        method: 'matched control map', dataset: 'synthetic fixture', metric: 'control coverage',
        comparator: 'matched classical readout', regime: 'small cpu-only', intended_evidence_level: 'diagnostic',
      },
    },
  },
  {
    id: 'work-noise', revision: 5, state: 'revision_requested', owner_actor_id: null,
    lease_expires_at: null, allowed_actions: [], blocking_overlap_ids: [], blocking_decision_ids: [],
    actionable_next_state: null, provenance_links: [],
    payload: {
      title: 'Survey noise assumptions', question: 'Which assumptions are explicit?',
      objective: 'Record assumptions without making a performance claim.', scope: 'Simulator studies',
      stop_rule: 'Stop after each source is noted.', acceptance_evidence: ['Source-linked assumption notes.'],
      domain: 'quantum', requested_capabilities: ['literature:read'],
      research_signature: {
        question: 'which assumptions are explicit?', system: 'statevector simulator', method: 'literature survey',
        dataset: 'synthetic fixture', metric: 'assumption coverage', comparator: 'not applicable',
        regime: 'small cpu-only', intended_evidence_level: 'untested',
      },
    },
  },
  {
    id: 'work-handoff', revision: 5, state: 'ready_for_human', owner_actor_id: null,
    lease_expires_at: null, allowed_actions: [], blocking_overlap_ids: [], blocking_decision_ids: [],
    actionable_next_state: null, provenance_links: [],
    payload: {
      title: 'Review bounded handoff', question: 'Is the provenance complete for a human handoff?',
      objective: 'Expose the reviewed handoff state.', scope: 'Coordination fixture',
      stop_rule: 'Stop after provenance review.', acceptance_evidence: ['A provisional reviewed handoff.'],
      domain: 'quantum', requested_capabilities: ['literature:read'],
      research_signature: {
        question: 'is the provenance complete?', system: 'statevector simulator', method: 'provenance review',
        dataset: 'synthetic fixture', metric: 'record completeness', comparator: 'not applicable',
        regime: 'small cpu-only', intended_evidence_level: 'diagnostic',
      },
    },
  },
  {
    id: 'work-overlap', revision: 1, state: 'available', owner_actor_id: null,
    lease_expires_at: null, allowed_actions: [], blocking_overlap_ids: ['prior-noise'], blocking_decision_ids: [],
    actionable_next_state: null, provenance_links: [],
    payload: {
      title: 'Resolve overlap warning', question: 'Does this duplicate a prior noise survey?',
      objective: 'Keep unresolved exact overlap visible.', scope: 'Coordination fixture',
      stop_rule: 'Stop after classification.', acceptance_evidence: ['A provisional classification.'],
      domain: 'quantum', requested_capabilities: ['literature:read'],
      research_signature: { question: 'does this duplicate a prior noise survey?', system: 'statevector simulator', method: 'overlap review', dataset: 'synthetic fixture', metric: 'classification', comparator: 'not applicable', regime: 'small cpu-only', intended_evidence_level: 'diagnostic' },
    },
  },
  {
    id: 'work-decision', revision: 4, state: 'running', owner_actor_id: 'actor.agent.delta',
    lease_expires_at: 1788347460, allowed_actions: [], blocking_overlap_ids: [], blocking_decision_ids: ['decision-controls'],
    actionable_next_state: null, provenance_links: [],
    payload: {
      title: 'Choose the next control family', question: 'Which bounded control should be reviewed next?',
      objective: 'Make the human decision due state visible.', scope: 'Coordination fixture',
      stop_rule: 'Stop after a decision is recorded.', acceptance_evidence: ['A provisional decision record.'],
      domain: 'quantum', requested_capabilities: ['literature:read'],
      research_signature: { question: 'which bounded control should be reviewed next?', system: 'statevector simulator', method: 'decision request', dataset: 'synthetic fixture', metric: 'decision completeness', comparator: 'not applicable', regime: 'small cpu-only', intended_evidence_level: 'diagnostic' },
    },
  },
  {
    id: 'work-active', revision: 3, state: 'running', owner_actor_id: 'actor.agent.gamma',
    lease_expires_at: 1788350400, allowed_actions: [], blocking_overlap_ids: [], blocking_decision_ids: [],
    actionable_next_state: null, provenance_links: [],
    payload: {
      title: 'Run bounded provenance review', question: 'Which anchors remain to be checked?',
      objective: 'Show an active agent lease without blockers.', scope: 'Coordination fixture',
      stop_rule: 'Stop after anchor review.', acceptance_evidence: ['A provisional anchor checklist.'],
      domain: 'quantum', requested_capabilities: ['literature:read'],
      research_signature: { question: 'which anchors remain to be checked?', system: 'statevector simulator', method: 'anchor review', dataset: 'synthetic fixture', metric: 'anchor completeness', comparator: 'not applicable', regime: 'small cpu-only', intended_evidence_level: 'diagnostic' },
    },
  },
]

export const DECISIONS = [
  {
    id: 'decision-controls', work_id: 'work-decision', revision: 1, status: 'open',
    question: 'Which control family should be prioritised?', options: ['Linear readout', 'Nonlinear readout'],
    required_human_capability: 'decision:resolve', answer: null, rationale: null, resolver_actor_id: null,
    allowed_actions: ['resolve'], blocking_decision_ids: [], actionable_next_state: 'resolved',
    provenance_links: [{ relation: 'attached_to', object_type: 'work', object_id: 'work-decision' }],
    request_payload: {
      question: 'Which control family should be prioritised?', options: ['Linear readout', 'Nonlinear readout'],
      recommendation: 'Linear readout', supporting_evidence_ids: ['submission-controls'],
      uncertainty: 'The current fixture does not distinguish nonlinear readout capacity.',
      consequences: ['The next correction will remain scoped to one control family.'],
      reversible: true, expires_at: '2100-01-01T00:00:00Z', required_human_capability: 'decision:resolve',
    },
  },
  {
    id: 'decision-controls-history', work_id: 'work-controls', revision: 2, status: 'resolved',
    question: 'Which control family was prioritised for the completed review?', options: ['Linear readout', 'Nonlinear readout'],
    required_human_capability: 'decision:resolve', answer: 'Linear readout', rationale: 'The historical review stayed bounded to the matched linear control.', resolver_actor_id: 'actor.human.reviewer',
    allowed_actions: [], blocking_decision_ids: [], actionable_next_state: null,
    provenance_links: [{ relation: 'attached_to', object_type: 'work', object_id: 'work-controls' }],
    request_payload: {
      question: 'Which control family was prioritised for the completed review?', options: ['Linear readout', 'Nonlinear readout'],
      recommendation: 'Linear readout', supporting_evidence_ids: ['submission-controls'],
      uncertainty: 'The synthetic comparison remains provisional.', consequences: ['The completed review remained scoped to one control family.'],
      reversible: true, expires_at: '2026-09-02T10:10:00Z', required_human_capability: 'decision:resolve',
    },
  },
]

export const WORK_DETAIL = {
  ...WORK[0],
  events: [
    { id: 1, work_id: 'work-controls', revision: 1, event_type: 'proposed', actor_id: 'actor.agent.alpha', capability: 'work:propose', previous_state: null, next_state: 'available', payload: WORK[0].payload, created_at: 1788343200 },
    { id: 2, work_id: 'work-controls', revision: 2, event_type: 'claimed', actor_id: 'actor.agent.alpha', capability: 'work:claim', previous_state: 'available', next_state: 'claimed', payload: { lease_seconds: 3600 }, created_at: 1788343260 },
    { id: 3, work_id: 'work-controls', revision: 3, event_type: 'started', actor_id: 'actor.agent.alpha', capability: 'work:start', previous_state: 'claimed', next_state: 'running', payload: {}, created_at: 1788343320 },
    { id: 4, work_id: 'work-controls', revision: 4, event_type: 'decision_requested', actor_id: 'actor.agent.alpha', capability: 'decision:request', previous_state: 'running', next_state: 'running', payload: { decision_id: 'decision-controls-history' }, created_at: 1788343380 },
    { id: 5, work_id: 'work-controls', revision: 5, event_type: 'decision_resolved', actor_id: 'actor.human.reviewer', capability: 'decision:resolve', previous_state: 'running', next_state: 'running', payload: { decision_id: 'decision-controls-history' }, created_at: 1788343440 },
    { id: 6, work_id: 'work-controls', revision: 6, event_type: 'claim_proposed', actor_id: 'actor.agent.alpha', capability: 'claim:propose', previous_state: 'running', next_state: 'submitted', payload: { claim_id: 'claim-controls', submission_id: 'submission-controls', claim_hash: '3'.repeat(64) }, created_at: 1788343500 },
    { id: 7, work_id: 'work-controls', revision: 7, event_type: 'claim_reviewed', actor_id: 'actor.agent.beta', capability: 'review:submit', previous_state: 'submitted', next_state: 'ready_for_human', payload: { review_id: 'review-controls', claim_id: 'claim-controls', outcome: 'verified_for_handoff' }, created_at: 1788343560 },
  ],
  revisions: [
    { work_id: 'work-controls', revision: 1, state: 'available', payload: WORK[0].payload, owner_actor_id: null, lease_expires_at: null, created_at: 1788343200 },
    { work_id: 'work-controls', revision: 2, state: 'claimed', payload: WORK[0].payload, owner_actor_id: 'actor.agent.alpha', lease_expires_at: 1788346860, created_at: 1788343260 },
    { work_id: 'work-controls', revision: 3, state: 'running', payload: WORK[0].payload, owner_actor_id: 'actor.agent.alpha', lease_expires_at: 1788346860, created_at: 1788343320 },
    { work_id: 'work-controls', revision: 4, state: 'running', payload: WORK[0].payload, owner_actor_id: 'actor.agent.alpha', lease_expires_at: 1788346860, created_at: 1788343380 },
    { work_id: 'work-controls', revision: 5, state: 'running', payload: WORK[0].payload, owner_actor_id: 'actor.agent.alpha', lease_expires_at: 1788346860, created_at: 1788343440 },
    { work_id: 'work-controls', revision: 6, state: 'submitted', payload: WORK[0].payload, owner_actor_id: null, lease_expires_at: null, created_at: 1788343500 },
    { work_id: 'work-controls', revision: 7, state: 'ready_for_human', payload: WORK[0].payload, owner_actor_id: null, lease_expires_at: null, created_at: 1788343560 },
  ],
}

export const WORK_DECISION_DETAIL = {
  ...WORK[4],
  events: [
    { id: 21, work_id: 'work-decision', revision: 1, event_type: 'proposed', actor_id: 'actor.agent.delta', capability: 'work:propose', previous_state: null, next_state: 'available', payload: WORK[4].payload, created_at: 1788343800 },
    { id: 22, work_id: 'work-decision', revision: 2, event_type: 'claimed', actor_id: 'actor.agent.delta', capability: 'work:claim', previous_state: 'available', next_state: 'claimed', payload: { lease_seconds: 3600 }, created_at: 1788343860 },
    { id: 23, work_id: 'work-decision', revision: 3, event_type: 'started', actor_id: 'actor.agent.delta', capability: 'work:start', previous_state: 'claimed', next_state: 'running', payload: {}, created_at: 1788343920 },
    { id: 24, work_id: 'work-decision', revision: 4, event_type: 'decision_requested', actor_id: 'actor.agent.delta', capability: 'decision:request', previous_state: 'running', next_state: 'running', payload: { decision_id: 'decision-controls' }, created_at: 1788343980 },
  ],
  revisions: [
    { work_id: 'work-decision', revision: 1, state: 'available', payload: WORK[4].payload, owner_actor_id: null, lease_expires_at: null, created_at: 1788343800 },
    { work_id: 'work-decision', revision: 2, state: 'claimed', payload: WORK[4].payload, owner_actor_id: 'actor.agent.delta', lease_expires_at: 1788347460, created_at: 1788343860 },
    { work_id: 'work-decision', revision: 3, state: 'running', payload: WORK[4].payload, owner_actor_id: 'actor.agent.delta', lease_expires_at: 1788347460, created_at: 1788343920 },
    { work_id: 'work-decision', revision: 4, state: 'running', payload: WORK[4].payload, owner_actor_id: 'actor.agent.delta', lease_expires_at: 1788347460, created_at: 1788343980 },
  ],
}

export const OVERLAPS_CONTROLS = {
  work_id: 'work-controls', exact_match_requires_classification: false,
  authority: 'coordination_provisional',
  candidates: [{
    id: 'prior-controls', candidate_type: 'prior_work', title: 'Synthetic matched-control reference',
    reasons: ['exact_identifier'], exact_match: true, classified: true, classification: 'replication',
  }],
}

export const OVERLAPS_DECISION = {
  work_id: 'work-decision', exact_match_requires_classification: false,
  authority: 'coordination_provisional', candidates: [],
}

export const SUBMISSION_CONTROLS = {
  id: 'submission-controls', work_id: 'work-controls', revision: 1, work_revision: 5,
  status: 'submitted', authority: 'coordination_provisional', scientific_acceptance: false,
  outcome: 'null', summary: 'The bounded synthetic comparison did not separate the readout families.',
  methods: 'Matched synthetic comparison.', controls: 'Linear classical readout.', regime: 'Small CPU-only fixture.',
  uncertainty: 'No empirical finding is claimed.', limitations: ['Synthetic fixture only.'],
  gaps: ['Human taste is needed to choose the next family.'],
  items: [
    { kind: 'supporting', statement: 'Matched-control setup was retained.', anchor_type: 'resultsdb_job', anchor_id: '7' },
    { kind: 'contradicting', statement: 'Nonlinear capacity remains unresolved.', anchor_type: 'resultsdb_study', anchor_id: '1' },
    { kind: 'null', statement: 'No separation was recorded.', anchor_type: 'resultsdb_run', anchor_id: RESULTSDB_RUN_IDS[0] },
    { kind: 'failed', statement: 'The exploratory branch did not complete.', anchor_type: 'prior_work', anchor_id: 'prior-controls' },
    { kind: 'excluded', statement: 'An unmatched comparison was excluded.', anchor_type: 'resultsdb_job', anchor_id: '8' },
  ],
  content_hash: '2'.repeat(64), predecessor_content_hash: null, correction_reason: null,
  submitted_by_actor_id: 'actor.agent.alpha', created_at: '2026-09-02T10:03:00Z',
}

export const CLAIM_CONTROLS = {
  id: 'claim-controls', submission_id: 'submission-controls', submission_revision: 1,
  work_id: 'work-controls', revision: 1, status: 'proposed', authority: 'coordination_provisional',
  scientific_acceptance: false, canonical_claim_updated: false, submission_hash: '2'.repeat(64),
  content_hash: '3'.repeat(64), statement: 'This fixture is an unreviewed null result only.',
  scope: 'Synthetic fixture and matched control only.', comparator: 'Matched linear readout.',
  regime: 'Small CPU-only fixture.', intended_evidence_level: 'diagnostic',
  uncertainty: 'No empirical finding is claimed.', limitations: ['Synthetic fixture only.'],
  predecessor_claim_content_hash: null, correction_reason: null, proposed_by_actor_id: 'actor.agent.alpha',
  created_at: '2026-09-02T10:04:00Z', work_revision: 6,
}

export const REVIEW_CONTROLS = {
  id: 'review-controls', submission_id: 'submission-controls', claim_id: 'claim-controls',
  work_id: 'work-controls', revision: 1, authority: 'coordination_provisional', scientific_acceptance: false,
  canonical_claim_updated: false, submission_hash: '2'.repeat(64), claim_hash: '3'.repeat(64),
  content_hash: '4'.repeat(64), outcome: 'verified_for_handoff',
  rationale: 'The bounded synthetic provenance is internally consistent.', checks: ['Matched-control scope retained.'],
  limitations: ['This is not scientific acceptance.'], reviewed_by_actor_id: 'actor.agent.beta',
  created_at: '2026-09-02T10:05:00Z', work_revision: 7,
}

export const LINEAGE_CONTROLS = {
  object_id: 'work-controls', work_id: 'work-controls', authority: 'coordination_provisional',
  scientific_acceptance: false, canonical_claim_updated: false,
  nodes: [
    { id: 'work-controls', type: 'work', revision: 7, hash: '1'.repeat(64) },
    { id: 'prior-controls', type: 'prior_work', revision: 1, hash: '5'.repeat(64) },
    { id: 'classification-controls', type: 'classification', revision: 1, hash: '6'.repeat(64) },
    { id: '7', type: 'resultsdb_job', revision: null, hash: null },
    { id: '8', type: 'resultsdb_job', revision: null, hash: null },
    { id: '1', type: 'resultsdb_study', revision: null, hash: null },
    { id: RESULTSDB_RUN_IDS[0], type: 'resultsdb_run', revision: null, hash: null },
    { id: 'submission-controls', type: 'submission', revision: 1, hash: '2'.repeat(64) },
    { id: 'claim-controls', type: 'claim', revision: 1, hash: '3'.repeat(64) },
    { id: 'review-controls', type: 'review', revision: 1, hash: '4'.repeat(64) },
    { id: 'decision-controls-history', type: 'decision', revision: 2, hash: '7'.repeat(64) },
  ],
  edges: [
    { from: 'prior-controls', to: 'work-controls', relation: 'informs' },
    { from: 'classification-controls', to: 'work-controls', relation: 'classifies' },
    { from: 'classification-controls', to: 'prior-controls', relation: 'classifies' },
    { from: 'submission-controls', to: 'work-controls', relation: 'submitted_for' },
    { from: 'submission-controls', to: '7', relation: 'anchors' },
    { from: 'submission-controls', to: '1', relation: 'anchors' },
    { from: 'submission-controls', to: RESULTSDB_RUN_IDS[0], relation: 'anchors' },
    { from: 'submission-controls', to: 'prior-controls', relation: 'anchors' },
    { from: 'submission-controls', to: '8', relation: 'anchors' },
    { from: 'claim-controls', to: 'submission-controls', relation: 'proposes_from' },
    { from: 'review-controls', to: 'claim-controls', relation: 'reviews' },
    { from: 'decision-controls-history', to: 'work-controls', relation: 'decides' },
  ],
}

export const LINEAGE_DECISION = {
  object_id: 'work-decision', work_id: 'work-decision', authority: 'coordination_provisional',
  scientific_acceptance: false, canonical_claim_updated: false,
  nodes: [
    { id: 'work-decision', type: 'work', revision: 4, hash: '8'.repeat(64) },
    { id: 'decision-controls', type: 'decision', revision: 1, hash: '9'.repeat(64) },
  ],
  edges: [{ from: 'decision-controls', to: 'work-decision', relation: 'decides' }],
}

export const OVERVIEW = { running: 1, queued: 1, done: 1, failed: 0, jobs: JOBS, verdicts: [], hypotheses: [], interpretation_warnings: [] }

export const PRESETS = [
  { id: 'quantum-ffn-4q', label: 'Quantum FFN 4q', kind: 'quantum', cost: 'light', summary: 'Quantum FFN block', architecture: 'ffn', quantum_role: 'ffn', classical_analogue: { label: 'Classical FFN twin', analogue_preset_id: 'classical-small', reason: 'Curated classical twin.' }, quantum_controls: { enabled: true, summary: 'Tune the quantum FFN circuit', warning: 'Larger circuits run slower.', fields: [{ key: 'n_qubits', label: 'Qubits', min: 2, max: 8, gpu_max: 12, step: 1, default: 4 }, { key: 'n_circuit_layers', label: 'Depth', min: 1, max: 4, gpu_max: 8, step: 1, default: 2 }] }, defaults: { steps: 2000, eval_every: 100, run_name: 'quantum-ffn' } },
  { id: 'classical-small', label: 'Classical small', kind: 'classical', cost: 'light', summary: 'Classical baseline', architecture: 'mlp', quantum_role: 'none', classical_analogue: null, quantum_controls: { enabled: false, fields: [] }, defaults: { steps: 50, eval_every: 10, run_name: 'classical-small' } },
]

export const DATASETS = [
  { name: 'monitored_ising', source: 'synthetic', source_type: 'quantum-native', split: 'train', n_rows: 1000 },
  { name: 'contextual', source: 'synthetic', source_type: 'quantum-native', split: 'train', n_rows: 1000 },
]

// A verdict store with a quantum-candidate AND a classical-holds (null) snapshot —
// so tests can assert claim_level vs replication_status are shown distinctly and
// that null outcomes are first-class.
export const VERDICTS = {
  snapshots: [
    { id: 101, verdict_key: 'qrnn-vs-gru', revision: 2, content_hash: 'ab12', source_kind: 'comparison', source_id: '7', claim_id: 'c-qrnn', claim_level: 'empirical', claim_status: 'candidate', replication_status: 'multi_seed_single_instance', assessment_level: 'descriptive', assessment_status: 'unassigned', created_ts: '2026-07-12T00:00:00Z' },
    { id: 102, verdict_key: 'qffn-vs-classical', revision: 1, content_hash: 'cd34', source_kind: 'comparison', source_id: '9', claim_id: 'c-qffn', claim_level: 'none', claim_status: 'refuted', replication_status: 'single_task_instance', assessment_level: null, assessment_status: 'negative', created_ts: '2026-07-12T00:00:00Z' },
  ],
}

export const WORKSPACE_7 = {
  job: JOBS[0],
  curve: { val_ppl: [{ step: 0, value: 9.2 }, { step: 100, value: 5.1 }, { step: 200, value: 3.4 }], grad_norm_ratio: [{ step: 0, value: 1.1 }, { step: 100, value: 0.9 }] },
  final_run: { val_ppl: 3.39, val_loss: 1.2, wall_seconds: 852, n_params: 18100 },
  comparison: {
    available: true,
    candidate: { final_run: { val_ppl: 3.39, val_loss: 1.2, wall_seconds: 852, n_params: 18100 }, curve: { val_ppl: [{ step: 0, value: 9.2 }, { step: 200, value: 3.39 }] } },
    baseline: { final_run: { val_ppl: 3.55, val_loss: 1.3, wall_seconds: 228, n_params: 18400 }, curve: { val_ppl: [{ step: 0, value: 9.4 }, { step: 200, value: 3.55 }] } },
    deltas: { val_ppl: -0.16, wall_seconds: 624, n_params: -300 },
    fairness: { same_dataset: true, same_seed: true, same_steps: true, same_eval_interval: true, same_device_target: true, role_validation: true, parameter_delta_ratio: 0.984 },
    evidence_ladder: { label: 'paired empirical', claim_level: 'empirical', reason: 'candidate leads its matched control', met_count: 4, total_count: 8, steps: [{ key: 'matched_baseline', label: 'Matched baseline', ok: true }, { key: 'multi_seed', label: 'Multiple seeds', ok: false, detail: 'single seed' }] },
    interpretation_warnings: [{ code: 'single_seed', severity: 'warning', title: 'One pair', message: 'Single seed per arm.' }],
  },
  interpretation_warnings: [{ code: 'single_seed', severity: 'warning', title: 'One pair', message: 'Single seed per arm.' }],
}

export const DIAGNOSTICS_7 = {
  job: { id: 7, run_name: 'qrnn-s42', status: 'running', group_id: 'scale-grp' },
  diagnostics: {
    gradient_variance: { status: 'measured', value: { grad_var_first_param: 2e-3, grad_var_mean: 1.2e-3, grad_var_max: 3e-3 }, source: 'summary', reason: null, provenance: {} },
    parameter_shift_gradient_snr: { status: 'measured', value: { median_snr: 8.4, mean_snr: 9.1 }, source: 'diagnostics', reason: null, provenance: {} },
    expressibility_kl: { status: 'measured', value: 0.18, source: 'summary', reason: null, provenance: {} },
    meyer_wallach_q: { status: 'measured', value: 0.61, source: 'summary', reason: null, provenance: {} },
    scaling_fit: { status: 'measured', value: { log_var_slope: -0.34, log_var_intercept: 0.1, variance_decay_factor_per_qubit: 0.71, exponential_decay_detected: true }, source: 'scaling', reason: null, provenance: {} },
  },
  interpretation_warnings: [{ code: 'diagnostics_scope', severity: 'warning', title: 'Diagnostics scope', message: 'These are mechanism observations, not evidence of quantum advantage.' }],
}

export const MODEL_GRAPH_7 = {
  nodes: [
    { id: 'tokens', label: 'Tokens', kind: 'input' },
    { id: 'embed', label: 'Classical Embedding', kind: 'classical', meta: { component_type: 'embedding' } },
    { id: 'qffn', label: 'Quantum FFN', kind: 'quantum', meta: { resource: { n_qubits: 4, n_circuit_layers: 2, backend: 'pennylane' } } },
    { id: 'head', label: 'Output Head', kind: 'classical' },
    { id: 'out', label: 'Logits', kind: 'output' },
  ],
  edges: [['tokens', 'embed'], ['embed', 'qffn'], ['qffn', 'head'], ['head', 'out']],
  summary: { arch: 'qffn', uses_quantum: true, model_family: 'qffn' },
}

export const MODEL_TESTS_7 = {
  job: { id: 7, run_name: 'qrnn-s42', status: 'running' },
  summary: { quantum_diagnostics: { grad_var_mean: 1.2e-3, meyer_wallach_q: 0.61, expressibility_kl: 0.18, availability: {} } },
  artifacts: {}, supported_tests: { summary_review: true, prompt_generation: false }, unsupported_reasons: [],
}

export const SCALING_GRP = {
  points: [
    { job: { id: 7, run_name: 'q4' }, status: 'done', n_qubits: 4, n_circuit_layers: 2, scale: 1, val_ppl: 5.8, val_loss: 1.6, wall_seconds: 12, n_params: 1000 },
    { job: { id: 11, run_name: 'q6' }, status: 'done', n_qubits: 6, n_circuit_layers: 2, scale: 1.5, val_ppl: 5.2, val_loss: 1.5, wall_seconds: 30, n_params: 2000 },
    { job: { id: 12, run_name: 'q8' }, status: 'done', n_qubits: 8, n_circuit_layers: 2, scale: 2, val_ppl: 4.9, val_loss: 1.4, wall_seconds: 70, n_params: 3200 },
  ],
  best: { n_qubits: 8, n_circuit_layers: 2, val_ppl: 4.9, wall_seconds: 70, n_params: 3200 },
  complete_count: 3, total_count: 3, protocol_warnings: [],
}

export const ARXIV_SCAN = {
  request: { topic: 'quant-ph', max_results: 10 },
  papers: [
    { arxiv_id: '2503.12345', title: 'Reuploading circuits resist barren plateaus', authors: ['Larocca', 'Cerezo'], categories: ['quant-ph', 'cs.LG'], published: '2025-03-01', updated: '2025-03-02', abs_url: 'https://arxiv.org/abs/2503.12345', version: 1 },
    { arxiv_id: '2101.11111', title: 'Quantum models as random features', authors: ['Schuld'], categories: ['quant-ph'], published: '2024-01-01', updated: '2024-01-01', abs_url: 'https://arxiv.org/abs/2101.11111', version: 2 },
  ],
  quota_used: 2, quota_remaining: 48, quota_limit: 50, capabilities: null,
}

// A full VerdictSnapshotDetail for /verdicts/{id} (snapshot + history).
export const VERDICT_DETAIL_101 = {
  snapshot: {
    ...VERDICTS.snapshots[0],
    source_job_id: 7,
    scorecard: { dimensions: { metric_type: 'ppl', deltas: { val_ppl: -0.16, wall_seconds: 624 } } },
    fairness: { same_dataset: true, same_seed: true, same_steps: true },
    controls: { frozen_circuit: true, random_feature: false },
    caveats: [{ code: 'single_seed', title: 'One pair', message: 'Single seed per arm.' }],
    evidence: { evidence_ladder: { steps: [{ key: 'matched_baseline', label: 'Matched baseline', ok: true }, { key: 'multi_seed', label: 'Multiple seeds', ok: false, detail: 'single seed' }] } },
    diagnostics: {},
    schema_version: 1,
  },
  // Append-only ledger: rev 1 was a null result, rev 2 promoted to empirical
  // after a second seed replicated. Both are kept — the timeline shows the change.
  history: [
    { id: 100, verdict_key: 'qrnn-vs-gru', revision: 1, content_hash: '77ff', claim_level: 'none', claim_status: 'candidate', replication_status: 'single_task_instance', created_ts: '2026-07-10T00:00:00Z' },
    VERDICTS.snapshots[0],
  ],
}

export const STUDIES = [
  { id: 1, name: 'qffn-multiseed', research_question: 'Does the quantum FFN hold across seeds?', evidence: { label: 'paired empirical', fair_pairs: 4, wins: 3, mean_delta_val_ppl: -0.12 } },
]

export const STUDY_1 = {
  id: 1, name: 'qffn-multiseed', research_question: 'Does the quantum FFN hold across seeds?',
  evidence: {
    label: 'paired empirical', reason: 'candidate leads across seeds',
    fair_pairs: 4, complete_pairs: 5, wins: 3, mean_delta_val_ppl: -0.12, std_delta_val_ppl: 0.08, rerun_required_pairs: 1,
    ladder: [{ key: 'multi_seed', label: 'Multiple seeds', ok: true }, { key: 'fair_protocol', label: 'Fair protocol', ok: true }],
    comparisons: [
      { delta_val_ppl: -0.2, fair: true, rerun_required: false, cell: 'q4/d2' },
      { delta_val_ppl: -0.1, fair: true, rerun_required: false, cell: 'q6/d2' },
      { delta_val_ppl: 0.05, fair: true, rerun_required: false, cell: 'q8/d2' },
      { delta_val_ppl: -0.15, fair: true, rerun_required: false, cell: 'q4/d3' },
    ],
  },
  jobs: [
    { id: 201, study_sweep: { n_qubits: 4, n_circuit_layers: 2 }, final_run: { val_ppl: 3.4 }, status: 'done' },
    { id: 202, study_sweep: { n_qubits: 6, n_circuit_layers: 2 }, final_run: { val_ppl: 3.5 }, status: 'done' },
    { id: 203, study_sweep: { n_qubits: 4, n_circuit_layers: 2 }, final_run: { val_ppl: 3.45 }, status: 'done' },
  ],
  interpretation_warnings: [{ code: 'single_task_instance', title: 'One task instance', message: 'Multi-seed, single task instance.' }],
}

// Per-seed workspaces for STUDY_1's runs — distinct val_ppl trajectories so the
// seed-band aggregates a real min–max spread over steps.
export const STUDY_1_WORKSPACES = {
  '/jobs/201/workspace': { curve: { val_ppl: [{ step: 0, value: 9.1 }, { step: 100, value: 5.0 }, { step: 200, value: 3.40 }] } },
  '/jobs/202/workspace': { curve: { val_ppl: [{ step: 0, value: 9.4 }, { step: 100, value: 5.4 }, { step: 200, value: 3.50 }] } },
  '/jobs/203/workspace': { curve: { val_ppl: [{ step: 0, value: 9.2 }, { step: 100, value: 5.2 }, { step: 200, value: 3.45 }] } },
}

export const CAPABILITIES = { metadata_only: true, full_text: false, unreviewed_preprints: true, claim_evidence_classification: false, human_review_required: true, paid_services_enabled: false, daily_cost_budget: null, llm_provider: null, embedding_provider: null, vector_store_provider: null, graph_store_provider: null, d4_human_gate_open: true }

// GET /designer/circuit — registry-backed capabilities (mirrors designer.py).
export const DESIGNER_CAPABILITIES = {
  schema_version: 1, validation_only: true, side_effect_free: true, client_estimates_authoritative: false,
  choices: {
    architecture: ['qrnn'],
    circuit_ansatz: ['hardware_efficient', 'reuploading'],
    qrnn_only_ansatz: ['ising'],
    backend: ['pennylane', 'tensorcircuit', 'tensorcircuit_mps'],
    readout: ['z', 'zz'],
  },
  defaults: { ansatz: 'reuploading', n_qubits: 4, n_circuit_layers: 2, backend: 'pennylane', readout: 'z', architecture: null, device: 'default.qubit', diff_method: 'backprop', shots: null, mps_max_bond_dimension: null },
  constraints: {
    n_qubits: { minimum: 1, maximum: 12 },
    n_circuit_layers: { minimum: 1, maximum: 8 },
    qrnn_only_ansatz_requires_architecture: 'qrnn',
    tensorcircuit_mps_requires: ['mps_max_bond_dimension'],
  },
  warnings: [
    'Validation never constructs a circuit, model, backend, job, or device.',
    'Circuit properties and diagnostics are not evidence of quantum advantage.',
  ],
}

// POST /designer/circuit — a successful validation (derived values authoritative).
export const DESIGNER_VALIDATION = {
  schema_version: 1, valid: true, validation_only: true,
  spec: { ansatz: 'hardware_efficient', n_qubits: 4, n_circuit_layers: 2, backend: 'pennylane', readout: 'z', architecture: null, device: 'default.qubit', diff_method: 'backprop', shots: null, mps_max_bond_dimension: null },
  derived: {
    circuit_weight_shape: [2, 4, 3],
    trainable_circuit_parameters: { status: 'derived', value: 24, scope: 'Variational circuit parameters only; surrounding model excluded.' },
    readout_features: { status: 'derived', value: 4, scope: 'Per-circuit expectation-value feature width.' },
    entangling_gates: { status: 'unavailable', scope: 'Backend-level circuit decomposition.', reason: 'The canonical config selects an ansatz family, not a stable compiled gate list.' },
  },
  ignored_fields: [],
  client_estimates: { trainable_params: { supplied: 8, authoritative: false, matches_derived: false }, entangling_gates: { supplied: 6, authoritative: false } },
  warnings: [
    'Validation only: no circuit, model, backend, job, or device was constructed.',
    'The client trainable_params estimate was ignored because it does not match the registry-backed circuit parameter shape.',
  ],
}

const EVIDENCE_KINDS = new Set(['supporting', 'contradicting', 'null', 'failed', 'excluded'])
const EVIDENCE_ANCHOR_TYPES = new Set(['resultsdb_job', 'resultsdb_study', 'resultsdb_run', 'prior_work'])
const REVIEWED_WORK_STATES = new Set(['revision_requested', 'ready_for_human'])

function fixtureContract(condition, message) {
  if (!condition) throw new Error(`coordination fixture contract: ${message}`)
}

// Keep the browser's hermetic examples aligned with the API enums and lifecycle
// invariants. This is intentionally dependency-free so `npm test` catches drift.
export function assertCoordinationFixtureContract() {
  const evidenceKinds = new Set(SUBMISSION_CONTROLS.items.map((item) => item.kind))
  fixtureContract(evidenceKinds.size === EVIDENCE_KINDS.size && [...EVIDENCE_KINDS].every((kind) => evidenceKinds.has(kind)), 'submission must include every evidence kind')
  const anchors = {
    resultsdb_job: new Set(JOBS.map((job) => String(job.id))),
    resultsdb_study: new Set(STUDIES.map((study) => String(study.id))),
    resultsdb_run: new Set(RESULTSDB_RUN_IDS),
    prior_work: new Set(LINEAGE_CONTROLS.nodes.filter((node) => node.type === 'prior_work').map((node) => node.id)),
  }
  for (const item of SUBMISSION_CONTROLS.items) {
    fixtureContract(EVIDENCE_KINDS.has(item.kind), `unsupported evidence kind ${item.kind}`)
    fixtureContract(EVIDENCE_ANCHOR_TYPES.has(item.anchor_type), `unsupported evidence anchor type ${item.anchor_type}`)
    fixtureContract(anchors[item.anchor_type].has(item.anchor_id), `unresolved ${item.anchor_type} anchor ${item.anchor_id}`)
  }
  for (const work of WORK) {
    fixtureContract(typeof work.id === 'string' && Number.isInteger(work.revision), `invalid work identity for ${work.id}`)
    if (REVIEWED_WORK_STATES.has(work.state)) {
      fixtureContract(work.owner_actor_id === null && work.lease_expires_at === null, `${work.state} must clear owner and lease`)
    }
    if (work.state === 'running' || work.state === 'claimed') {
      fixtureContract(typeof work.owner_actor_id === 'string' && typeof work.lease_expires_at === 'number', `${work.state} requires an active owner and lease`)
      fixtureContract(work.blocking_overlap_ids.length === 0, `${work.state} cannot retain an unresolved overlap blocker`)
    }
    if (work.state === 'running') fixtureContract(work.revision >= (work.blocking_decision_ids.length ? 4 : 3), `${work.id} has an impossible running revision`)
    if (REVIEWED_WORK_STATES.has(work.state)) {
      const hasDecisionEvent = WORK_DETAIL.events.some((event) => event.work_id === work.id && event.event_type === 'decision_requested')
      fixtureContract(work.revision >= (hasDecisionEvent ? 6 : 5), `${work.id} has an impossible review-terminal revision`)
    }
  }
  const workIds = new Set(WORK.map((work) => work.id))
  const decisionIds = new Set(DECISIONS.map((decision) => decision.id))
  for (const decision of DECISIONS) {
    fixtureContract(workIds.has(decision.work_id), `decision ${decision.id} references missing work`)
    fixtureContract(decision.status === 'open' ? decision.revision === 1 : decision.revision === 2, `${decision.status} decision ${decision.id} has an invalid revision`)
  }
  for (const work of WORK) for (const decisionId of work.blocking_decision_ids) fixtureContract(decisionIds.has(decisionId), `work ${work.id} references missing decision ${decisionId}`)
  const lineageNodes = new Set(LINEAGE_CONTROLS.nodes.map((node) => `${node.type}:${node.id}`))
  const lineageEdges = new Set(LINEAGE_CONTROLS.edges.map((edge) => `${edge.from}:${edge.relation}:${edge.to}`))
  for (const item of SUBMISSION_CONTROLS.items) {
    fixtureContract(lineageNodes.has(`${item.anchor_type}:${item.anchor_id}`), `missing lineage node for ${item.anchor_type}:${item.anchor_id}`)
    fixtureContract(lineageEdges.has(`${SUBMISSION_CONTROLS.id}:anchors:${item.anchor_id}`), `missing submission anchor edge for ${item.anchor_id}`)
  }
  for (const event of WORK_DETAIL.events.filter((event) => event.event_type === 'decision_requested')) {
    const decision = DECISIONS.find((item) => item.id === event.payload.decision_id)
    fixtureContract(decision?.work_id === event.work_id, `decision event ${event.id} references a decision for another work item`)
  }
  for (const decision of DECISIONS.filter((item) => item.status === 'resolved')) {
    const requests = WORK_DETAIL.events.filter((event) => event.event_type === 'decision_requested' && event.payload.decision_id === decision.id)
    const resolutions = WORK_DETAIL.events.filter((event) => event.event_type === 'decision_resolved' && event.payload.decision_id === decision.id)
    fixtureContract(requests.length === 1 && resolutions.length === 1, `resolved decision ${decision.id} requires exactly one request and resolution event`)
    const [request] = requests
    const [resolution] = resolutions
    fixtureContract(request.work_id === decision.work_id && resolution.work_id === decision.work_id && resolution.revision === request.revision + 1, `resolved decision ${decision.id} has incoherent work revisions`)
  }
  for (const detail of [WORK_DETAIL, WORK_DECISION_DETAIL]) {
    for (const event of detail.events.filter((item) => item.event_type === 'decision_requested' || item.event_type === 'decision_resolved')) {
      const before = detail.revisions.find((item) => item.revision === event.revision - 1)
      const after = detail.revisions.find((item) => item.revision === event.revision)
      fixtureContract(before?.owner_actor_id === after?.owner_actor_id && before?.lease_expires_at === after?.lease_expires_at, `${event.event_type} must preserve owner and lease without a renewal event`)
    }
  }
  for (const node of LINEAGE_CONTROLS.nodes.filter((node) => node.type === 'decision')) {
    const decision = DECISIONS.find((item) => item.id === node.id)
    fixtureContract(decision?.work_id === LINEAGE_CONTROLS.work_id, `lineage decision ${node.id} belongs to another work item`)
    fixtureContract(lineageEdges.has(`${node.id}:decides:${LINEAGE_CONTROLS.work_id}`), `lineage decision ${node.id} is not attached to its work item`)
  }
  const reviewed = WORK_DETAIL.events.find((event) => event.event_type === 'claim_reviewed')
  fixtureContract(reviewed?.next_state === 'ready_for_human' && reviewed?.payload?.outcome === 'verified_for_handoff', 'verified review transition must produce the ready-for-human fixture')
  const proposed = WORK_DETAIL.events.find((event) => event.event_type === 'claim_proposed')
  const resolved = WORK_DETAIL.events.find((event) => event.event_type === 'decision_resolved')
  fixtureContract(proposed?.revision === resolved?.revision + 1 && reviewed?.revision === proposed?.revision + 1 && SUBMISSION_CONTROLS.work_revision === resolved?.revision && CLAIM_CONTROLS.work_revision === proposed?.revision && REVIEW_CONTROLS.work_revision === reviewed?.revision, 'submission, claim, and review revisions must follow the resolved decision')
  fixtureContract(WORK_DETAIL.revisions.at(-1)?.state === 'ready_for_human' && WORK_DETAIL.revisions.at(-1)?.owner_actor_id === null && WORK_DETAIL.revisions.at(-1)?.lease_expires_at === null, 'ready-for-human revision must clear owner and lease')
  fixtureContract(WORK_DECISION_DETAIL.id === DECISIONS[0].work_id && WORK_DECISION_DETAIL.events.at(-1)?.payload?.decision_id === DECISIONS[0].id, 'open decision detail must project its work event')
  fixtureContract(LINEAGE_DECISION.nodes.some((node) => node.type === 'decision' && node.id === DECISIONS[0].id) && LINEAGE_DECISION.edges.some((edge) => edge.from === DECISIONS[0].id && edge.to === DECISIONS[0].work_id && edge.relation === 'decides'), 'open decision lineage must project its work relation')
}

// Install stubs. Pass overrides to change/absent a route (set to null → 404).
export async function mockApi(page, overrides = {}) {
  assertCoordinationFixtureContract()
  const table = {
    '/lab/overview': OVERVIEW,
    '/work': WORK,
    '/decisions': DECISIONS,
    '/decisions/decision-controls': DECISIONS[0],
    '/decisions/decision-controls-history': DECISIONS[1],
    '/work/work-controls': WORK_DETAIL,
    '/work/work-controls/overlaps': OVERLAPS_CONTROLS,
    '/objects/work-controls/lineage': LINEAGE_CONTROLS,
    '/work/work-decision': WORK_DECISION_DETAIL,
    '/work/work-decision/overlaps': OVERLAPS_DECISION,
    '/objects/work-decision/lineage': LINEAGE_DECISION,
    '/evidence/submission-controls': SUBMISSION_CONTROLS,
    '/claim-proposals/claim-controls': CLAIM_CONTROLS,
    '/reviews/review-controls': REVIEW_CONTROLS,
    '/jobs': JOBS,
    '/status': STATUS,
    '/research-world': RESEARCH_WORLD_SNAPSHOT,
    '/presets': PRESETS,
    '/datasets': DATASETS,
    '/verdicts': VERDICTS,
    '/verdicts/101': VERDICT_DETAIL_101,
    '/research/capabilities': CAPABILITIES,
    '/jobs/7/workspace': WORKSPACE_7,
    '/jobs/7/diagnostics': DIAGNOSTICS_7,
    '/jobs/7/model-tests': MODEL_TESTS_7,
    '/jobs/7/model-graph': MODEL_GRAPH_7,
    '/jobs/7/comparison': WORKSPACE_7.comparison,
    '/scaling-tests/scale-grp': SCALING_GRP,
    '/studies': STUDIES,
    '/studies/1': STUDY_1,
    ...STUDY_1_WORKSPACES,
    '/atlas/ontology': ATLAS_ONTOLOGY,
    '/designer/circuit': DESIGNER_CAPABILITIES, // GET; POST handled below
    ...overrides,
  }
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname.replace(/^\/api/, '')
    const method = route.request().method()
    if (path.startsWith('/stream/')) return route.abort() // SSE → app falls back to polling
    if (method === 'POST' && path === '/jobs') {
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 99, run_name: 'queued', status: 'queued' }) })
    }
    if (method === 'POST' && path === '/discover/arxiv/scan') {
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(overrides['/discover/arxiv/scan'] ?? ARXIV_SCAN) })
    }
    if (method === 'POST' && path === '/designer/circuit') {
      // Overridable: pass { 'POST /designer/circuit': {status, body} } to test
      // a registry rejection; null on the GET key 404s both verbs.
      const custom = overrides['POST /designer/circuit']
      if (custom) return route.fulfill({ status: custom.status ?? 200, contentType: 'application/json', body: JSON.stringify(custom.body) })
      if (table['/designer/circuit'] === null) return route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'not found' }) })
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(DESIGNER_VALIDATION) })
    }
    if (method === 'POST' && path === '/decisions/decision-controls/resolve') {
      const payload = route.request().postDataJSON()
      const resolved = {
        ...DECISIONS[0], revision: 2, status: 'resolved', answer: payload.answer,
        rationale: payload.rationale, resolver_actor_id: 'actor.human.reviewer',
        allowed_actions: [], actionable_next_state: null,
      }
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(resolved) })
    }
    const body = table[path]
    if (body === undefined || body === null) {
      return route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'not found' }) })
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
  })
}
