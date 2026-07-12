// Deterministic mock backend for the E2E tests. The app fetches `/api/...`;
// each test installs these stubs via page.route so no FastAPI/GPU is needed and
// runs stay hermetic. Shapes mirror qllm/dashboard/openapi.json.

export const JOBS = [
  { id: 7, run_name: 'qrnn-s42', status: 'running', comparison_role: 'candidate', preset_id: 'quantum-ffn-4q', dataset_name: 'monitored_ising', seed: 42, steps: 2000, model_family: 'qrnn', analogue_state: 'linked', device_target: 'cpu', comparison_state: 'available', uses_quantum: true, claim: { label: 'paired empirical' } },
  { id: 8, run_name: 'gru-s42', status: 'done', comparison_role: 'analogue', preset_id: 'classical-small', dataset_name: 'monitored_ising', seed: 42, steps: 2000, model_family: 'gru', analogue_state: 'none', device_target: 'cpu', comparison_state: 'none', uses_quantum: false },
  { id: 9, run_name: 'qattn-s77', status: 'error', comparison_role: 'candidate', preset_id: 'quantum-attn', dataset_name: 'contextual', seed: 77, steps: 2000, model_family: 'qattn', analogue_state: 'none', device_target: 'cpu', comparison_state: 'missing', uses_quantum: true },
  { id: 10, run_name: 'tsq-s11', status: 'queued', comparison_role: 'candidate', preset_id: 'quantum-ffn-4q', dataset_name: 'contextual', seed: 11, steps: 2000, model_family: 'qffn', analogue_state: 'none', device_target: 'cpu', comparison_state: 'none', uses_quantum: true },
]

export const STATUS = { worker: 'CPU · active', gpu_available: false, queued: 1, running: 1, runs: 312 }

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
  history: [VERDICTS.snapshots[0]],
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

// Install stubs. Pass overrides to change/absent a route (set to null → 404).
export async function mockApi(page, overrides = {}) {
  const table = {
    '/lab/overview': OVERVIEW,
    '/jobs': JOBS,
    '/status': STATUS,
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
    const body = table[path]
    if (body === undefined || body === null) {
      return route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'not found' }) })
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
  })
}
