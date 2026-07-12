// Deterministic mock backend for the E2E tests. The app fetches `/api/...`;
// each test installs these stubs via page.route so no FastAPI/GPU is needed and
// runs stay hermetic. Shapes mirror qllm/dashboard/openapi.json.

export const JOBS = [
  { id: 7, run_name: 'qrnn-s42', status: 'running', comparison_role: 'candidate', preset_id: 'quantum-ffn-4q', dataset_name: 'monitored_ising', seed: 42, steps: 2000, model_family: 'qrnn', analogue_state: 'linked', device_target: 'cpu', comparison_state: 'available', uses_quantum: true, claim: { label: 'paired empirical' } },
  { id: 8, run_name: 'gru-s42', status: 'done', comparison_role: 'analogue', preset_id: 'classical-small', dataset_name: 'monitored_ising', seed: 42, steps: 2000, model_family: 'gru', analogue_state: 'none', device_target: 'cpu', comparison_state: 'none', uses_quantum: false },
]

export const STATUS = { worker: 'CPU · active', gpu_available: false, queued: 1, running: 1, runs: 312 }

export const OVERVIEW = { running: 1, queued: 1, done: 1, failed: 0, jobs: JOBS, verdicts: [], hypotheses: [], interpretation_warnings: [] }

export const PRESETS = [
  { id: 'quantum-ffn-4q', label: 'Quantum FFN 4q', kind: 'quantum', cost: 'light', summary: 'Quantum FFN block', architecture: 'ffn', quantum_role: 'ffn', classical_analogue: { label: 'Classical FFN twin', analogue_preset_id: 'classical-small', reason: 'Curated classical twin.' }, quantum_controls: { enabled: true, fields: [] }, defaults: { steps: 2000, eval_every: 100, run_name: 'quantum-ffn' } },
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
    ...overrides,
  }
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname.replace(/^\/api/, '')
    if (path.startsWith('/stream/')) return route.abort() // SSE → app falls back to polling
    if (route.request().method() === 'POST' && path === '/jobs') {
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 99, run_name: 'queued', status: 'queued' }) })
    }
    const body = table[path]
    if (body === undefined || body === null) {
      return route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'not found' }) })
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
  })
}
