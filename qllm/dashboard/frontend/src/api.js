const base = ''

export async function responseError(response, path) {
  let detail = `${path}: ${response.status}`
  try {
    const payload = await response.json()
    detail = payload.detail || detail
  } catch (_) {}
  const err = new Error(detail)
  // Surface the HTTP status so callers can distinguish a not-yet-built endpoint
  // (404 on a `proposed` route) from a real failure and degrade gracefully.
  err.status = response.status
  return err
}

export async function get(path, options) {
  const r = await fetch(`${base}/api${path}`, options)
  if (!r.ok) throw await responseError(r, path)
  return r.json()
}

export async function post(path, body = {}, options = {}) {
  const r = await fetch(`${base}/api${path}`, {
    method: 'POST',
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw await responseError(r, path)
  return r.json()
}

export async function patch(path, body = {}) {
  const r = await fetch(`${base}/api${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw await responseError(r, path)
  return r.json()
}

export const api = {
  configChoices: () => get('/config/choices'),
  presets: () => get('/presets'),
  overview: () => get('/lab/overview'),
  work: () => get('/work', coordinationReadOptions),
  decisions: () => get('/decisions', coordinationReadOptions),
  decision: (id) => get(`/decisions/${encodeURIComponent(id)}`, coordinationReadOptions),
  resolveDecision: (id, payload) => post(`/decisions/${encodeURIComponent(id)}/resolve`, payload, coordinationReadOptions),
  workDetail: (id) => get(`/work/${encodeURIComponent(id)}`, coordinationReadOptions),
  workOverlaps: (id) => get(`/work/${encodeURIComponent(id)}/overlaps`, coordinationReadOptions),
  lineage: (id) => get(`/objects/${encodeURIComponent(id)}/lineage`, coordinationReadOptions),
  evidence: (id) => get(`/evidence/${encodeURIComponent(id)}`, coordinationReadOptions),
  claimProposal: (id) => get(`/claim-proposals/${encodeURIComponent(id)}`, coordinationReadOptions),
  review: (id) => get(`/reviews/${encodeURIComponent(id)}`, coordinationReadOptions),
  explore: () => get('/explore'),
  exploreDomain: (domain) => get(`/explore/domain/${encodeURIComponent(domain)}`),
  exploreDataset: (dataset) => get(`/explore/dataset/${encodeURIComponent(dataset)}`),
  exploreTask: (task, domain) =>
    get(`/explore/task/${encodeURIComponent(task)}${domain ? `?domain=${encodeURIComponent(domain)}` : ''}`),
  scalingTests: () => get('/scaling-tests'),
  scalingTest: (groupId) => get(`/scaling-tests/${encodeURIComponent(groupId)}`),
  status: () => get('/status'),
  researchWorld: () => get('/research-world'),
  datasets: () => get('/datasets'),
  importHfDataset: (payload) => post('/datasets/hf/import', payload),
  jobs: () => get('/jobs'),
  job: (id) => get(`/jobs/${id}`),
  workspace: (id) => get(`/jobs/${id}/workspace`),
  comparison: (id) => get(`/jobs/${id}/comparison`),
  classicalAnalogueForJob: (id) => get(`/jobs/${id}/classical-analogue`),
  queueClassicalAnalogue: (id, payload = {}) => post(`/jobs/${id}/classical-analogue`, payload),
  queueGroupClassicalAnalogues: (id) => post(`/groups/${encodeURIComponent(id)}/classical-analogues`),
  jobGraph: (id) => get(`/jobs/${id}/model-graph`),
  modelTests: (id) => get(`/jobs/${id}/model-tests`),
  runModelTest: (id, payload) => post(`/jobs/${id}/model-tests`, payload),
  presetGraph: (id) => get(`/presets/${encodeURIComponent(id)}/model-graph`),
  presetClassicalAnalogue: (id) => get(`/presets/${encodeURIComponent(id)}/classical-analogue`),
  modelSpecs: () => get('/model-specs'),
  createModelSpec: (payload) => post('/model-specs', payload),
  updateModelSpec: (id, payload) => patch(`/model-specs/${id}`, payload),
  validateModelSpec: (payload) => post('/model-specs/validate', payload),
  runModelSpec: (id, payload) => post(`/model-specs/${id}/jobs`, payload),
  modelSpecDiff: (id, base) => get(`/model-specs/${id}/diff${base ? `?base=${base}` : ''}`),
  createJob: (payload) => post('/jobs', payload),
  createSweep: (payload) => post('/jobs/sweep', payload),
  studies: () => get('/studies'),
  study: (id) => get(`/studies/${id}`),
  studyReport: (id) => get(`/studies/${id}/report`),
  createStudy: (payload) => post('/studies', payload),
  queueStudy: (id) => post(`/studies/${id}/queue`),
  cancelJob: (id) => post(`/jobs/${id}/cancel`),
  suites: () => get('/suites'),
  suite: (name, dataset) =>
    get(`/suite/${encodeURIComponent(name)}${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ''}`),
  runs: (suite) => get(`/runs${suite ? `?suite=${encodeURIComponent(suite)}` : ''}`),
  run: (id) => get(`/run/${id}`),
  live: () => get('/live'),
  liveCurve: (key) => get(`/live/${key}/curve`),
  plots: () => get('/plots'),
  // Shipped research contracts (backend-owned) — see qllm/dashboard/openapi.json
  // and docs/BUILD_COORDINATION.md. Callers still degrade gracefully on 404 so
  // the UI stays usable against an older backend build.
  diagnostics: (id) => get(`/jobs/${id}/diagnostics`),
  verdicts: () => get('/verdicts'),
  verdict: (id) => get(`/verdicts/${encodeURIComponent(id)}`),
  atlasOntology: () => get('/atlas/ontology'),
  trackBReadiness: () => get('/atlas/track-b-readiness'),
  researchCapabilities: () => get('/research/capabilities'),
  arxivScan: (payload = {}) => post('/discover/arxiv/scan', payload),
  designerCapabilities: () => get('/designer/circuit'),
  designerCircuit: (payload) => post('/designer/circuit', payload),
}

// The fixed selector is not a credential and is honored only when the backend
// explicitly enables synthetic coordination test mode.
const coordinationReadOptions = {
  headers: { 'X-StateVector-Test-Actor': 'human-reviewer' },
}
