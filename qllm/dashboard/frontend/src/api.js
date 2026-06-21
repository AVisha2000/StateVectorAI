const base = ''

export async function get(path) {
  const r = await fetch(`${base}/api${path}`)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

export async function post(path, body = {}) {
  const r = await fetch(`${base}/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    let detail = `${path}: ${r.status}`
    try {
      const payload = await r.json()
      detail = payload.detail || detail
    } catch (_) {}
    throw new Error(detail)
  }
  return r.json()
}

export async function patch(path, body = {}) {
  const r = await fetch(`${base}/api${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    let detail = `${path}: ${r.status}`
    try {
      const payload = await r.json()
      detail = payload.detail || detail
    } catch (_) {}
    throw new Error(detail)
  }
  return r.json()
}

export const api = {
  presets: () => get('/presets'),
  overview: () => get('/lab/overview'),
  scalingTests: () => get('/scaling-tests'),
  scalingTest: (groupId) => get(`/scaling-tests/${encodeURIComponent(groupId)}`),
  status: () => get('/status'),
  datasets: () => get('/datasets'),
  importHfDataset: (payload) => post('/datasets/hf/import', payload),
  jobs: () => get('/jobs'),
  job: (id) => get(`/jobs/${id}`),
  workspace: (id) => get(`/jobs/${id}/workspace`),
  comparison: (id) => get(`/jobs/${id}/comparison`),
  jobGraph: (id) => get(`/jobs/${id}/model-graph`),
  presetGraph: (id) => get(`/presets/${encodeURIComponent(id)}/model-graph`),
  modelSpecs: () => get('/model-specs'),
  createModelSpec: (payload) => post('/model-specs', payload),
  updateModelSpec: (id, payload) => patch(`/model-specs/${id}`, payload),
  validateModelSpec: (payload) => post('/model-specs/validate', payload),
  runModelSpec: (id, payload) => post(`/model-specs/${id}/jobs`, payload),
  modelSpecDiff: (id, base) => get(`/model-specs/${id}/diff${base ? `?base=${base}` : ''}`),
  createJob: (payload) => post('/jobs', payload),
  createSweep: (payload) => post('/jobs/sweep', payload),
  cancelJob: (id) => post(`/jobs/${id}/cancel`),
  suites: () => get('/suites'),
  suite: (name, dataset) =>
    get(`/suite/${encodeURIComponent(name)}${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ''}`),
  runs: (suite) => get(`/runs${suite ? `?suite=${encodeURIComponent(suite)}` : ''}`),
  run: (id) => get(`/run/${id}`),
  live: () => get('/live'),
  liveCurve: (key) => get(`/live/${key}/curve`),
  plots: () => get('/plots'),
}
