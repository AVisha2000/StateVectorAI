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
  explore: () => get('/explore'),
  exploreDomain: (domain) => get(`/explore/domain/${encodeURIComponent(domain)}`),
  exploreDataset: (dataset) => get(`/explore/dataset/${encodeURIComponent(dataset)}`),
  exploreTask: (task, domain) =>
    get(`/explore/task/${encodeURIComponent(task)}${domain ? `?domain=${encodeURIComponent(domain)}` : ''}`),
  scalingTests: () => get('/scaling-tests'),
  scalingTest: (groupId) => get(`/scaling-tests/${encodeURIComponent(groupId)}`),
  status: () => get('/status'),
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
}
