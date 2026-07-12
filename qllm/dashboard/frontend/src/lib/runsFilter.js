// Pure filtering for the Runs table: status + dataset + free-text search over
// id / run_name / preset / dataset / model_family. Framework-free for node --test.

export function filterRuns(jobs, { status = 'all', dataset = 'all', search = '' } = {}) {
  const q = String(search || '').trim().toLowerCase()
  return (Array.isArray(jobs) ? jobs : []).filter((j) => {
    if (status !== 'all' && j.status !== status) return false
    if (dataset !== 'all' && j.dataset_name !== dataset) return false
    if (q) {
      const hay = `${j.id} ${j.run_name || ''} ${j.preset_id || ''} ${j.dataset_name || ''} ${j.model_family || ''}`.toLowerCase()
      if (!hay.includes(q)) return false
    }
    return true
  })
}

export function uniqueDatasets(jobs) {
  return [...new Set((Array.isArray(jobs) ? jobs : []).map((j) => j.dataset_name).filter(Boolean))].sort()
}
