export function studiesForDomain(studies = [], selectedDomain = null) {
  const rows = Array.isArray(studies) ? studies : []
  if (!selectedDomain) return rows
  return rows.filter((study) => study.domain === selectedDomain.name)
}

export function taskLinkForStudy(study, tasks = []) {
  const task = tasks.find((item) => (
    item.domain === study.domain && item.name === study.task
  ))
  if (!task) return null
  return `/explore/task/${encodeURIComponent(task.slug)}?domain=${encodeURIComponent(task.domain_slug)}`
}
