import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import EvidenceWarnings from '../components/EvidenceWarnings'

function fmt(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return '-'
  return Number(value).toFixed(digits)
}

export default function ResultsHub() {
  const [explore, setExplore] = useState(null)
  const [studies, setStudies] = useState(null)
  const [suites, setSuites] = useState(null)
  const [errors, setErrors] = useState({})
  const [loading, setLoading] = useState({ explore: true, studies: true, suites: true })

  useEffect(() => {
    let active = true
    const requests = [
      ['explore', api.explore(), setExplore],
      ['studies', api.studies(), setStudies],
      ['suites', api.suites(), setSuites],
    ]
    requests.forEach(([key, request, setter]) => {
      request
        .then((payload) => {
          if (!active) return
          setter(payload)
          setErrors((current) => ({ ...current, [key]: '' }))
        })
        .catch((error) => {
          if (active) setErrors((current) => ({ ...current, [key]: error.message }))
        })
        .finally(() => {
          if (active) setLoading((current) => ({ ...current, [key]: false }))
        })
    })
    return () => { active = false }
  }, [])

  const topStudies = useMemo(() => (
    [...(studies || [])]
      .sort((a, b) => String(a.evidence?.label || '').localeCompare(String(b.evidence?.label || '')) || b.job_count - a.job_count)
      .slice(0, 6)
  ), [studies])

  const topDatasets = useMemo(() => (
    [...(explore?.datasets || [])]
      .sort((a, b) => (a.best_val_ppl ?? Number.POSITIVE_INFINITY) - (b.best_val_ppl ?? Number.POSITIVE_INFINITY))
      .slice(0, 6)
  ), [explore])

  const topTasks = useMemo(() => (
    [...(explore?.tasks || [])]
      .sort((a, b) => (b.runs + b.jobs) - (a.runs + a.jobs))
      .slice(0, 6)
  ), [explore])

  const topSuites = useMemo(() => (suites || []).slice(0, 6), [suites])

  if (Object.values(loading).every(Boolean)) return <div className="loading">Loading results hub...</div>

  return (
    <div>
      <h1>Results</h1>
      <h2>Study-level verdicts, dataset/task slices, and report links for cautious quantum-advantage review.</h2>
      <EvidenceWarnings warnings={[
        ...(explore?.interpretation_warnings || []),
        ...(studies || []).flatMap((study) => study.interpretation_warnings || study.evidence?.interpretation_warnings || []),
        ...(suites || []).flatMap((suite) => suite.interpretation_warnings || []),
      ]} />
      <div className="stat-row">
        <div className="metric-card"><div className="metric-label">Domains</div><div className="metric-value">{explore ? explore.domains?.length || 0 : '-'}</div><div className="muted">research areas inferred from runs/jobs</div></div>
        <div className="metric-card"><div className="metric-label">Tasks</div><div className="metric-value">{explore ? explore.tasks?.length || 0 : '-'}</div><div className="muted">task-level evidence slices</div></div>
        <div className="metric-card"><div className="metric-label">Datasets</div><div className="metric-value">{explore ? explore.datasets?.length || 0 : '-'}</div><div className="muted">dataset-level result views</div></div>
        <div className="metric-card"><div className="metric-label">Studies</div><div className="metric-value">{studies ? studies.length : '-'}</div><div className="muted">multi-seed protocols with reports</div></div>
      </div>

      <div className="workspace-grid">
        <section className="panel">
          <div className="workspace-header">
            <div>
              <h3>Study Reports</h3>
              <p className="panel-copy">Start here for cautious verdicts, limitations, protocol details, and cost summaries.</p>
            </div>
            <Link className="small-link" to="/studies">All studies</Link>
          </div>
          <div className="research-card-list">
            {loading.studies && <p className="loading">Loading study reports...</p>}
            {errors.studies && <div className="alert error">Study reports could not be loaded: {errors.studies}</div>}
            {topStudies.map((study) => (
              <div key={study.id} className="research-card evidence-card">
                <EvidenceWarnings warnings={study.interpretation_warnings || study.evidence?.interpretation_warnings} compact />
                <Link to={`/studies/${study.id}/report`}><b>#{study.id} {study.name}</b></Link>
                <span>{study.research_question || 'Multi-run protocol'}</span>
                <span>{study.job_count} jobs - {study.evidence?.label || 'pending'}</span>
              </div>
            ))}
            {!loading.studies && !errors.studies && topStudies.length === 0 && <p className="muted">No studies yet. Create one from the Studies page.</p>}
          </div>
        </section>

        <section className="panel">
          <div className="workspace-header">
            <div>
              <h3>Task Slices</h3>
              <p className="panel-copy">Advantage claims should usually be read at the task/study level before dataset-level scoreboards.</p>
            </div>
            <Link className="small-link" to="/explore">Explore map</Link>
          </div>
          <div className="research-card-list">
            {loading.explore && <p className="loading">Loading task slices...</p>}
            {errors.explore && <div className="alert error">Task slices could not be loaded: {errors.explore}</div>}
            {topTasks.map((task) => (
              <Link className="research-card" key={`${task.domain_slug}-${task.slug}`} to={`/explore/task/${task.slug}?domain=${encodeURIComponent(task.domain_slug)}`}>
                <b>{task.name}</b>
                <span>{task.domain}</span>
                <span>{task.datasets.length} datasets - {task.runs} runs, {task.jobs} jobs</span>
              </Link>
            ))}
            {!loading.explore && !errors.explore && topTasks.length === 0 && <p className="muted">No task slices yet.</p>}
          </div>
        </section>
      </div>

      <div className="workspace-grid">
        <section className="panel">
          <div className="workspace-header">
            <div>
              <h3>Dataset Result Views</h3>
              <p className="panel-copy">Use dataset slices to compare models while keeping verdicts and resource cost visible.</p>
            </div>
          </div>
          <div className="research-card-list">
            {loading.explore && <p className="loading">Loading dataset views...</p>}
            {errors.explore && <div className="alert error">Dataset views could not be loaded: {errors.explore}</div>}
            {topDatasets.map((dataset) => (
              <Link className="research-card" key={dataset.name} to={`/explore/dataset/${encodeURIComponent(dataset.name)}`}>
                <b>{dataset.name}</b>
                <span>{dataset.tasks.join(', ')}</span>
                <span>best ppl {fmt(dataset.best_val_ppl)} - {dataset.runs} runs, {dataset.jobs} jobs</span>
              </Link>
            ))}
            {!loading.explore && !errors.explore && topDatasets.length === 0 && <p className="muted">No dataset results yet.</p>}
          </div>
        </section>

        <section className="panel">
          <div className="workspace-header">
            <div>
              <h3>Legacy Suite Analytics</h3>
              <p className="panel-copy">The original suite leaderboard remains available for run-centric comparisons and historical slices.</p>
            </div>
          </div>
          <div className="research-card-list">
            {loading.suites && <p className="loading">Loading legacy suites...</p>}
            {errors.suites && <div className="alert error">Legacy suites could not be loaded: {errors.suites}</div>}
            {topSuites.map((suite) => (
              <Link className="research-card" key={suite.suite} to={`/suite/${encodeURIComponent(suite.suite)}`}>
                <b>{suite.suite}</b>
                <span>{suite.n} runs - {suite.variants} variants - {suite.datasets} datasets</span>
                <span>{suite.metric_contract?.rerun_required ? 'teacher-forced side-information - rerun required' : `best ppl ${suite.best_ppl == null ? '-' : fmt(suite.best_ppl)}`}</span>
              </Link>
            ))}
            {!loading.suites && !errors.suites && topSuites.length === 0 && <p className="muted">No legacy suite runs yet.</p>}
          </div>
        </section>
      </div>
    </div>
  )
}
