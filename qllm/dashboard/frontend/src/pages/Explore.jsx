import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api'

function fmt(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return '-'
  return Number(value).toFixed(digits)
}

function countLabel(item) {
  const runs = item.runs || 0
  const jobs = item.jobs || 0
  return `${runs} runs, ${jobs} jobs`
}

function roleClass(role) {
  if (role === 'quantum') return 'quantum'
  if (role === 'hybrid') return 'best'
  return ''
}

export default function Explore() {
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState('')
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const activeDomain = params.get('domain') || ''

  useEffect(() => {
    api.explore().then(setPayload).catch((e) => setError(e.message))
  }, [])

  const domains = payload?.domains || []
  const selectedDomain = useMemo(() => (
    domains.find((domain) => domain.slug === activeDomain) || domains[0]
  ), [domains, activeDomain])

  const domainTasks = useMemo(() => (
    (payload?.tasks || []).filter((task) => !selectedDomain || task.domain_slug === selectedDomain.slug)
  ), [payload, selectedDomain])

  const domainDatasets = useMemo(() => (
    (payload?.datasets || []).filter((dataset) => !selectedDomain || dataset.domains.includes(selectedDomain.name))
  ), [payload, selectedDomain])

  const domainRuns = useMemo(() => (
    (payload?.runs || []).filter((run) => !selectedDomain || run.context.domain_slug === selectedDomain.slug).slice(0, 10)
  ), [payload, selectedDomain])

  const domainJobs = useMemo(() => (
    (payload?.jobs || []).filter((job) => !selectedDomain || job.context.domain_slug === selectedDomain.slug).slice(0, 8)
  ), [payload, selectedDomain])

  const rerunWarnings = useMemo(() => Array.from(new Set(
    domainRuns
      .filter((run) => run.rerun_required && run.metric_contract?.limitation)
      .map((run) => run.metric_contract.limitation)
  )), [domainRuns])

  if (!payload && !error) return <div className="loading">Loading research map...</div>

  return (
    <div>
      <h1>Explore</h1>
      <h2>Navigate QML evidence by domain, task, dataset, study, and run.</h2>
      {error && <div className="alert error">{error}</div>}

      <div className="grid explore-domain-grid">
        {domains.map((domain) => (
          <button
            className={`card preset ${selectedDomain?.slug === domain.slug ? 'selected' : ''}`}
            key={domain.slug}
            onClick={() => navigate(`/explore?domain=${encodeURIComponent(domain.slug)}`)}
            type="button"
          >
            <h3>{domain.name}</h3>
            <div className="stat"><span className="k">tasks</span><span className="v">{domain.tasks.length}</span></div>
            <div className="stat"><span className="k">datasets</span><span className="v">{domain.datasets.length}</span></div>
            <div className="stat"><span className="k">evidence</span><span className="v">{countLabel(domain)}</span></div>
            <div className="muted">Inferred from: {domain.inferred_from.join(', ')}</div>
          </button>
        ))}
        {domains.length === 0 && (
          <div className="panel">
            <h3>No research map yet</h3>
            <p className="panel-copy">Runs and queued jobs will appear here once experiments exist.</p>
            <Link className="small-link" to="/launch">Queue an experiment</Link>
          </div>
        )}
      </div>

      {selectedDomain && (
        <>
          <div className="workspace-header explore-heading">
            <div>
              <h1>{selectedDomain.name}</h1>
              <h2>{countLabel(selectedDomain)} across {selectedDomain.datasets.length} datasets.</h2>
            </div>
            <Link className="small-link" to="/results">Open legacy results</Link>
          </div>
          {rerunWarnings.map((warning) => <div className="alert error" key={warning}>{warning}</div>)}

          <div className="workspace-grid">
            <section className="panel">
              <h3>Tasks</h3>
              <div className="research-card-list">
                {domainTasks.map((task) => (
                  <Link
                    className="research-card"
                    key={`${task.domain_slug}-${task.slug}`}
                    to={`/explore/task/${task.slug}?domain=${encodeURIComponent(task.domain_slug)}`}
                  >
                    <b>{task.name}</b>
                    <span>{task.datasets.length} datasets - {countLabel(task)}</span>
                  </Link>
                ))}
                {domainTasks.length === 0 && <p className="muted">No tasks inferred yet.</p>}
              </div>
            </section>

            <section className="panel">
              <h3>Datasets</h3>
              <div className="research-card-list">
                {domainDatasets.map((dataset) => (
                  <Link className="research-card" key={dataset.name} to={`/explore/dataset/${encodeURIComponent(dataset.name)}`}>
                    <b>{dataset.name}</b>
                    <span>{dataset.tasks.join(', ')}</span>
                    <span>best ppl {fmt(dataset.best_val_ppl)} - {countLabel(dataset)}</span>
                  </Link>
                ))}
                {domainDatasets.length === 0 && <p className="muted">No datasets found for this domain.</p>}
              </div>
            </section>
          </div>

          <section className="panel table-panel">
            <table>
              <thead>
                <tr>
                  <th>Run</th><th>Model</th><th>Role</th><th>Dataset</th><th>Task</th>
                  <th className="num">Val ppl</th><th className="num">Wall</th><th className="num">Qubits</th>
                </tr>
              </thead>
              <tbody>
                {domainRuns.map((run) => (
                  <tr key={run.id}>
                    <td><Link to={run.link}>#{run.id}</Link></td>
                    <td>{run.variant}<div className="muted">{run.model_family}</div></td>
                    <td><span className={`badge ${roleClass(run.role)}`}>{run.role}</span></td>
                    <td><Link to={`/explore/dataset/${encodeURIComponent(run.dataset)}`}>{run.dataset}</Link></td>
                    <td>{run.context.task}</td>
                    <td className="num">{run.rerun_required ? <><b>rerun required</b><div className="muted">historical {fmt(run.val_ppl)} ppl</div></> : fmt(run.val_ppl)}</td>
                    <td className="num">{run.wall_seconds == null ? '-' : `${fmt(run.wall_seconds, 2)}s`}</td>
                    <td className="num">{run.resource.n_qubits ?? '-'}</td>
                  </tr>
                ))}
                {domainRuns.length === 0 && <tr><td colSpan="8">No completed runs for this domain yet.</td></tr>}
              </tbody>
            </table>
          </section>

          <section className="panel">
            <h3>Queued And Recent Jobs</h3>
            <div className="research-card-list">
              {domainJobs.map((job) => (
                <Link className="research-card" key={job.id} to={job.link}>
                  <b>#{job.id} {job.run_name}</b>
                  <span>{job.preset_id} - {job.dataset} - {job.status}</span>
                  <span>{job.context.task} - {job.role} - resource {job.resource.resource_band || 'unknown'}</span>
                </Link>
              ))}
              {domainJobs.length === 0 && <p className="muted">No lab jobs for this domain yet.</p>}
            </div>
          </section>
        </>
      )}
    </div>
  )
}
