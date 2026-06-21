import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'

function count(counts, key) {
  return counts?.[key] || 0
}

function fmt(value) {
  if (value == null) return '-'
  return Number(value).toFixed(3)
}

export default function LabOverview() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  const refresh = () => api.overview().then(setData).catch((e) => setError(e.message))
  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 3000)
    return () => clearInterval(id)
  }, [])

  if (!data && !error) return <div className="loading">Loading lab overview...</div>

  return (
    <div>
      <h1>Lab Overview</h1>
      <h2>Running work, queue health, recent comparisons, and system readiness.</h2>
      {error && <div className="alert error">{error}</div>}

      <div className="stat-row">
        <div className="metric-card"><div className="metric-label">Running</div><div className="metric-value">{count(data?.counts, 'running')}</div></div>
        <div className="metric-card"><div className="metric-label">Queued</div><div className="metric-value">{count(data?.counts, 'queued')}</div></div>
        <div className="metric-card"><div className="metric-label">Done</div><div className="metric-value">{count(data?.counts, 'done')}</div></div>
        <div className="metric-card"><div className="metric-label">Failed</div><div className="metric-value">{count(data?.counts, 'error')}</div></div>
        <div className="metric-card"><div className="metric-label">GPU</div><div className="metric-value">{data?.gpu_status?.ready ? 'ready' : 'cpu'}</div><div className="muted">{data?.gpu_status?.jax_backend || 'JAX unknown'}</div></div>
      </div>

      <div className="action-grid">
        <Link className="action-card" to="/launch"><b>New Experiment</b><span>Queue a preset or matched classical comparison.</span></Link>
        <Link className="action-card" to="/experiments"><b>Manage Experiments</b><span>Filter, cancel, open, rerun, and compare jobs.</span></Link>
        <Link className="action-card" to="/studies"><b>Create Study</b><span>Queue multi-seed protocols with baselines, controls, and sweeps.</span></Link>
        <Link className="action-card" to="/scaling"><b>Scaling Tests</b><span>Review legacy qubit/depth sweep groups.</span></Link>
        <Link className="action-card" to="/models"><b>Browse Models</b><span>Inspect preset architectures before running them.</span></Link>
        <Link className="action-card" to="/results"><b>Review Results</b><span>Open leaderboards and study-style summaries.</span></Link>
      </div>

      <section className="panel table-panel">
        <table>
          <thead><tr><th>Active job</th><th>Status</th><th>Preset</th><th>Dataset</th><th>Family</th><th className="num">Steps</th></tr></thead>
          <tbody>
            {(data?.active_jobs || []).map((j) => (
              <tr key={j.id}>
                <td><Link to={`/jobs/${j.id}`}>#{j.id} {j.run_name}</Link></td>
                <td><span className={`badge ${j.status}`}>{j.status}</span></td>
                <td>{j.preset_id}</td>
                <td>{j.dataset_name}</td>
                <td>{j.model_family}</td>
                <td className="num">{j.steps}</td>
              </tr>
            ))}
            {(data?.active_jobs || []).length === 0 && <tr><td colSpan="6">Nothing is running or queued. Start from New Experiment.</td></tr>}
          </tbody>
        </table>
      </section>

      <div className="workspace-grid">
        <section className="panel">
          <h3>Recent Comparisons</h3>
          {(data?.recent_comparisons || []).map((item) => (
            <div className="comparison-row" key={item.job_id}>
              <div>
                <Link to={`/comparisons/${item.job_id}`}>#{item.candidate?.id} vs #{item.baseline?.id}</Link>
                <div className="muted">{item.candidate?.preset_id} vs {item.baseline?.preset_id}</div>
              </div>
              <span className="badge">{item.verdict?.label || 'pending'}</span>
            </div>
          ))}
          {(data?.recent_comparisons || []).length === 0 && <p className="muted">No linked comparisons yet.</p>}
        </section>

        <section className="panel">
          <h3>Leaderboard Highlights</h3>
          {(data?.leaderboard_highlights || []).map((r) => (
            <div className="comparison-row" key={`${r.variant}-${r.dataset}`}>
              <div><b>{r.variant}</b><div className="muted">{r.dataset}</div></div>
              <span className="mono">{fmt(r.best_ppl)} ppl</span>
            </div>
          ))}
          {(data?.leaderboard_highlights || []).length === 0 && <p className="muted">Finished runs will appear here.</p>}
        </section>
      </div>

      {(data?.recent_failed_jobs || []).length > 0 && (
        <section className="panel">
          <h3>Failed Runs Requiring Attention</h3>
          {(data?.recent_failed_jobs || []).map((j) => (
            <div className="comparison-row" key={j.id}>
              <Link to={`/jobs/${j.id}`}>#{j.id} {j.run_name}</Link>
              <span className="badge error">failed</span>
            </div>
          ))}
        </section>
      )}
    </div>
  )
}
