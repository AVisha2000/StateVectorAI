import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'

function fmt(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return '-'
  return Number(value).toFixed(digits)
}

function statusEntries(counts = {}) {
  return Object.entries(counts).sort(([a], [b]) => a.localeCompare(b))
}

export default function Study() {
  const { id } = useParams()
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState('')

  const refresh = () => api.study(id).then(setPayload).catch((e) => setError(e.message))

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 3000)
    return () => clearInterval(timer)
  }, [id])

  const evidenceRows = useMemo(() => (
    (payload?.evidence?.comparisons || [])
      .filter((item) => item.delta_val_ppl != null && item.fair && !item.rerun_required)
      .map((item, index) => ({ ...item, index: index + 1 }))
  ), [payload])

  if (!payload && !error) return <div className="loading">Loading study...</div>

  return (
    <div>
      {error && <div className="alert error">{error}</div>}
      {payload && (
        <>
          <div className="workspace-header">
            <div>
              <h1>Study #{payload.id}: {payload.name}</h1>
              <h2>{payload.research_question || 'Multi-run research protocol'}</h2>
            </div>
            <div className="header-actions">
              <span className={`badge ${payload.status}`}>{payload.status}</span>
              <Link className="small" to={`/studies/${payload.id}/report`}>Open report</Link>
              <Link className="small" to={`/experiments?group=${payload.group_id}`}>Open group</Link>
            </div>
          </div>
          {(payload.evidence?.rerun_required_pairs || 0) > 0 && <div className="alert error">{payload.evidence.reason}</div>}

          <section className="panel">
            <div className="stat-row">
              <div className="metric-card"><div className="metric-label">Evidence label</div><div className="metric-value">{payload.evidence?.label || 'pending'}</div><div className="muted">{payload.evidence?.reason}</div></div>
              <div className="metric-card"><div className="metric-label">Fair pairs</div><div className="metric-value">{payload.evidence?.fair_pairs || 0}</div><div className="muted">{payload.evidence?.complete_pairs || 0} complete</div></div>
              <div className="metric-card"><div className="metric-label">Wins</div><div className="metric-value">{payload.evidence?.wins || 0}</div><div className="muted">candidate lower val ppl</div></div>
              <div className="metric-card"><div className="metric-label">Mean delta val ppl</div><div className="metric-value">{fmt(payload.evidence?.mean_delta_val_ppl)}</div><div className="muted">candidate minus baseline</div></div>
              <div className="metric-card"><div className="metric-label">Std delta val ppl</div><div className="metric-value">{fmt(payload.evidence?.std_delta_val_ppl)}</div><div className="muted">across fair pairs</div></div>
            </div>
          </section>

          <div className="workspace-grid">
            <section className="panel">
              <h3>Protocol</h3>
              <div className="kv compact">
                <div className="k">task</div><div className="v">{payload.task || '-'}</div>
                <div className="k">candidate</div><div className="v">{payload.candidate_preset_id}</div>
                <div className="k">baseline policy</div><div className="v">{payload.baseline_policy}</div>
                <div className="k">controls</div><div className="v">{(payload.control_preset_ids || []).join(', ') || '-'}</div>
                <div className="k">datasets</div><div className="v">{(payload.dataset_names || []).join(', ')}</div>
                <div className="k">seeds</div><div className="v">{(payload.seeds || []).join(', ')}</div>
                <div className="k">qubits</div><div className="v">{(payload.sweep?.qubits || []).join(', ') || '-'}</div>
                <div className="k">depths</div><div className="v">{(payload.sweep?.depths || []).join(', ') || '-'}</div>
                <div className="k">group</div><div className="v mono">{payload.group_id}</div>
              </div>
            </section>

            <section className="panel">
              <h3>Evidence Ladder</h3>
              {(payload.evidence?.ladder || []).map((item) => (
                <div className="comparison-row" key={item.label}>
                  <div>
                    <b>{item.label}</b>
                    <div className="muted">{item.detail}</div>
                    {item.caution && <div className="muted">{item.caution}</div>}
                  </div>
                  <span className={`badge ${item.ok ? 'done' : 'error'}`}>{item.ok ? 'met' : 'not met'}</span>
                </div>
              ))}
            </section>
          </div>

          <section className="panel chart-panel">
            <div className="pill chart-title">Per-pair validation perplexity delta</div>
            {evidenceRows.length ? (
              <ResponsiveContainer width="100%" height="92%">
                <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                  <XAxis dataKey="index" name="pair" tick={{ fill: '#8b949e', fontSize: 11 }} />
                  <YAxis dataKey="delta_val_ppl" name="candidate-baseline" tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, fontFamily: 'monospace', fontSize: 12 }} />
                  <Scatter name="delta val ppl" data={evidenceRows} fill="#a371f7" />
                </ScatterChart>
              </ResponsiveContainer>
            ) : <p className="muted">Fair completed pairs will appear here. Negative deltas favor the candidate.</p>}
          </section>

          <section className="panel table-panel">
            <div className="workspace-header">
              <h3>Study Jobs</h3>
              <div className="chips">
                {statusEntries(payload.job_counts).map(([status, count]) => (
                  <span key={status} className={`badge ${status}`}>{status} {count}</span>
                ))}
              </div>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Job</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>Preset</th>
                  <th>Dataset</th>
                  <th className="num">Seed</th>
                  <th>Grid</th>
                  <th className="num">Val ppl</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {(payload.jobs || []).map((job) => (
                  <tr key={job.id}>
                    <td><Link to={`/jobs/${job.id}`}>#{job.id} {job.run_name}</Link></td>
                    <td><span className="badge">{job.study_role}</span></td>
                    <td><span className={`badge ${job.status}`}>{job.status}</span></td>
                    <td>{job.preset_id}</td>
                    <td>{job.dataset_name}</td>
                    <td className="num">{job.seed}</td>
                    <td className="mono">
                      {job.study_sweep?.n_qubits ? `q${job.study_sweep.n_qubits}/d${job.study_sweep.n_circuit_layers}` : '-'}
                    </td>
                    <td className="num">{fmt(job.final_run?.val_ppl)}</td>
                    <td>
                      {job.compare_to_job_id && <Link className="small-link" to={`/comparisons/${job.id}`}>comparison</Link>}
                      {job.analogue_state === 'missing' && <span className="badge error">analogue missing</span>}
                    </td>
                  </tr>
                ))}
                {(payload.jobs || []).length === 0 && <tr><td colSpan="9">No jobs are linked to this study yet.</td></tr>}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  )
}
