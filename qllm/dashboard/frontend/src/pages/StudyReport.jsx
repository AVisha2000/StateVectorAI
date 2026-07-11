import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'

function fmt(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return '-'
  const n = Number(value)
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return n.toFixed(digits)
}

function RoleSummary({ label, item }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{item?.completed_jobs ?? 0}</div>
      <div className="muted">mean wall {item?.mean_wall_seconds == null ? '-' : `${fmt(item.mean_wall_seconds, 2)}s`}</div>
      <div className="muted">mean params {item?.mean_n_params == null ? '-' : Math.round(item.mean_n_params).toLocaleString()}</div>
    </div>
  )
}

export default function StudyReport() {
  const { id } = useParams()
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.studyReport(id).then(setPayload).catch((e) => setError(e.message))
  }, [id])

  if (!payload && !error) return <div className="loading">Loading study report...</div>

  return (
    <div>
      {error && <div className="alert error">{error}</div>}
      {payload && (
        <>
          <div className="workspace-header">
            <div>
              <h1>Study Report: {payload.name}</h1>
              <h2>{payload.research_question || 'Multi-run quantum/classical report with cautious verdict and limitations.'}</h2>
            </div>
            <div className="header-actions">
              <span className={`badge ${payload.status}`}>{payload.status}</span>
              <Link className="small" to={`/studies/${payload.id}`}>Back to study</Link>
            </div>
          </div>
          {(payload.statistics?.rerun_required_pairs || 0) > 0 && <div className="alert error">{payload.verdict.reason}</div>}

          <section className="panel">
            <div className="workspace-header">
              <div>
                <h3>Verdict</h3>
                <p className="panel-copy">{payload.verdict.reason}</p>
              </div>
              <span className="badge best">{payload.verdict.label}</span>
            </div>
            <div className="stat-row">
              <div className="metric-card"><div className="metric-label">Fair pairs</div><div className="metric-value">{payload.statistics.fair_pairs}</div></div>
              <div className="metric-card"><div className="metric-label">Rerun-required pairs</div><div className="metric-value">{payload.statistics.rerun_required_pairs || 0}</div></div>
              <div className="metric-card"><div className="metric-label">Wins</div><div className="metric-value">{payload.statistics.wins}</div><div className="muted">win rate {payload.statistics.win_rate == null ? '-' : `${fmt(payload.statistics.win_rate * 100, 1)}%`}</div></div>
              <div className="metric-card"><div className="metric-label">Mean delta val ppl</div><div className="metric-value">{fmt(payload.statistics.mean_delta_val_ppl)}</div><div className="muted">candidate minus baseline</div></div>
              <div className="metric-card"><div className="metric-label">Std delta val ppl</div><div className="metric-value">{fmt(payload.statistics.std_delta_val_ppl)}</div></div>
            </div>
          </section>

          <div className="workspace-grid">
            <section className="panel">
              <h3>Research Question</h3>
              <p>{payload.research_question || 'No explicit research question recorded.'}</p>
              <div className="kv compact">
                <div className="k">task</div><div className="v">{payload.protocol.task || '-'}</div>
                <div className="k">datasets</div><div className="v">{(payload.protocol.dataset_names || []).join(', ') || '-'}</div>
                <div className="k">seeds</div><div className="v">{(payload.protocol.seeds || []).join(', ') || '-'}</div>
                <div className="k">steps / eval</div><div className="v">{payload.protocol.steps} / {payload.protocol.eval_every}</div>
                <div className="k">device target</div><div className="v">{payload.protocol.device_target}</div>
                <div className="k">batch / seq len</div><div className="v">{payload.protocol.batch_size || '-'} / {payload.protocol.seq_len || '-'}</div>
                <div className="k">group</div><div className="v mono">{payload.protocol.group_id}</div>
              </div>
            </section>

            <section className="panel">
              <h3>Candidate Architecture</h3>
              <div className="kv compact">
                <div className="k">candidate</div><div className="v">{payload.candidate.label}</div>
                <div className="k">kind</div><div className="v">{payload.candidate.kind}</div>
                <div className="k">architecture</div><div className="v">{payload.candidate.architecture}</div>
                <div className="k">quantum role</div><div className="v">{payload.candidate.quantum_role}</div>
                <div className="k">recommended use</div><div className="v">{payload.candidate.recommended_use}</div>
                <div className="k">risks</div><div className="v">{payload.candidate.risks}</div>
              </div>
            </section>
          </div>

          <section className="panel">
            <h3>Evidence Ladder</h3>
            {(payload.verdict.ladder || []).map((item) => (
              <div className="comparison-row" key={item.key}>
                <div>
                  <b>{item.label}</b>
                  <div className="muted">{item.detail}</div>
                  {item.caution && <div className="muted">{item.caution}</div>}
                </div>
                <span className={`badge ${item.ok ? 'done' : 'error'}`}>{item.ok ? 'met' : 'not met'}</span>
              </div>
            ))}
          </section>

          <section className="panel">
            <h3>Statistical Summary</h3>
            <div className="stat-row">
              <RoleSummary label="Candidate completions" item={payload.resource_summary.candidate} />
              <RoleSummary label="Baseline completions" item={payload.resource_summary.baseline} />
              <RoleSummary label="Control completions" item={payload.resource_summary.control} />
            </div>
          </section>

          <section className="panel">
            <h3>Cost And Resource Summary</h3>
            <div className="workspace-grid">
              <div className="kv compact">
                <div className="k">candidate mean qubits</div><div className="v">{fmt(payload.resource_summary.candidate.mean_qubits, 1)}</div>
                <div className="k">candidate mean depth</div><div className="v">{fmt(payload.resource_summary.candidate.mean_depth, 1)}</div>
                <div className="k">candidate resource bands</div><div className="v">{Object.entries(payload.resource_summary.candidate.resource_bands || {}).map(([key, value]) => `${key}:${value}`).join(', ') || '-'}</div>
              </div>
              <div className="kv compact">
                <div className="k">baseline mean qubits</div><div className="v">{fmt(payload.resource_summary.baseline.mean_qubits, 1)}</div>
                <div className="k">baseline mean depth</div><div className="v">{fmt(payload.resource_summary.baseline.mean_depth, 1)}</div>
                <div className="k">baseline resource bands</div><div className="v">{Object.entries(payload.resource_summary.baseline.resource_bands || {}).map(([key, value]) => `${key}:${value}`).join(', ') || '-'}</div>
              </div>
            </div>
          </section>

          <section className="panel table-panel">
            <h3>Per-Pair Evidence</h3>
            <table>
              <thead>
                <tr><th>Candidate</th><th>Baseline</th><th>Dataset</th><th className="num">Seed</th><th>Grid</th><th className="num">Delta ppl</th><th className="num">Delta wall</th><th>Verdict</th></tr>
              </thead>
              <tbody>
                {(payload.pair_rows || []).map((row) => (
                  <tr key={row.candidate_job_id}>
                    <td><Link to={`/jobs/${row.candidate_job_id}`}>#{row.candidate_job_id}</Link></td>
                    <td>{row.baseline_job_id ? <Link to={`/jobs/${row.baseline_job_id}`}>#{row.baseline_job_id}</Link> : '-'}</td>
                    <td>{row.dataset}</td>
                    <td className="num">{row.seed}</td>
                    <td className="mono">{row.grid?.n_qubits ? `q${row.grid.n_qubits}/d${row.grid.n_circuit_layers}` : '-'}</td>
                    <td className="num">{fmt(row.delta_val_ppl)}</td>
                    <td className="num">{fmt(row.delta_wall_seconds)}</td>
                    <td>{row.comparison_link ? <Link to={row.comparison_link}>{row.verdict_label || 'comparison'}</Link> : (row.verdict_label || row.reason || '-')}</td>
                  </tr>
                ))}
                {(payload.pair_rows || []).length === 0 && <tr><td colSpan="8">No paired evidence rows yet.</td></tr>}
              </tbody>
            </table>
          </section>

          <section className="panel">
            <h3>Limitations</h3>
            <div className="research-card-list">
              {(payload.limitations || []).map((item, index) => (
                <div className="research-card" key={`${index}-${item}`}>
                  <span>{item}</span>
                </div>
              ))}
              {(payload.limitations || []).length === 0 && <p className="muted">No additional limitations recorded.</p>}
            </div>
          </section>

          <section className="panel">
            <h3>Markdown Export</h3>
            <pre className="code-block">{payload.markdown}</pre>
          </section>
        </>
      )}
    </div>
  )
}
