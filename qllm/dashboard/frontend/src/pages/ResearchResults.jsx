import { useEffect, useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../api'

function fmt(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return '-'
  const n = Number(value)
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return n.toFixed(digits)
}

function roleClass(role) {
  if (role === 'quantum') return 'quantum'
  if (role === 'hybrid') return 'best'
  return ''
}

function chartRows(rows) {
  return rows
    .filter((row) => row.val_ppl != null)
    .map((row) => ({
      ...row,
      label: row.model.length > 18 ? `${row.model.slice(0, 16)}...` : row.model,
      params: row.n_params || 0,
      wall: row.wall_seconds || 0,
      qubits: row.resource?.n_qubits || 0,
      depth: row.resource?.n_circuit_layers || 0,
    }))
}

function SummaryCard({ card }) {
  return (
    <div className="metric-card research-summary-card">
      <div className="metric-label">{card.label}</div>
      {card.available ? (
        <>
          <div className="metric-value">{card.model}</div>
          <div className="muted">{card.role} - ppl {fmt(card.val_ppl)} - wall {card.wall_seconds == null ? '-' : `${fmt(card.wall_seconds, 2)}s`}</div>
          <div className="muted">q {card.resource?.n_qubits ?? '-'} / depth {card.resource?.n_circuit_layers ?? '-'} / {card.resource?.backend || 'backend n/a'}</div>
          {card.verdict_label && <span className="badge best">{card.verdict_label}</span>}
        </>
      ) : (
        <p className="muted">{card.note}</p>
      )}
    </div>
  )
}

function ChartPanel({ title, children, empty }) {
  return (
    <section className="panel chart-panel">
      <div className="pill chart-title">{title}</div>
      {children || <p className="muted">{empty || 'No chartable completed runs yet.'}</p>}
    </section>
  )
}

export default function ResearchResults({ mode }) {
  const { dataset, task } = useParams()
  const [params] = useSearchParams()
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const request = mode === 'task'
      ? api.exploreTask(task, params.get('domain'))
      : api.exploreDataset(dataset)
    request.then(setPayload).catch((e) => setError(e.message))
  }, [mode, dataset, task, params])

  const rows = payload?.rows || []
  const charts = useMemo(() => chartRows(rows), [rows])
  const hasQuantumScale = charts.some((row) => row.qubits > 0 || row.depth > 0)

  if (!payload && !error) return <div className="loading">Loading research results...</div>

  return (
    <div>
      <h1>{mode === 'task' ? (payload?.tasks?.[0] || task) : dataset}</h1>
      <h2>
        {payload?.domains?.join(', ') || 'Research'} - {payload?.tasks?.join(', ') || 'task'}.
        Performance is shown with resource cost and cautious verdict labels.
      </h2>
      {error && <div className="alert error">{error}</div>}

      {payload?.available && (
        <>
          <div className="stat-row">
            {(payload.summaries || []).map((card) => <SummaryCard card={card} key={card.label} />)}
          </div>

          <div className="chart-grid">
            <ChartPanel title="Validation perplexity by model">
              {charts.length > 0 && (
                <ResponsiveContainer width="100%" height="92%">
                  <BarChart data={charts} margin={{ top: 8, right: 16, bottom: 54, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="label" angle={-30} textAnchor="end" height={60} tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} />
                    <Tooltip />
                    <Bar dataKey="val_ppl" name="val ppl" fill="#2f81f7" />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </ChartPanel>

            <ChartPanel title="Metric vs parameter count">
              {charts.length > 0 && (
                <ResponsiveContainer width="100%" height="92%">
                  <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="params" name="params" tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <YAxis dataKey="val_ppl" name="val ppl" tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter data={charts} fill="#a371f7" />
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </ChartPanel>

            <ChartPanel title="Metric vs wall time">
              {charts.length > 0 && (
                <ResponsiveContainer width="100%" height="92%">
                  <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="wall" name="wall seconds" tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <YAxis dataKey="val_ppl" name="val ppl" tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter data={charts} fill="#3fb950" />
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </ChartPanel>

            <ChartPanel title="Metric vs quantum scale" empty="Quantum scale appears when qubit/depth metadata exists.">
              {hasQuantumScale && (
                <ResponsiveContainer width="100%" height="92%">
                  <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="qubits" name="qubits" tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <YAxis dataKey="val_ppl" name="val ppl" tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter data={charts} fill="#d29922" />
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </ChartPanel>
          </div>

          <section className="panel table-panel">
            <table>
              <thead>
                <tr>
                  <th>Run</th><th>Model</th><th>Role</th><th>Dataset/task</th><th className="num">Seed</th><th className="num">Steps</th>
                  <th className="num">Val loss</th><th className="num">Val ppl</th><th className="num">BPC</th><th className="num">Accuracy</th>
                  <th className="num">Wall</th><th className="num">Params</th><th className="num">Qubits</th><th className="num">Depth</th><th>Shots/backend</th><th>Resource</th><th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={`${row.source}-${row.id}`}>
                    <td><Link to={row.link}>{row.source === 'job' ? `#${row.id}` : `run ${row.id}`}</Link><div className="muted">{row.run_name}</div></td>
                    <td>{row.model}<div className="muted">{row.model_family}</div></td>
                    <td><span className={`badge ${roleClass(row.role)}`}>{row.role}</span></td>
                    <td><Link to={`/explore/dataset/${encodeURIComponent(row.dataset)}`}>{row.dataset}</Link><div className="muted">{row.task}</div></td>
                    <td className="num">{row.seed}</td>
                    <td className="num">{row.steps}</td>
                    <td className="num">{fmt(row.val_loss)}</td>
                    <td className="num">{fmt(row.val_ppl)}</td>
                    <td className="num">{fmt(row.val_bpc)}</td>
                    <td className="num">{fmt(row.accuracy)}</td>
                    <td className="num">{row.wall_seconds == null ? '-' : `${fmt(row.wall_seconds, 2)}s`}</td>
                    <td className="num">{row.n_params?.toLocaleString?.() || '-'}</td>
                    <td className="num">{row.resource?.n_qubits ?? '-'}</td>
                    <td className="num">{row.resource?.n_circuit_layers ?? '-'}</td>
                    <td>{row.resource?.shots ?? 'analytic'} / {row.resource?.backend || '-'}</td>
                    <td>{row.resource?.resource_band || '-'}</td>
                    <td>{row.comparison_link ? <Link to={row.comparison_link}>{row.verdict_label || 'comparison'}</Link> : (row.verdict_label || '-')}</td>
                  </tr>
                ))}
                {rows.length === 0 && <tr><td colSpan="17">No runs found for this slice.</td></tr>}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  )
}
