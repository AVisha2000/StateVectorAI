import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  Line,
  LineChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api'
import EvidenceWarnings from '../components/EvidenceWarnings'
import RunLedger from '../components/RunLedger'

function fmt(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return '-'
  const n = Number(value)
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return n.toFixed(digits)
}

function chartRows(points) {
  return points
    .filter((p) => !p.rerun_required && (p.val_ppl != null || p.wall_seconds != null))
    .map((p) => ({
      ...p,
      label: `q${p.n_qubits}/d${p.n_circuit_layers}`,
    }))
}

export default function ScalingTest() {
  const { groupId } = useParams()
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    setPayload(null)
    setError('')
    api.scalingTest(groupId)
      .then((result) => { if (active) { setPayload(result); setError('') } })
      .catch((e) => { if (active) setError(e.message) })
    return () => { active = false }
  }, [groupId])

  const points = payload?.points || []
  const rows = useMemo(() => chartRows(points), [points])

  if (!payload && !error) return <div className="loading">Loading scaling test...</div>

  return (
    <div>
      <h1>Scaling Test</h1>
      <h2>Same preset across qubit and depth scales, grouped as one experiment.</h2>
      {error && <div className="alert error">{error}</div>}
      <EvidenceWarnings warnings={payload?.interpretation_warnings} />
      {(payload?.protocol_warnings || []).map((warning) => <div className="alert error" key={warning}>{warning}</div>)}
      {payload && !payload.available && <div className="panel muted">No scaling test data is available for this group.</div>}

      {payload?.available && (
        <>
          <section className="panel">
            <div className="workspace-header">
              <div>
                <h3>{payload.preset_id}</h3>
                <p className="panel-copy">
                  Dataset {payload.dataset_name}, seed {payload.seed}, {payload.steps} steps,
                  target {payload.device_target}. Group <span className="mono">{payload.group_id.slice(0, 8)}</span>.
                </p>
              </div>
              <span className="badge best">{payload.complete_count}/{payload.total_count} complete</span>
            </div>
            {payload.best && (
              <div className="stat-row">
                <div className="metric-card"><div className="metric-label">Best scale</div><div className="metric-value">q{payload.best.n_qubits} d{payload.best.n_circuit_layers}</div></div>
                <div className="metric-card"><div className="metric-label">Best val ppl</div><div className="metric-value">{fmt(payload.best.val_ppl)}</div></div>
                <div className="metric-card"><div className="metric-label">Wall time</div><div className="metric-value">{fmt(payload.best.wall_seconds, 2)}s</div></div>
                <div className="metric-card"><div className="metric-label">Parameters</div><div className="metric-value">{payload.best.n_params?.toLocaleString?.() || '-'}</div></div>
              </div>
            )}
          </section>

          <div className="chart-grid">
            <section className="panel chart-panel">
              <div className="pill chart-title">Validation perplexity by scale</div>
              {rows.length ? (
                <ResponsiveContainer width="100%" height="92%">
                  <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <XAxis dataKey="label" tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, fontFamily: 'monospace', fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="val_ppl" name="val ppl" stroke="#a371f7" dot strokeWidth={2} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              ) : <p className="muted">Results will appear as the scaled runs finish.</p>}
            </section>

            <section className="panel chart-panel">
              <div className="pill chart-title">Wall time by scale</div>
              {rows.length ? (
                <ResponsiveContainer width="100%" height="92%">
                  <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <XAxis dataKey="label" tick={{ fill: '#8b949e', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, fontFamily: 'monospace', fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="wall_seconds" name="wall seconds" stroke="#2f81f7" dot strokeWidth={2} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              ) : <p className="muted">Wall-time points will appear after runs finish.</p>}
            </section>
          </div>

          <section className="panel table-panel">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th className="num">Qubits</th>
                  <th className="num">Depth</th>
                  <th className="num">Scale</th>
                  <th className="num">Val ppl</th>
                  <th className="num">Val loss</th>
                  <th className="num">Wall</th>
                  <th className="num">Params</th>
                </tr>
              </thead>
              <tbody>
                {points.map((p) => (
                  <tr key={p.job.id}>
                    <td><Link to={`/jobs/${p.job.id}`}>#{p.job.id} {p.job.run_name}</Link><div className="muted mono">{p.variant}</div></td>
                    <td><span className={`badge ${p.status}`}>{p.status}</span></td>
                    <td className="num">{p.n_qubits}</td>
                    <td className="num">{p.n_circuit_layers}</td>
                    <td className="num">{p.scale}</td>
                    <td className="num">{p.rerun_required ? <><b>rerun required</b><div className="muted">historical {fmt(p.val_ppl)}</div></> : fmt(p.val_ppl)}</td>
                    <td className="num">{fmt(p.val_loss)}</td>
                    <td className="num">{p.wall_seconds == null ? '-' : `${fmt(p.wall_seconds, 2)}s`}</td>
                    <td className="num">{p.n_params?.toLocaleString?.() || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="resource-ledger-list" aria-label="Scaling point resource ledgers">
            <h3>Recorded scaling ledgers</h3>
            {points.map((point) => (
              <details className="ledger-details" key={`${point.job.id}-ledger`}>
                <summary>Job #{point.job.id} — q{point.n_qubits}/d{point.n_circuit_layers}</summary>
                <EvidenceWarnings warnings={point.interpretation_warnings} />
                <RunLedger manifest={point.manifest} durability={point.durability} resourceLedger={point.resource_ledger} backendCapabilities={point.backend_capabilities} />
              </details>
            ))}
          </section>
        </>
      )}
    </div>
  )
}
