import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { api } from '../api'

export default function Run() {
  const { id } = useParams()
  const [run, setRun] = useState(null)
  useEffect(() => { api.run(id).then(setRun).catch(console.error) }, [id])
  if (!run) return <div className="loading">Loading run...</div>

  const curve = run.steps_curve || {}
  const loss = curve.train_loss || []
  const series = loss.map((p) => ({ step: p.step, train_loss: p.value }))

  return (
    <div>
      <h1>{run.run_name || `run ${run.id}`}</h1>
      <h2>{run.suite} - {run.variant} - {run.dataset} - seed {run.seed}</h2>
      {run.metric_contract?.rerun_required && <div className="alert error">{run.metric_contract.limitation}</div>}

      <div className="panel">
        <p className="pill" style={{ marginTop: 0 }}>
          Lower validation perplexity is better. Grad norm ratio compares signal
          reaching quantum circuit weights against classical weights.
        </p>
        <div className="kv">
          <div className="k">val ppl</div><div className="v">{run.val_ppl?.toFixed(4)}</div>
          <div className="k">params</div><div className="v">{run.n_params?.toLocaleString()}</div>
          <div className="k">steps</div><div className="v">{run.steps}</div>
          <div className="k">wall (s)</div><div className="v">{run.wall_seconds?.toFixed(0)}</div>
          {Object.entries(run.metrics || {}).map(([k, v]) => (
            <div key={k} style={{ display: 'contents' }}>
              <div className="k">{k}</div><div className="v">{Number(v).toFixed(4)}</div>
            </div>
          ))}
        </div>
      </div>

      {series.length > 0 && (
        <div className="panel" style={{ height: 280 }}>
          <div className="pill" style={{ marginBottom: 8 }}>training curve</div>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={series}>
              <XAxis dataKey="step" tick={{ fill: '#8b949e', fontSize: 11 }} />
              <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, fontFamily: 'monospace', fontSize: 12 }} />
              <Line type="monotone" dataKey="train_loss" stroke="#2f81f7" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <details className="panel">
        <summary className="pill">full config</summary>
        <div className="kv" style={{ marginTop: 12 }}>
          {Object.entries(run.config || {}).map(([k, v]) => (
            <div key={k} style={{ display: 'contents' }}>
              <div className="k">{k}</div><div className="v">{String(v)}</div>
            </div>
          ))}
        </div>
      </details>
    </div>
  )
}
