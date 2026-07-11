import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { api } from '../api'
import { chartAxisTick, chartSeries, chartTooltipProps } from '../chartTheme'
import EvidenceSummary from '../components/EvidenceSummary'
import EvidenceWarnings from '../components/EvidenceWarnings'
import RunLedger from '../components/RunLedger'

export default function Run() {
  const { id } = useParams()
  const [run, setRun] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    setRun(null)
    setError('')
    api.run(id)
      .then((payload) => { if (active) { setRun(payload); setError('') } })
      .catch((e) => { if (active) setError(e.message) })
    return () => { active = false }
  }, [id])
  if (!run && !error) return <div className="loading">Loading run...</div>
  if (!run) return <div><h1>Run {id}</h1><div className="alert error">{error}</div></div>

  const curve = run.steps_curve || {}
  const loss = curve.train_loss || []
  const series = loss.map((p) => ({ step: p.step, train_loss: p.value }))

  return (
    <div>
      <h1>{run.run_name || `run ${run.id}`}</h1>
      <h2>{run.suite} - {run.variant} - {run.dataset} - seed {run.seed}</h2>
      <EvidenceWarnings warnings={run.interpretation_warnings} />
      {run.metric_contract?.rerun_required && <div className="alert error">{run.metric_contract.limitation}</div>}
      <EvidenceSummary evidence={run} title="Run evidence contract" />
      <RunLedger
        manifest={run.manifest}
        durability={{
          status: 'completed',
          immutable_identity: {
            experiment_uuid: run.experiment_uuid,
            run_uuid: run.run_uuid,
            manifest_hash: run.manifest_hash,
            config_hash: run.config_hash,
            code_hash: run.code_hash,
            data_hash: run.data_hash,
            environment_hash: run.environment_hash,
            seed_axes_hash: run.seed_axes_hash,
          },
        }}
        resourceLedger={run.resource_ledger}
        backendCapabilities={run.backend_capabilities}
      />

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
              <XAxis dataKey="step" tick={chartAxisTick} />
              <YAxis tick={chartAxisTick} />
              <Tooltip {...chartTooltipProps} />
              <Line type="monotone" dataKey="train_loss" stroke={chartSeries.blue} dot={false} strokeWidth={2} />
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
