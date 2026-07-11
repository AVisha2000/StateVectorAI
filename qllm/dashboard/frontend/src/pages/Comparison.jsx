import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Line, LineChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import { chartAxisTick, chartSeries, chartTooltipProps } from '../chartTheme'
import ModelDiagram from '../components/ModelDiagram'
import EvidenceSummary from '../components/EvidenceSummary'
import EvidenceWarnings from '../components/EvidenceWarnings'
import RunLedger from '../components/RunLedger'

function fmt(value, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) return '-'
  const n = Number(value)
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return n.toFixed(digits)
}

function mergeComparison(candidateCurve = {}, baselineCurve = {}, metric = 'val_ppl') {
  const byStep = new Map()
  ;(candidateCurve[metric] || []).forEach((point) => {
    const row = byStep.get(point.step) || { step: point.step }
    row.candidate = point.value
    byStep.set(point.step, row)
  })
  ;(baselineCurve[metric] || []).forEach((point) => {
    const row = byStep.get(point.step) || { step: point.step }
    row.baseline = point.value
    byStep.set(point.step, row)
  })
  return Array.from(byStep.values()).sort((a, b) => a.step - b.step)
}

function Flag({ label, ok }) {
  return <span className={`badge ${ok ? 'done' : 'error'}`}>{label}: {ok ? 'yes' : 'no'}</span>
}

function Delta({ label, value, lowerBetter = true }) {
  const good = value != null && (lowerBetter ? value < 0 : value > 0)
  const bad = value != null && (lowerBetter ? value > 0 : value < 0)
  return <div className="stat"><span className="k">{label}</span><span className={`v ${good ? 'good-text' : ''} ${bad ? 'warn-text' : ''}`}>{value == null ? '-' : `${value > 0 ? '+' : ''}${fmt(value)}`}</span></div>
}

function EvidenceLadder({ ladder }) {
  if (!ladder) return null
  return (
    <section className="panel">
      <div className="workspace-header">
        <div>
          <h3>Quantum Advantage Evidence Ladder</h3>
          <p className="panel-copy">{ladder.label}: {ladder.reason}</p>
        </div>
        <span className="badge">{ladder.met_count}/{ladder.total_count} rungs met</span>
      </div>
      <div className="research-card-list">
        {(ladder.steps || []).map((step) => (
          <div className="comparison-row" key={step.key}>
            <div>
              <b>{step.label}</b>
              <div className="muted">{step.detail}</div>
              {step.caution && <div className="muted">{step.caution}</div>}
            </div>
            <span className={`badge ${step.ok ? 'done' : 'error'}`}>{step.ok ? 'met' : 'not met'}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

export default function Comparison() {
  const { id } = useParams()
  const [payload, setPayload] = useState(null)
  const [graphs, setGraphs] = useState({})
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  useEffect(() => {
    api.comparison(id).then((data) => {
      setPayload(data)
      const candidateId = data?.candidate?.job?.id
      const baselineId = data?.baseline?.job?.id
      Promise.all([
        candidateId ? api.jobGraph(candidateId) : Promise.resolve(null),
        baselineId ? api.jobGraph(baselineId) : Promise.resolve(null),
      ]).then(([candidate, baseline]) => setGraphs({ candidate, baseline }))
    }).catch((e) => setError(e.message))
  }, [id])

  const metric = payload?.candidate?.curve?.val_ppl?.length || payload?.baseline?.curve?.val_ppl?.length ? 'val_ppl' : 'train_loss'
  const series = useMemo(() => mergeComparison(payload?.candidate?.curve, payload?.baseline?.curve, metric), [payload, metric])

  if (!payload && !error) return <div className="loading">Loading comparison...</div>
  const c = payload?.candidate
  const b = payload?.baseline
  const flags = payload?.fairness || {}

  const queueAnalogue = async () => {
    const candidateId = c?.job?.id || id
    setError(''); setNotice('')
    try {
      const updated = await api.queueClassicalAnalogue(candidateId)
      setNotice(`Queued classical analogue job #${updated.comparison_job?.id || updated.analogue_job_id}.`)
      const next = await api.comparison(candidateId)
      setPayload(next)
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div>
      <h1>Comparison</h1>
      <h2>Quantum/classical protocol, architecture, metric deltas, and run-level verdict.</h2>
      {error && <div className="alert error">{error}</div>}
      {notice && <div className="alert good">{notice}</div>}
      <EvidenceWarnings warnings={payload?.interpretation_warnings} />
      {payload && <EvidenceSummary evidence={payload} title="Comparison evidence contract" />}
      {payload?.metric_contract?.rerun_required && <div className="alert error">{payload.metric_contract.limitation}</div>}
      {!payload?.available && (
        <div className="alert error">
          <b>{payload?.verdict?.label || 'incomplete'}</b>
          <p className="panel-copy">
            {payload?.reason || 'Comparison unavailable.'} Queue a matched classical analogue before treating this run as evidence.
          </p>
          {c?.job?.analogue_state === 'missing' && (
            <button className="primary" type="button" onClick={queueAnalogue}>Queue matched classical analogue</button>
          )}
        </div>
      )}

      {payload?.available && (
        <>
          <section className="panel">
            <div className="workspace-header">
              <div>
                <h3>Protocol</h3>
                <p className="panel-copy">{payload.verdict?.label}: {payload.verdict?.reason}</p>
              </div>
              <span className="badge best">{payload.verdict?.label}</span>
            </div>
            <div className="comparison-grid">
              <div>
                <div className="pill">candidate run</div>
                <h3><Link to={`/jobs/${c.job.id}`}>#{c.job.id} {c.job.preset_id}</Link></h3>
                <div className="kv compact">
                  <div className="k">dataset</div><div className="v">{c.job.dataset_name}</div>
                  <div className="k">seed</div><div className="v">{c.job.seed}</div>
                  <div className="k">steps</div><div className="v">{c.job.steps}</div>
                  <div className="k">target</div><div className="v">{c.job.device_target || 'auto'}</div>
                </div>
              </div>
              <div>
                <div className="pill">baseline run</div>
                <h3><Link to={`/jobs/${b.job.id}`}>#{b.job.id} {b.job.preset_id}</Link></h3>
                <div className="kv compact">
                  <div className="k">dataset</div><div className="v">{b.job.dataset_name}</div>
                  <div className="k">seed</div><div className="v">{b.job.seed}</div>
                  <div className="k">steps</div><div className="v">{b.job.steps}</div>
                  <div className="k">target</div><div className="v">{b.job.device_target || 'auto'}</div>
                </div>
              </div>
              <div className="flag-list">
                <Flag label="dataset" ok={flags.same_dataset} />
                <Flag label="seed" ok={flags.same_seed} />
                <Flag label="steps" ok={flags.same_steps} />
                <Flag label="eval" ok={flags.same_eval_interval} />
                <Flag label="device" ok={flags.same_device_target} />
                <Flag label="roles" ok={flags.role_validation} />
              </div>
            </div>
          </section>

          <EvidenceLadder ladder={payload.evidence_ladder} />

          <div className="workspace-grid">
            <RunLedger title="Candidate run ledger" manifest={c?.manifest} durability={c?.durability} resourceLedger={c?.resource_ledger} backendCapabilities={c?.backend_capabilities} />
            <RunLedger title="Baseline run ledger" manifest={b?.manifest} durability={b?.durability} resourceLedger={b?.resource_ledger} backendCapabilities={b?.backend_capabilities} />
          </div>

          <div className="workspace-grid">
            <section className="panel"><ModelDiagram graph={graphs.candidate} title="Candidate architecture" /></section>
            <section className="panel"><ModelDiagram graph={graphs.baseline} title="Baseline architecture" /></section>
          </div>

          <section className="panel">
            <h3>Metric Deltas</h3>
            <div className="comparison-grid">
              <Delta label="validation loss" value={payload.deltas?.val_loss} />
              <Delta label="validation perplexity" value={payload.deltas?.val_ppl} />
              <Delta label="bpc" value={payload.deltas?.val_bpc} />
              <Delta label="wall time" value={payload.deltas?.wall_seconds} />
              <Delta label="parameter count" value={payload.deltas?.n_params} />
              <div className="stat"><span className="k">parameter delta ratio</span><span className="v">{flags.parameter_delta_ratio == null ? '-' : fmt(flags.parameter_delta_ratio, 3)}</span></div>
              <div className="stat"><span className="k">resource improvement</span><span className="v">{fmt(payload.resource_normalized?.improvement)}</span></div>
              <div className="stat"><span className="k">improvement / extra second</span><span className="v">{fmt(payload.resource_normalized?.improvement_per_extra_second)}</span></div>
            </div>
          </section>

          <section className="panel chart-panel">
            <div className="pill chart-title">Comparison curve: {metric}</div>
            {series.length ? (
              <ResponsiveContainer width="100%" height="92%">
                <LineChart data={series} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                  <XAxis dataKey="step" tick={chartAxisTick} />
                  <YAxis tick={chartAxisTick} domain={['auto', 'auto']} />
                  <Tooltip {...chartTooltipProps} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="candidate" name="candidate" stroke={chartSeries.accent} dot={false} strokeWidth={2} connectNulls />
                  <Line type="monotone" dataKey="baseline" name="baseline" stroke={chartSeries.blue} dot={false} strokeWidth={2} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            ) : <p className="muted">Curves appear once either linked run starts logging.</p>}
          </section>
        </>
      )}
    </div>
  )
}
