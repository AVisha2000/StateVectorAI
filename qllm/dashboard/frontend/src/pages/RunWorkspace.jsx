import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { api } from '../api'
import { chartAxisTick, chartSeries, chartTooltipProps } from '../chartTheme'
import ModelDiagram from '../components/ModelDiagram'
import EvidenceSummary from '../components/EvidenceSummary'
import EvidenceWarnings from '../components/EvidenceWarnings'
import RunLedger from '../components/RunLedger'

const METRICS = ['train_loss', 'val_loss', 'val_ppl', 'val_bpc', 'grad_norm_ratio']

function fmt(value, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) return '-'
  const n = Number(value)
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) >= 100) return n.toFixed(2)
  return n.toFixed(digits)
}

function mergeCurve(curve = {}, names = METRICS) {
  const byStep = new Map()
  names.forEach((name) => {
    ;(curve[name] || []).forEach((point) => {
      const row = byStep.get(point.step) || { step: point.step }
      row[name] = point.value
      byStep.set(point.step, row)
    })
  })
  return Array.from(byStep.values()).sort((a, b) => a.step - b.step)
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

function canCancel(job) {
  return job?.status === 'queued' || job?.status === 'running'
}

function Stat({ label, value, hint }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {hint && <div className="muted">{hint}</div>}
    </div>
  )
}

function CurvePanel({ title, data, lines, empty }) {
  if (!data.length) {
    return (
      <div className="panel chart-panel empty-chart">
        <div className="pill">{title}</div>
        <p className="muted">{empty || 'No points logged yet.'}</p>
      </div>
    )
  }
  return (
    <div className="panel chart-panel">
      <div className="pill chart-title">{title}</div>
      <ResponsiveContainer width="100%" height="92%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <XAxis dataKey="step" tick={chartAxisTick} />
          <YAxis tick={chartAxisTick} domain={['auto', 'auto']} />
          <Tooltip {...chartTooltipProps} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {lines.map((line) => (
            <Line
              key={line.key}
              type="monotone"
              dataKey={line.key}
              name={line.name}
              stroke={line.color}
              dot={false}
              strokeWidth={2}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function Delta({ label, value, lowerBetter = true }) {
  const isGood = value != null && (lowerBetter ? value < 0 : value > 0)
  const isBad = value != null && (lowerBetter ? value > 0 : value < 0)
  return (
    <div className="stat">
      <span className="k">{label}</span>
      <span className={`v ${isGood ? 'good-text' : ''} ${isBad ? 'warn-text' : ''}`}>
        {value == null ? '-' : `${value > 0 ? '+' : ''}${fmt(value)}`}
      </span>
    </div>
  )
}

export default function RunWorkspace() {
  const { id } = useParams()
  const [payload, setPayload] = useState(null)
  const [graph, setGraph] = useState(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [testInfo, setTestInfo] = useState(null)
  const [testResult, setTestResult] = useState(null)
  const [testBusy, setTestBusy] = useState(false)
  const [testPrompt, setTestPrompt] = useState('\n')
  const [testTokens, setTestTokens] = useState(120)
  const [testTemperature, setTestTemperature] = useState(0.8)

  const refresh = () => api.workspace(id)
    .then((nextPayload) => {
      setPayload(nextPayload)
      setError('')
    })
    .catch((e) => setError(e.message))

  useEffect(() => {
    refresh()
    api.jobGraph(id).then(setGraph).catch(() => {})
    api.modelTests(id).then(setTestInfo).catch(() => {})
    const timer = setInterval(refresh, 2000)
    return () => clearInterval(timer)
  }, [id])

  const job = payload?.job
  const live = payload?.live
  const finalRun = payload?.final_run
  const preset = payload?.preset
  const dataset = payload?.dataset
  const comparison = payload?.comparison
  const analogue = job?.analogue
  const artifactDir = testInfo?.artifacts?.directory || job?.artifact_dir || `results/${job?.run_name || '-'}`
  const summaryPath = testInfo?.artifacts?.summary_path || `${artifactDir}/summary.json`
  const paramsPath = testInfo?.artifacts?.params_path || `${artifactDir}/params.msgpack`
  const series = useMemo(() => mergeCurve(payload?.curve), [payload])
  const comparisonMetric = comparison?.candidate?.curve?.val_ppl?.length || comparison?.baseline?.curve?.val_ppl?.length
    ? 'val_ppl'
    : 'train_loss'
  const comparisonSeries = useMemo(() => mergeComparison(
    comparison?.candidate?.curve,
    comparison?.baseline?.curve,
    comparisonMetric,
  ), [comparison, comparisonMetric])

  if (!payload && !error) return <div className="loading">Loading run workspace...</div>
  if (!payload && error) {
    return (
      <div className="panel empty-state">
        <h1>Run workspace unavailable</h1>
        <div className="alert error">{error}</div>
        <button type="button" onClick={refresh}>Retry</button>
      </div>
    )
  }

  const progressStep = live?.current_step ?? (job?.status === 'done' ? job?.steps : 0)
  const totalSteps = live?.total_steps ?? job?.steps ?? 1
  const progress = Math.min(100, Math.round((100 * progressStep) / Math.max(totalSteps, 1)))

  const cancel = async () => {
    setError('')
    try {
      await api.cancelJob(job.id)
      refresh()
    } catch (e) {
      setError(e.message)
    }
  }

  const queueAnalogue = async () => {
    setError(''); setNotice('')
    try {
      const updated = await api.queueClassicalAnalogue(job.id)
      setNotice(`Queued classical analogue job #${updated.comparison_job?.id || updated.analogue_job_id}.`)
      refresh()
    } catch (e) {
      setError(e.message)
    }
  }

  const runManualTest = async () => {
    setError(''); setNotice(''); setTestResult(null); setTestBusy(true)
    try {
      const result = await api.runModelTest(job.id, {
        prompt: testPrompt,
        max_new_tokens: Number(testTokens),
        temperature: Number(testTemperature),
      })
      setTestResult(result)
      api.modelTests(id).then(setTestInfo).catch(() => {})
    } catch (e) {
      setError(e.message)
    } finally {
      setTestBusy(false)
    }
  }

  return (
    <div>
      {error && <div className="alert error">{error}</div>}
      {notice && <div className="alert good">{notice}</div>}
      {job && (
        <>
          <div className="workspace-header">
            <div>
              <h1>#{job.id} {job.run_name}</h1>
              <h2>{job.preset_id} on {job.dataset_name} - seed {job.seed} - target {job.device_target || 'auto'}</h2>
            </div>
            <div className="header-actions">
              <span className={`badge ${job.status}`}>{job.status}</span>
              <span className={`badge ${job.analogue_state === 'missing' ? 'error' : job.analogue_state === 'done' ? 'done' : ''}`}>
                analogue: {job.analogue_state || 'none'}
              </span>
              {canCancel(job) && <button className="small" onClick={cancel}>Cancel</button>}
            </div>
          </div>

          <EvidenceWarnings warnings={payload?.interpretation_warnings} />
          <EvidenceSummary evidence={{ ...payload, ...(comparison || {}), interpretation_warnings: payload?.interpretation_warnings }} title="Run evidence contract" />
          <RunLedger manifest={payload?.manifest} durability={payload?.durability} resourceLedger={payload?.resource_ledger} backendCapabilities={payload?.backend_capabilities} />

          <div className="panel">
            <div className="stat-row">
              <Stat label="Progress" value={`${progressStep}/${totalSteps}`} hint={`${progress}% complete`} />
              <Stat label="Dataset" value={dataset?.name || job.dataset_name} hint={dataset?.source_type} />
              <Stat label="Preset" value={preset?.label || job.preset_id} hint={preset?.kind} />
              <Stat label="Device target" value={job.device_target || 'auto'} hint={live ? 'active telemetry' : 'waiting for worker'} />
            </div>
            <div className="progress wide"><div style={{ width: `${progress}%` }} /></div>
          </div>

          {job.error && (
            <div className="alert error">
              <b>Job error</b>
              <pre className="code-block">{job.error}</pre>
            </div>
          )}

          <div className="workspace-grid">
            <section className="panel">
              <h3>Model description</h3>
              <p>{preset?.description}</p>
              <div className="kv compact">
                <div className="k">architecture</div><div className="v">{preset?.architecture}</div>
                <div className="k">quantum role</div><div className="v">{preset?.quantum_role}</div>
                <div className="k">recommended use</div><div className="v">{preset?.recommended_use}</div>
                <div className="k">risks</div><div className="v">{preset?.risks}</div>
                <div className="k">classical analogue</div><div className="v">{job.analogue_preset_id || job.analogue_model_spec_id || preset?.classical_twin_id || 'none'}</div>
              </div>
            </section>

            <section className="panel">
              <ModelDiagram graph={graph} />
            </section>

            <section className="panel">
              <h3>Results</h3>
              <div className="kv">
                <div className="k">current step</div><div className="v">{progressStep}</div>
                <div className="k">last train loss</div><div className="v">{fmt(live?.last_train_loss)}</div>
                <div className="k">last val ppl</div><div className="v">{fmt(live?.last_val_ppl)}</div>
                <div className="k">final val loss</div><div className="v">{fmt(finalRun?.val_loss)}</div>
                <div className="k">final val ppl</div><div className="v">{fmt(finalRun?.val_ppl)}</div>
                <div className="k">final val bpc</div><div className="v">{fmt(finalRun?.val_bpc)}</div>
                <div className="k">parameters</div><div className="v">{finalRun?.n_params?.toLocaleString?.() || '-'}</div>
                <div className="k">wall time</div><div className="v">{finalRun?.wall_seconds ? `${fmt(finalRun.wall_seconds, 2)}s` : '-'}</div>
              </div>
            </section>
          </div>

          <div className="chart-grid">
            <CurvePanel
              title="Loss curves"
              data={series}
              lines={[
                { key: 'train_loss', name: 'train loss', color: chartSeries.blue },
                { key: 'val_loss', name: 'val loss', color: chartSeries.accent },
              ]}
            />
            <CurvePanel
              title="Validation and quantum diagnostics"
              data={series}
              lines={[
                { key: 'val_ppl', name: 'val ppl', color: chartSeries.green },
                { key: 'val_bpc', name: 'val bpc', color: chartSeries.amber },
                { key: 'grad_norm_ratio', name: 'grad norm ratio', color: chartSeries.pink },
              ]}
            />
          </div>

          <section className="panel">
            <h3>Classical comparison</h3>
            {!comparison?.available && (
              <>
                <p className="muted">{comparison?.reason || 'No comparison is linked to this job.'}</p>
                {job.analogue_state === 'missing' && (
                  <div className="alert">
                    <b>Matched classical analogue is missing.</b>
                    <p className="panel-copy">
                      Queue one before treating this run as evidence of quantum advantage.
                      {analogue?.reason ? ` ${analogue.reason}` : ''}
                    </p>
                    <button className="primary" type="button" onClick={queueAnalogue}>Queue matched classical analogue</button>
                  </div>
                )}
              </>
            )}
            {comparison?.available && (
              <>
              {comparison.metric_contract?.rerun_required && <div className="alert error">{comparison.metric_contract.limitation}</div>}
              <div className="comparison-grid">
                <div>
                  <div className="pill">candidate</div>
                  <h3>
                    <Link to={`/jobs/${comparison.candidate.job.id}`}>
                      #{comparison.candidate.job.id} {comparison.candidate.job.preset_id}
                    </Link>
                  </h3>
                  <span className={`badge ${comparison.candidate.job.status}`}>{comparison.candidate.job.status}</span>
                  <div className="kv compact">
                    <div className="k">val ppl</div><div className="v">{fmt(comparison.candidate.final_run?.val_ppl)}</div>
                    <div className="k">val loss</div><div className="v">{fmt(comparison.candidate.final_run?.val_loss)}</div>
                    <div className="k">wall</div><div className="v">{comparison.candidate.final_run?.wall_seconds ? `${fmt(comparison.candidate.final_run.wall_seconds, 2)}s` : '-'}</div>
                  </div>
                </div>
                <div>
                  <div className="pill">classical baseline</div>
                  <h3>
                    <Link to={`/jobs/${comparison.baseline.job.id}`}>
                      #{comparison.baseline.job.id} {comparison.baseline.job.preset_id}
                    </Link>
                  </h3>
                  <span className={`badge ${comparison.baseline.job.status}`}>{comparison.baseline.job.status}</span>
                  <div className="kv compact">
                    <div className="k">val ppl</div><div className="v">{fmt(comparison.baseline.final_run?.val_ppl)}</div>
                    <div className="k">val loss</div><div className="v">{fmt(comparison.baseline.final_run?.val_loss)}</div>
                    <div className="k">wall</div><div className="v">{comparison.baseline.final_run?.wall_seconds ? `${fmt(comparison.baseline.final_run.wall_seconds, 2)}s` : '-'}</div>
                  </div>
                </div>
                {!comparison.metric_contract?.rerun_required && <div>
                  <div className="pill">candidate minus baseline</div>
                  <Delta label="val ppl" value={comparison.deltas?.val_ppl} />
                  <Delta label="val loss" value={comparison.deltas?.val_loss} />
                  <Delta label="val bpc" value={comparison.deltas?.val_bpc} />
                  <Delta label="wall seconds" value={comparison.deltas?.wall_seconds} />
                  <Delta label="parameters" value={comparison.deltas?.n_params} />
                </div>}
              </div>
              </>
            )}
          </section>

          {comparison?.available && !comparison.metric_contract?.rerun_required && (
            <CurvePanel
              title={`Comparison curve: ${comparisonMetric}`}
              data={comparisonSeries}
              empty="Comparison curves appear once either linked run starts logging."
              lines={[
                { key: 'candidate', name: 'candidate', color: chartSeries.accent },
                { key: 'baseline', name: 'classical baseline', color: chartSeries.blue },
              ]}
            />
          )}

          <div className="workspace-grid">
            <section className="panel">
              <h3>Dataset provenance</h3>
              <div className="kv">
                <div className="k">source</div><div className="v">{dataset?.source}</div>
                <div className="k">split</div><div className="v">{dataset?.split || '-'}</div>
                <div className="k">text column</div><div className="v">{dataset?.text_column || '-'}</div>
                <div className="k">rows</div><div className="v">{dataset?.n_rows?.toLocaleString?.() || '-'}</div>
                <div className="k">chars</div><div className="v">{dataset?.n_chars?.toLocaleString?.() || '-'}</div>
              </div>
            </section>

            <section className="panel">
              <h3>Artifacts</h3>
              <div className="kv">
                <div className="k">summary</div><div className="v mono">{summaryPath}</div>
                <div className="k">params</div><div className="v mono">{paramsPath}</div>
                <div className="k">run key</div><div className="v">{job.run_key || '-'}</div>
              </div>
            </section>
          </div>

          <section className="panel">
            <div className="workspace-header">
              <div>
                <h3>Trained Model Testing</h3>
                <p className="panel-copy">Manual prompt generation is available when a completed text run has reloadable `params.msgpack` artifacts.</p>
              </div>
              <span className={`badge ${testInfo?.supported_tests?.prompt_generation ? 'done' : 'cancelled'}`}>
                {testInfo?.supported_tests?.prompt_generation ? 'generation ready' : 'generation unavailable'}
              </span>
            </div>
            {testInfo && (
              <div className="kv compact">
                <div className="k">summary artifact</div><div className="v">{testInfo.artifacts?.summary_exists ? 'found' : 'missing'}</div>
                <div className="k">params artifact</div><div className="v">{testInfo.artifacts?.params_exists ? 'found' : 'missing'}</div>
                <div className="k">artifact directory</div><div className="v mono">{testInfo.artifacts?.directory}</div>
                <div className="k">summary review</div><div className="v">{testInfo.supported_tests?.summary_review ? 'available' : 'unavailable'}</div>
              </div>
            )}
            {testInfo?.unsupported_reasons?.length > 0 && (
              <div className="alert">
                {testInfo.unsupported_reasons.join('; ')}
              </div>
            )}
            {testInfo?.summary && (
              <div className="stat-row">
                <Stat label="Summary val ppl" value={fmt(testInfo.summary.val_ppl)} />
                <Stat label="Summary steps" value={testInfo.summary.steps ?? '-'} />
                <Stat label="Summary params" value={testInfo.summary.n_params?.toLocaleString?.() || '-'} />
                <Stat label="Summary wall" value={testInfo.summary.wall_seconds == null ? '-' : `${fmt(testInfo.summary.wall_seconds, 2)}s`} />
              </div>
            )}
            <div className="form-grid">
              <label>Prompt<input value={testPrompt} onChange={(e) => setTestPrompt(e.target.value)} /></label>
              <label>New tokens<input type="number" min="1" max="240" value={testTokens} onChange={(e) => setTestTokens(e.target.value)} /></label>
              <label>Temperature<input type="number" min="0.1" max="2" step="0.1" value={testTemperature} onChange={(e) => setTestTemperature(e.target.value)} /></label>
            </div>
            <button
              className="primary"
              type="button"
              disabled={!testInfo?.supported_tests?.prompt_generation || testBusy}
              onClick={runManualTest}
            >
              {testBusy ? 'Running test...' : 'Generate from trained model'}
            </button>
            {testResult && (
              <div className={`alert ${testResult.ok ? 'good' : 'error'}`}>
                <b>{testResult.ok ? 'Generated sample' : 'Test unavailable'}</b>
                {testResult.ok ? (
                  <pre className="code-block">{testResult.generated_text}</pre>
                ) : (
                  <p>{testResult.reason}</p>
                )}
              </div>
            )}
          </section>

          <details className="panel">
            <summary className="pill">Config summary</summary>
            <div className="kv config-list">
              {Object.entries(job.config || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'contents' }}>
                  <div className="k">{k}</div><div className="v">{String(v)}</div>
                </div>
              ))}
            </div>
          </details>
        </>
      )}
    </div>
  )
}
