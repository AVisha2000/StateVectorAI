import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { api } from '../api'
import { chartAxisTick, chartSeries, chartTooltipProps } from '../chartTheme'

export default function Live() {
  const [runs, setRuns] = useState([])
  const [jobs, setJobs] = useState([])
  const [sel, setSel] = useState(null)
  const [curve, setCurve] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const tick = () => {
      api.live().then(setRuns).catch((e) => setError(e.message))
      api.jobs().then(setJobs).catch((e) => setError(e.message))
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    if (!sel?.run_key) return
    const tick = () => api.liveCurve(sel.run_key).then(setCurve).catch((e) => setError(e.message))
    tick()
    const id = setInterval(tick, 2000)
    return () => clearInterval(id)
  }, [sel])

  const cancel = async (jobId) => {
    try {
      await api.cancelJob(jobId)
      setJobs(await api.jobs())
    } catch (e) {
      setError(e.message)
    }
  }

  const series = curve && curve.train_loss
    ? curve.train_loss.map((p) => ({
        step: p.step,
        train_loss: p.value,
        val_ppl: curve.val_ppl?.find((q) => q.step === p.step)?.value,
      }))
    : []

  return (
    <div>
      <h1>Live runs</h1>
      <h2>{jobs.filter((j) => j.status === 'running').length} running - {jobs.filter((j) => j.status === 'queued').length} queued</h2>
      {error && <div className="alert error">{error}</div>}

      <div className="panel" style={{ padding: 0, overflow: 'hidden' }}>
        <table>
          <thead><tr><th>Job</th><th>Preset</th><th>Dataset</th><th className="num">Steps</th><th>Status</th><th></th></tr></thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id} onClick={() => j.run_key && setSel(j)} style={{ cursor: j.run_key ? 'pointer' : 'default' }}>
                <td>#{j.id} {j.run_name}</td>
                <td>{j.preset_id}</td>
                <td>{j.dataset_name}</td>
                <td className="num">{j.steps}</td>
                <td><span className={`badge ${j.status}`}>{j.status}</span></td>
                <td className="num">
                  {(j.status === 'queued' || j.status === 'running') &&
                    <button className="small" onClick={(e) => { e.stopPropagation(); cancel(j.id) }}>Cancel</button>}
                </td>
              </tr>
            ))}
            {jobs.length === 0 && <tr><td colSpan="6">No lab jobs yet. Queue one from the Run tab.</td></tr>}
          </tbody>
        </table>
      </div>

      <h2>Training telemetry</h2>
      <div className="panel" style={{ padding: 0, overflow: 'hidden' }}>
        <table>
          <thead><tr><th>run</th><th>suite</th><th>variant</th><th className="num">step</th><th className="num">train loss</th><th className="num">val ppl</th><th>status</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_key} onClick={() => setSel(r)} style={{ cursor: 'pointer' }}>
                <td>{r.run_name}</td><td>{r.suite}</td><td>{r.variant}</td>
                <td className="num">{r.current_step}/{r.total_steps}
                  <div className="progress"><div style={{ width: `${100 * r.current_step / Math.max(r.total_steps, 1)}%` }} /></div></td>
                <td className="num">{r.last_train_loss?.toFixed(4) ?? '-'}</td>
                <td className="num">{r.last_val_ppl?.toFixed(3) ?? '-'}</td>
                <td><span className={`badge ${r.status}`}>{r.status}</span></td>
              </tr>
            ))}
            {runs.length === 0 && <tr><td colSpan="7">No telemetry yet.</td></tr>}
          </tbody>
        </table>
      </div>

      {sel && (
        <div className="panel" style={{ height: 320 }}>
          <div className="pill" style={{ marginBottom: 8 }}>{sel.run_name} - live curve</div>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={series} margin={{ top: 6, right: 16, bottom: 6, left: 0 }}>
              <XAxis dataKey="step" tick={chartAxisTick} />
              <YAxis yAxisId="l" tick={chartAxisTick} />
              <YAxis yAxisId="r" orientation="right" tick={chartAxisTick} />
              <Tooltip {...chartTooltipProps} />
              <Line yAxisId="l" type="monotone" dataKey="train_loss" stroke={chartSeries.blue} dot={false} strokeWidth={2} />
              <Line yAxisId="r" type="monotone" dataKey="val_ppl" stroke={chartSeries.accent} dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
