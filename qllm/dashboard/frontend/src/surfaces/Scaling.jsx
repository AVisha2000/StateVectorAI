import { useMemo } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useScalingTest, useDiagnostics } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState, StatusTag, rowActivation } from '../lib/ui.jsx'
import { ScalingChart, Legend } from '../components/charts.jsx'
import { chartSeries } from '../chartTheme.js'
import { scalingChartRows, scalingSummary, representativeJobId, scalingFitView, scalingPointRows } from '../lib/scalingView.js'
import { fmtNum, fmtSeconds, DASH } from '../lib/format.js'

// Folds the old ScalingTest page into the redesign (plan §4: Runs → scaling
// view). Plots the metrics a qubit×depth sweep actually records — validation
// perplexity and simulator wall-time by scale. The barren-plateau
// gradient-variance-vs-qubit fit is a separate card that lights up only when the
// diagnostics/scaling-fit data ships (not computed for dashboard jobs today).
export default function Scaling() {
  const { groupId } = useParams()
  const navigate = useNavigate()
  const { data, isLoading, isError, error } = useScalingTest(groupId)

  const { rows, dropped } = useMemo(() => scalingChartRows(data?.points), [data])
  const pointRows = useMemo(() => scalingPointRows(data?.points), [data])
  const summary = useMemo(() => scalingSummary(data), [data])
  // The group-level barren-plateau fit comes from a member run's diagnostics.
  const repJobId = useMemo(() => representativeJobId(data?.points), [data])
  const { data: diag } = useDiagnostics(repJobId)
  const fit = useMemo(() => scalingFitView(diag?.diagnostics?.scaling_fit), [diag])

  if (isError) return <ErrorState error={error} label="Could not load this scaling group." />
  if (isLoading) return <Loading label="Loading scaling group…" />

  return (
    <>
      <PageHeader
        title="Scaling"
        sub="Does the signal survive as the circuit grows? Wall-time is simulator cost, never a QPU cost claim."
        actions={<Link className="btn" to="/runs">← Runs</Link>}
      />

      <div className="kpis" style={{ marginTop: 16 }}>
        <div className="kpi"><span className="microlabel">Progress</span><div className="v num">{summary.complete}/{summary.total}</div><div className="s">points complete</div></div>
        <div className="kpi"><span className="microlabel">Best val_ppl</span><div className="v num">{fmtNum(summary.best?.val_ppl, 2)}</div><div className="s">{summary.best?.n_qubits != null ? `q${summary.best.n_qubits}/d${summary.best.n_circuit_layers}` : DASH}</div></div>
        <div className="kpi"><span className="microlabel">Best wall</span><div className="v num">{fmtSeconds(summary.best?.wall_seconds)}</div><div className="s">simulator cost</div></div>
        <div className="kpi"><span className="microlabel">Params</span><div className="v num">{fmtNum(summary.best?.n_params, 0)}</div><div className="s">at best point</div></div>
      </div>

      {summary.protocolWarnings.length ? (
        <div className="notice crit" style={{ marginTop: 14 }}>
          {summary.protocolWarnings.map((w, i) => <div key={i}>{typeof w === 'string' ? w : w?.message}</div>)}
        </div>
      ) : null}

      <div className="grid2" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd"><h3>Validation perplexity by scale</h3><Legend items={[{ label: 'val_ppl', color: chartSeries.pink }]} /></div>
          <div className="bd chart-wrap">
            <ScalingChart rows={rows} xKey="x" series={[{ key: 'val_ppl', label: 'val_ppl', color: chartSeries.pink }]} />
          </div>
        </div>
        <div className="card">
          <div className="hd"><h3>Wall-time by scale <span className="hint" style={{ display: 'inline' }}>— simulator cost</span></h3></div>
          <div className="bd chart-wrap">
            <ScalingChart rows={rows} xKey="x" series={[{ key: 'wall_seconds', label: 'wall (s)', color: chartSeries.blue }]} logY />
          </div>
        </div>
      </div>

      {dropped ? (
        <p className="hint" style={{ marginTop: 10 }}>
          {dropped} point{dropped === 1 ? '' : 's'} omitted from the chart (rerun-required or no measured metric) — kept in the run table below, never silently dropped.
        </p>
      ) : null}

      {pointRows.length ? (
        <div className="card scroll-x" style={{ marginTop: 14 }}>
          <div className="hd">
            <h3>Runs in this sweep</h3>
            <span className="hint">{pointRows.length} point{pointRows.length === 1 ? '' : 's'} · every point kept, including chart-omitted ones</span>
          </div>
          <div className="bd" style={{ padding: '4px 16px 8px' }}>
            <table className="data">
              <thead>
                <tr><th>Grid</th><th>Run</th><th className="right-td">val_ppl</th><th className="right-td">wall (sim)</th><th className="right-td">params</th><th>Status</th><th /></tr>
              </thead>
              <tbody>
                {pointRows.map((p, i) => {
                  const open = p.jobId != null ? () => navigate(`/runs/${p.jobId}`) : null
                  return (
                    <tr key={p.jobId ?? i} className={open ? 'click' : undefined}
                      aria-label={open ? `Open run ${p.runName || p.jobId}` : undefined}
                      {...(open ? rowActivation(open) : {})}>
                      <td className="mono">{p.n_qubits != null ? `q${p.n_qubits}/d${p.n_circuit_layers}` : DASH}</td>
                      <td className="mono">{p.runName || (p.jobId != null ? `#${p.jobId}` : DASH)}</td>
                      <td className="right-td num">{fmtNum(p.val_ppl, 2)}</td>
                      <td className="right-td num">{fmtSeconds(p.wall_seconds)}</td>
                      <td className="right-td num">{fmtNum(p.n_params, 0)}</td>
                      <td>
                        {p.status ? <StatusTag status={p.status} /> : DASH}
                        {p.omitted ? <span className="tag warn sm" style={{ marginLeft: 6 }} title={p.omittedReason}>off-chart</span> : null}
                      </td>
                      <td className="right-td">{p.jobId != null ? <span className="hint">→</span> : null}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      <div className="card" style={{ marginTop: 14 }}>
        <div className="hd">
          <h3>Barren-plateau scaling fit</h3>
          <span className={`tag ${fit.available ? 'good' : 'plain'}`}>{fit.available ? 'measured' : 'unavailable'}</span>
        </div>
        <div className="bd">
          {fit.available ? (
            <>
              <div className="kpis">
                <div className="kpi">
                  <span className="microlabel">Variance decay / qubit</span>
                  <div className="v num">{fmtNum(fit.decayPerQubit, 3)}×</div>
                  <div className="s">factor per added qubit</div>
                </div>
                <div className="kpi">
                  <span className="microlabel">log Var slope</span>
                  <div className="v num">{fmtNum(fit.slope, 3)}</div>
                  <div className="s">fit of log Var[grad] vs qubits</div>
                </div>
                <div className="kpi">
                  <span className="microlabel">Exponential decay</span>
                  <div className="v" style={{ fontSize: 16, color: fit.exponentialDecay ? 'var(--warn)' : 'var(--good)' }}>
                    {fit.exponentialDecay ? 'detected' : 'not detected'}
                  </div>
                  <div className="s">plateau signature</div>
                </div>
              </div>
              <p className="hint" style={{ marginTop: 12 }}>
                From <span className="mono">gradient_variance_scaling_fit</span> over this group's persisted qubit counts — a
                <b> mechanism diagnostic</b>, not an advantage. Decay &lt; 1× per qubit signals a barren-plateau trend.
              </p>
            </>
          ) : (
            <p className="hint" style={{ margin: 0 }}>
              {fit.reason} — the fit needs at least two distinct persisted qubit counts in this group, read from{' '}
              <span className="mono">/jobs/{'{id}'}/diagnostics</span>.
            </p>
          )}
        </div>
      </div>
    </>
  )
}
