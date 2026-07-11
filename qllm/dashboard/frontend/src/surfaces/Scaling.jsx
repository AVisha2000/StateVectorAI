import { useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useScalingTest } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState } from '../lib/ui.jsx'
import { ScalingChart, Legend } from '../components/charts.jsx'
import { chartSeries } from '../chartTheme.js'
import { scalingChartRows, scalingSummary } from '../lib/scalingView.js'
import { fmtNum, fmtSeconds, DASH } from '../lib/format.js'

// Folds the old ScalingTest page into the redesign (plan §4: Runs → scaling
// view). Plots the metrics a qubit×depth sweep actually records — validation
// perplexity and simulator wall-time by scale. The barren-plateau
// gradient-variance-vs-qubit fit is a separate card that lights up only when the
// diagnostics/scaling-fit data ships (not computed for dashboard jobs today).
export default function Scaling() {
  const { groupId } = useParams()
  const { data, isLoading, isError, error } = useScalingTest(groupId)

  const { rows, dropped } = useMemo(() => scalingChartRows(data?.points), [data])
  const summary = useMemo(() => scalingSummary(data), [data])

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
          {dropped} point{dropped === 1 ? '' : 's'} omitted from the chart (rerun-required or no measured metric) — kept in the run table, never silently dropped.
        </p>
      ) : null}

      <div className="card" style={{ marginTop: 14 }}>
        <div className="hd"><h3>Barren-plateau scaling fit</h3><span className="tag plain">awaiting backend</span></div>
        <div className="bd">
          <p className="hint" style={{ margin: 0 }}>
            Gradient-variance-vs-qubit with an exponential-decay fit (from <span className="mono">gradient_variance_scaling_fit</span>)
            is not computed for dashboard jobs yet — it needs the proposed <span className="mono">/jobs/{'{id}'}/diagnostics</span>
            {' '}scaling data. This card renders the fit and trainable-floor band once that ships.
          </p>
        </div>
      </div>
    </>
  )
}
