// Reusable, token-themed charts for the redesigned surfaces (Run detail,
// Verdicts, Scaling). All colors come from CSS custom properties via
// chartTheme.js so light/dark both work; nothing here is hard-coded per page.
import {
  ResponsiveContainer,
  ComposedChart,
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
} from 'recharts'
import {
  chartAxisTick,
  chartGridStroke,
  chartTooltipProps,
  chartSeries,
} from '../chartTheme.js'

// Quantum / classical arm colors — the validated CVD-safe palette tokens.
export const ARM = Object.freeze({ quantum: 'var(--q)', classical: 'var(--c)' })

function EmptyChart({ label = 'No data to plot yet.' }) {
  return <div className="state" style={{ padding: '28px 12px' }}>{label}</div>
}

// A small legend chip row, matching the design reference.
export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label}>
          <span className="sw" style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  )
}

const AXIS = { tick: chartAxisTick, stroke: 'var(--axis)' }

// Candidate-vs-matched-control curve for one metric (e.g. validation perplexity).
// `rows` come from mergeComparison(); `candidateLabel`/`baselineLabel` name arms.
export function ComparisonCurve({
  rows,
  metricLabel = 'val_ppl',
  candidateColor = ARM.quantum,
  baselineColor = ARM.classical,
  height = 220,
}) {
  if (!rows || rows.length === 0) return <EmptyChart />
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={chartGridStroke} vertical={false} />
        <XAxis dataKey="step" {...AXIS} tickLine={false} />
        <YAxis {...AXIS} tickLine={false} width={46} domain={['auto', 'auto']}
          label={{ value: metricLabel, angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 11 }} />
        <Tooltip {...chartTooltipProps} />
        <Line type="monotone" dataKey="candidate" name="candidate" stroke={candidateColor}
          strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
        <Line type="monotone" dataKey="baseline" name="control" stroke={baselineColor}
          strokeWidth={2} strokeDasharray="4 3" dot={false} isAnimationActive={false} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  )
}

// Generic multi-series metric curve over training steps (mergeCurve rows).
export function MetricCurve({ rows, series, height = 220, logY = false }) {
  if (!rows || rows.length === 0) return <EmptyChart />
  const palette = [chartSeries.accent, chartSeries.pink, chartSeries.blue, chartSeries.amber, chartSeries.green]
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={chartGridStroke} vertical={false} />
        <XAxis dataKey="step" {...AXIS} tickLine={false} />
        <YAxis {...AXIS} tickLine={false} width={46}
          scale={logY ? 'log' : 'auto'} domain={logY ? ['auto', 'auto'] : ['auto', 'auto']} allowDataOverflow />
        <Tooltip {...chartTooltipProps} />
        {series.map((s, i) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label || s.key}
            stroke={s.color || palette[i % palette.length]} strokeWidth={2} dot={false}
            isAnimationActive={false} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

// Trainability / barren-plateau watch: a single series (e.g. grad_norm_ratio or
// gradient variance) over steps, optionally log-scaled with a reference line for
// the trainable-signal floor. The threshold is passed in (backend-owned), never
// invented here.
export function TrainabilityChart({
  rows,
  metricKey = 'grad_norm_ratio',
  label = 'grad norm ratio',
  color = chartSeries.pink,
  logY = false,
  threshold = null,
  thresholdLabel = 'floor',
  height = 220,
}) {
  if (!rows || rows.length === 0) return <EmptyChart label="No gradient trace recorded for this run." />
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={chartGridStroke} vertical={false} />
        <XAxis dataKey="step" {...AXIS} tickLine={false} />
        <YAxis {...AXIS} tickLine={false} width={54}
          scale={logY ? 'log' : 'auto'} domain={['auto', 'auto']} allowDataOverflow />
        <Tooltip {...chartTooltipProps} />
        {threshold != null ? (
          <ReferenceLine y={threshold} stroke="var(--crit)" strokeDasharray="5 4"
            label={{ value: thresholdLabel, fill: 'var(--crit)', fontSize: 10, position: 'insideTopRight' }} />
        ) : null}
        <Line type="monotone" dataKey={metricKey} name={label} stroke={color}
          strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  )
}

// Seed-band ribbon (min–max) + mean line for one arm. Rows from seedBand().
// Used by Verdicts when per-seed data is available; degrades to empty otherwise.
export function SeedBandChart({ rows, color = ARM.quantum, label = 'mean', height = 240 }) {
  if (!rows || rows.length === 0) return <EmptyChart label="Per-seed band needs multi-seed data." />
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={chartGridStroke} vertical={false} />
        <XAxis dataKey="step" {...AXIS} tickLine={false} />
        <YAxis {...AXIS} tickLine={false} width={46} domain={['auto', 'auto']} />
        <Tooltip {...chartTooltipProps} />
        <Area type="monotone" dataKey="band" name="min–max" stroke="none" fill={color}
          fillOpacity={0.15} isAnimationActive={false} connectNulls />
        <Line type="monotone" dataKey="mean" name={label} stroke={color} strokeWidth={2}
          dot={false} isAnimationActive={false} connectNulls />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

// Scaling view: a measured series across a scale axis (qubits or scale label),
// with optional dashed extrapolation from a fit and an optional below-threshold
// band. `xKey` names the x field; series is a list of {key,label,color,dashed}.
export function ScalingChart({
  rows,
  xKey = 'x',
  series,
  logY = false,
  threshold = null,
  belowThresholdFloor = null,
  height = 300,
}) {
  if (!rows || rows.length === 0) return <EmptyChart label="No scaling points measured yet." />
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={chartGridStroke} vertical={false} />
        <XAxis dataKey={xKey} {...AXIS} tickLine={false} />
        <YAxis {...AXIS} tickLine={false} width={54}
          scale={logY ? 'log' : 'auto'} domain={['auto', 'auto']} allowDataOverflow />
        <Tooltip {...chartTooltipProps} />
        {threshold != null && belowThresholdFloor != null ? (
          <ReferenceArea y1={belowThresholdFloor} y2={threshold} fill="var(--crit)" fillOpacity={0.08}
            stroke="none" />
        ) : null}
        {threshold != null ? (
          <ReferenceLine y={threshold} stroke="var(--crit)" strokeDasharray="5 4"
            label={{ value: 'trainable floor', fill: 'var(--crit)', fontSize: 10, position: 'insideTopRight' }} />
        ) : null}
        {series.map((s) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label || s.key}
            stroke={s.color || chartSeries.accent} strokeWidth={2}
            strokeDasharray={s.dashed ? '5 4' : undefined}
            dot={s.dashed ? false : { r: 2 }} isAnimationActive={false} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
