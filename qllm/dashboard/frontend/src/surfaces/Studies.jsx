import { useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ReferenceArea,
} from 'recharts'
import { useStudies, useStudy } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState, StatusTag } from '../lib/ui.jsx'
import { chartAxisTick, chartGridStroke, chartTooltipProps, chartSeries } from '../chartTheme.js'
import { studySummary, deltaPairs, winConsistency, studyLadder, studyJobs, studyCaveats } from '../lib/studyView.js'
import { fmtNum, fmtPct, DASH } from '../lib/format.js'

const AXIS = { tick: chartAxisTick, stroke: 'var(--axis)' }

// ---- List ----------------------------------------------------------------

function StudiesList() {
  const { data, isLoading, isError, error } = useStudies()
  const studies = Array.isArray(data) ? data : data?.studies ?? []

  return (
    <>
      <PageHeader
        title="Studies — multi-seed rigor"
        sub="Where a claim is tested across seeds, not just one run. A study reports per-pair candidate-vs-control deltas and their spread — replication distinct from the claim, no composite score."
      />
      {isError ? (
        <ErrorState error={error} />
      ) : isLoading ? (
        <Loading label="Loading studies…" />
      ) : studies.length === 0 ? (
        <div className="state" style={{ marginTop: 16 }}>
          No studies yet. Queue a <b>full study</b> from the <Link to="/bench">Bench</Link> to test a claim across seeds.
        </div>
      ) : (
        <div className="card scroll-x" style={{ marginTop: 16 }}>
          <table className="data">
            <thead>
              <tr><th>Study</th><th>Evidence</th><th className="right-td">Fair pairs</th><th className="right-td">Wins</th><th className="right-td">Mean Δ ppl</th><th className="right-td" /></tr>
            </thead>
            <tbody>
              {studies.map((s) => {
                const e = s.evidence || {}
                return (
                  <tr key={s.id} className="click">
                    <td><b>{s.name || `#${s.id}`}</b>{s.research_question ? <div className="hint">{s.research_question}</div> : null}</td>
                    <td>{e.label ? <span className="tag plain">{e.label}</span> : <span className="hint">pending</span>}</td>
                    <td className="right-td num">{Number.isFinite(e.fair_pairs) ? e.fair_pairs : DASH}</td>
                    <td className="right-td num">{Number.isFinite(e.wins) ? e.wins : DASH}</td>
                    <td className="right-td num">{fmtNum(e.mean_delta_val_ppl, 3)}</td>
                    <td className="right-td"><Link className="btn sm" to={`/studies/${s.id}`}>Open →</Link></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

// ---- Detail --------------------------------------------------------------

function DeltaStrip({ pairs, meanDelta, stdDelta }) {
  if (!pairs.length) return <div className="state" style={{ padding: '24px 8px' }}>No fair, complete pairs to plot yet.</div>
  const band = meanDelta != null && stdDelta != null ? [meanDelta - stdDelta, meanDelta + stdDelta] : null
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 10, right: 16, bottom: 8, left: 4 }}>
        <CartesianGrid stroke={chartGridStroke} />
        <XAxis type="number" dataKey="i" name="pair" {...AXIS} tickLine={false} domain={[0.5, pairs.length + 0.5]}
          allowDecimals={false} label={{ value: 'fair pair', position: 'insideBottom', offset: -2, fill: 'var(--muted)', fontSize: 11 }} />
        <YAxis type="number" dataKey="delta" name="Δ val_ppl" {...AXIS} tickLine={false} width={54}
          label={{ value: 'Δ val_ppl (cand − ctrl)', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 11 }} />
        <Tooltip {...chartTooltipProps} cursor={{ strokeDasharray: '3 3', stroke: 'var(--chart-cursor)' }} />
        {band ? <ReferenceArea y1={band[0]} y2={band[1]} fill="var(--accent)" fillOpacity={0.1} stroke="none" /> : null}
        <ReferenceLine y={0} stroke="var(--axis)" strokeDasharray="5 4"
          label={{ value: 'parity', fill: 'var(--muted)', fontSize: 10, position: 'insideTopRight' }} />
        {meanDelta != null ? <ReferenceLine y={meanDelta} stroke="var(--accent)" strokeDasharray="2 2" /> : null}
        <Scatter data={pairs} fill={chartSeries.pink} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

function StudyDetail({ id }) {
  const { data: study, isLoading, isError, error } = useStudy(id)
  const summary = useMemo(() => studySummary(study), [study])
  const pairs = useMemo(() => deltaPairs(study), [study])
  const consistency = useMemo(() => winConsistency(study), [study])
  const ladder = useMemo(() => studyLadder(study), [study])
  const jobs = useMemo(() => studyJobs(study), [study])
  const caveats = useMemo(() => studyCaveats(study), [study])

  if (isError) return <ErrorState error={error} label="Could not load this study." />
  if (isLoading) return <Loading label="Loading study…" />
  if (!study) return <ErrorState error={{ message: `Study ${id} not found.` }} label="Study not found." />

  return (
    <>
      <PageHeader
        title={<>{study.name || `Study #${id}`}</>}
        sub={study.research_question || 'Multi-seed replication of a candidate against its matched control.'}
        actions={<Link className="btn" to="/studies">← Studies</Link>}
      />

      <div className="kpis" style={{ marginTop: 16 }}>
        <div className="kpi"><span className="microlabel">Evidence</span><div className="v" style={{ fontSize: 15 }}>{summary.label || 'pending'}</div><div className="s">claim label</div></div>
        <div className="kpi"><span className="microlabel">Replication</span><div className="v num">{summary.fairPairs}</div><div className="s">fair pairs · {summary.completePairs} complete</div></div>
        <div className="kpi"><span className="microlabel">Consistency</span><div className="v num">{consistency.total ? `${consistency.wins}/${consistency.total}` : DASH}</div><div className="s">{consistency.fraction != null ? `${fmtPct(consistency.fraction)} candidate wins` : 'no pairs'}</div></div>
        <div className="kpi"><span className="microlabel">Mean Δ val_ppl</span><div className="v num" style={{ color: summary.meanDelta != null ? (summary.meanDelta < 0 ? 'var(--good)' : 'var(--crit)') : 'var(--ink)' }}>{fmtNum(summary.meanDelta, 3)}</div><div className="s">± {fmtNum(summary.stdDelta, 3)} across seeds</div></div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="hd"><h3>Per-pair candidate − control</h3><span className="hint">each point a fair seed/cell pair · band = mean ± sd · below parity = candidate lower ppl</span></div>
        <div className="bd chart-wrap"><DeltaStrip pairs={pairs} meanDelta={summary.meanDelta} stdDelta={summary.stdDelta} /></div>
        <p className="hint" style={{ padding: '0 16px 12px' }}>
          This is the multi-seed spread, not a single verdict — the value single-seed comparisons can't give. No composite
          advantage score; wall-time (elsewhere) is simulator cost.
        </p>
      </div>

      <div className="grid2" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd"><h3>Evidence ladder</h3></div>
          <div className="bd" style={{ paddingTop: 8 }}>
            {ladder.length ? ladder.map((r, i) => (
              <div className="check" key={r.key || i}>
                <span className={`ok ${r.ok || r.met ? '' : 'plan'}`}>{r.ok || r.met ? '✓' : '·'}</span>
                {r.label || r.key}{r.detail ? <span className="hint"> — {r.detail}</span> : null}
              </div>
            )) : <p className="hint" style={{ margin: 0 }}>No ladder reported.</p>}
            {caveats.length ? (
              <div style={{ marginTop: 10 }}>
                {caveats.map((w, i) => <p className="hint" style={{ margin: '0 0 6px' }} key={w.code || i}>• <b>{w.title || w.code}</b>{w.message ? ` — ${w.message}` : ''}</p>)}
              </div>
            ) : null}
            {summary.rerunRequiredPairs ? (
              <p className="hint" style={{ marginTop: 6, color: 'var(--warn)' }}>{summary.rerunRequiredPairs} pair(s) need a rerun before they count.</p>
            ) : null}
          </div>
        </div>
        <div className="card scroll-x">
          <div className="hd"><h3>Study runs</h3><span className="hint">{jobs.length} jobs</span></div>
          <div className="bd" style={{ padding: '4px 16px 8px' }}>
            <table className="data">
              <thead><tr><th>Grid</th><th className="right-td">val_ppl</th><th>Status</th></tr></thead>
              <tbody>
                {jobs.map((j, i) => (
                  <tr key={j.job?.id ?? j.id ?? i}>
                    <td className="mono">{j.study_sweep?.n_qubits != null ? `q${j.study_sweep.n_qubits}/d${j.study_sweep.n_circuit_layers}` : DASH}</td>
                    <td className="right-td num">{fmtNum(j.final_run?.val_ppl, 2)}</td>
                    <td>{j.status ? <StatusTag status={j.status} /> : DASH}</td>
                  </tr>
                ))}
                {jobs.length === 0 ? <tr><td colSpan="3" className="hint">No jobs linked yet.</td></tr> : null}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  )
}

export default function Studies() {
  const { id } = useParams()
  return id ? <StudyDetail id={id} /> : <StudiesList />
}
