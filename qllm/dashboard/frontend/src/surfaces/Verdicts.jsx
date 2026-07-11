import { useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useJobs, useComparison, useVerdicts, isNotYetBuilt } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState, StatusTag } from '../lib/ui.jsx'
import { ComparisonCurve, Legend, ARM } from '../components/charts.jsx'
import { mergeComparison } from '../lib/curves.js'
import {
  scorecardRows,
  fairnessChecks,
  passedFairnessCount,
  ladderView,
  caveats,
} from '../lib/verdictView.js'
import { fmtNum, fmtPct, fmtSeconds, DASH } from '../lib/format.js'

// ---- List ----------------------------------------------------------------

function VerdictsList() {
  // Prefer the persistent verdict store; it 404s until the backend ships it, so
  // fall back to deriving candidate verdicts from jobs that carry a comparison.
  const verdicts = useVerdicts()
  const { data: jobs = [], isLoading, isError, error } = useJobs()

  const derived = useMemo(
    () =>
      jobs.filter(
        (j) =>
          (j.comparison_state === 'available' || j.comparison_state === 'linked') &&
          j.comparison_role !== 'analogue',
      ),
    [jobs],
  )

  const storeMissing = verdicts.isError && isNotYetBuilt(verdicts.error)
  const rows = Array.isArray(verdicts.data) ? verdicts.data : derived

  return (
    <>
      <PageHeader
        title="Verdicts"
        sub="Advantage adjudication bound to the claim ladder. Diagnostics are labeled as diagnostics — never advantage — and promotion is human-gated."
      />

      {storeMissing ? (
        <div className="notice" style={{ marginTop: 14 }}>
          The persistent verdict store (<span className="mono">/api/verdicts</span>) isn't shipped yet — showing
          verdicts <b>derived on the fly</b> from runs that have a matched comparison. They light up in full once the
          backend adds the store.
        </div>
      ) : null}

      {isError ? (
        <ErrorState error={error} />
      ) : isLoading ? (
        <Loading label="Loading verdicts…" />
      ) : rows.length === 0 ? (
        <div className="state" style={{ marginTop: 14 }}>
          No adjudicable comparisons yet. Queue a matched pair on the <Link to="/bench">Bench</Link>.
        </div>
      ) : (
        <div className="card scroll-x" style={{ marginTop: 14 }}>
          <table className="data">
            <thead>
              <tr><th>Comparison</th><th>Dataset</th><th>Claim</th><th>Status</th><th className="right-td" /></tr>
            </thead>
            <tbody>
              {rows.map((j) => (
                <tr key={j.id} className="click">
                  <td className="mono">#{j.id} {j.run_name}</td>
                  <td>{j.dataset_name || DASH}</td>
                  <td>{j.claim?.label ? <span className="tag plain">{j.claim.label}</span> : <span className="hint">unassigned</span>}</td>
                  <td><StatusTag status={j.status} /></td>
                  <td className="right-td"><Link className="btn sm" to={`/verdicts/${j.id}`}>Open →</Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

// ---- Detail --------------------------------------------------------------

function LadderRungs({ steps, claimLevel }) {
  if (!steps.length) return <span className="hint">No ladder rungs reported for this comparison.</span>
  const lastHit = steps.reduce((acc, s, i) => (s.ok ? i : acc), -1)
  return (
    <div className="ladder">
      {steps.map((s, i) => (
        <span
          key={s.key || i}
          className={`rung ${s.ok ? 'hit' : ''} ${i === lastHit ? 'cur' : ''}`}
          title={s.detail || s.caution || s.label}
        >
          {s.label}
        </span>
      ))}
      {claimLevel ? <span className="rung level">claim: {claimLevel}</span> : null}
    </div>
  )
}

function ScoreDelta({ row }) {
  if (row.favors === 'tie' || row.favors == null) return <span className="delta tie">matched</span>
  const arm = row.favors === 'quantum' ? 'q' : 'c'
  const label = row.favors === 'quantum' ? 'Q' : 'C'
  const pct = row.deltaPct != null ? ` ${fmtPct(row.deltaPct)}` : ''
  return <span className={`delta ${arm}`}>{label}{pct}</span>
}

function VerdictDetail({ id }) {
  const { data: comparison, isLoading, isError, error } = useComparison(id)

  const pplRows = useMemo(
    () => (comparison?.available ? mergeComparison(comparison.candidate?.curve, comparison.baseline?.curve, 'val_ppl') : []),
    [comparison],
  )
  const scorecard = useMemo(() => scorecardRows(comparison), [comparison])
  const fairness = useMemo(() => fairnessChecks(comparison), [comparison])
  const ladder = useMemo(() => ladderView(comparison), [comparison])
  const notes = useMemo(() => caveats(comparison), [comparison])

  if (isError) return <ErrorState error={error} label="Could not load this comparison." />
  if (isLoading) return <Loading label="Loading verdict…" />
  if (!comparison?.available) {
    return (
      <>
        <PageHeader title={<>Verdict · <span className="mono">#{id}</span></>} />
        <div className="state" style={{ marginTop: 14 }}>
          {comparison?.reason || 'This run has no matched comparison yet — queue its classical twin to adjudicate.'}
          <div style={{ marginTop: 10 }}><Link className="btn sm" to={`/runs/${id}`}>Back to run →</Link></div>
        </div>
      </>
    )
  }

  const { passed, total } = passedFairnessCount(fairness)

  return (
    <>
      <div className="banner">
        <div className="row">
          {ladder.label ? <span className="tag good">{ladder.label}</span> : <span className="tag plain">unadjudicated</span>}
          {ladder.assessmentStatus ? <span className="tag plain">{ladder.assessmentStatus}</span> : null}
          <span className="tag warn">awaiting human review</span>
          <span className="spacer" />
          <button className="btn" type="button">Reject with note</button>
          <button className="btn primary" type="button" disabled title="Claim promotion is human-gated">
            Promote to claim ladder…
          </button>
        </div>
        <h1 className="h1" style={{ marginTop: 12 }}>{ladder.reason || 'Candidate vs its matched control'}</h1>
        <p className="sub">
          Diagnostics are labeled diagnostics; no composite advantage score is produced, and a strong diagnostic can never
          raise the claim level. <b>Promotion up the ladder is yours alone.</b>
        </p>
        <LadderRungs steps={ladder.steps} claimLevel={ladder.claimLevel} />
      </div>

      <div className="grid32" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd">
            <h3>Perplexity vs steps</h3>
            <Legend items={[
              { label: 'candidate', color: ARM.quantum },
              { label: 'control', color: ARM.classical },
            ]} />
          </div>
          <div className="bd chart-wrap"><ComparisonCurve rows={pplRows} metricLabel="val_ppl" height={260} /></div>
          <p className="hint" style={{ padding: '0 16px 12px' }}>
            Single seed per arm — <span className="mono">paired_stats</span> and a min–max seed band need multi-seed data
            (a full study), not this pairwise comparison.
          </p>
        </div>

        <div className="card">
          <div className="hd"><h3>Advantage scorecard</h3><span className="hint">per dimension · no total</span></div>
          <div className="bd scroll-x" style={{ padding: '4px 16px 8px' }}>
            <table className="data">
              <thead>
                <tr>
                  <th>Dimension</th>
                  <th className="right-td" style={{ color: 'var(--q)' }}>Quantum</th>
                  <th className="right-td" style={{ color: 'var(--c)' }}>Classical</th>
                  <th className="right-td">Δ</th>
                </tr>
              </thead>
              <tbody>
                {scorecard.map((r) => (
                  <tr key={r.key}>
                    <td>{r.label}</td>
                    <td className={`right-td num ${r.favors === 'quantum' ? 'win-mark' : ''}`}>
                      {r.kind === 'cost' ? fmtSeconds(r.quantum) : fmtNum(r.quantum, r.kind === 'size' ? 0 : 2)}
                    </td>
                    <td className={`right-td num ${r.favors === 'classical' ? 'win-mark' : ''}`}>
                      {r.kind === 'cost' ? fmtSeconds(r.classical) : fmtNum(r.classical, r.kind === 'size' ? 0 : 2)}
                    </td>
                    <td className="right-td"><ScoreDelta row={r} /></td>
                  </tr>
                ))}
                {scorecard.length === 0 ? (
                  <tr><td colSpan="4" className="hint">No final metrics recorded on both arms yet.</td></tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="grid2" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd">
            <h3>Fairness &amp; controls</h3>
            <span className={`tag ${passed === total ? 'good' : 'warn'}`}>{passed} / {total} pass</span>
          </div>
          <div className="bd" style={{ paddingTop: 8 }}>
            {fairness.map((f) => (
              <div className="check" key={f.key}>
                <span className={`ok ${f.ok === true ? '' : f.ok === false ? 'bad' : 'plan'}`}>
                  {f.ok === true ? '✓' : f.ok === false ? '✕' : '·'}
                </span>
                {f.label}
              </div>
            ))}
          </div>
        </div>
        <div className="card">
          <div className="hd"><h3>Honest caveats</h3><span className="hint">auto-generated</span></div>
          <div className="bd" style={{ fontSize: 13, color: 'var(--ink2)' }}>
            {notes.length ? notes.map((w, i) => (
              <p style={{ margin: '0 0 8px' }} key={w.code || i}>• <b>{w.title}</b> — {w.message}</p>
            )) : <p className="hint" style={{ margin: 0 }}>No interpretation warnings on this comparison.</p>}
            <p className="hint" style={{ marginTop: 6 }}>Simulator timings only — never a QPU cost claim.</p>
          </div>
        </div>
      </div>
    </>
  )
}

export default function Verdicts() {
  const { id } = useParams()
  return id ? <VerdictDetail id={id} /> : <VerdictsList />
}
