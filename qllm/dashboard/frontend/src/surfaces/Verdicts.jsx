import { useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useJobs, useComparison, useVerdicts, useVerdict, isNotYetBuilt } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState, StatusTag } from '../lib/ui.jsx'
import { ComparisonCurve, Legend, ARM } from '../components/charts.jsx'
import { mergeComparison } from '../lib/curves.js'
import {
  scorecardRows,
  fairnessChecks,
  passedFairnessCount,
  ladderView,
  caveats,
  snapshotClaim,
  snapshotScorecardRows,
  snapshotMetricType,
  booleanChecks,
  snapshotCaveats,
  revisionHistory,
} from '../lib/verdictView.js'
import { fmtNum, fmtPct, fmtSeconds, DASH } from '../lib/format.js'

// ---- List ----------------------------------------------------------------

function VerdictsList() {
  // Prefer the persistent verdict store; it 404s until the backend merges to
  // main, so fall back to deriving candidate verdicts from jobs with a comparison.
  const verdicts = useVerdicts()
  const { data: jobs = [], isLoading, isError, error } = useJobs()

  const snapshots = Array.isArray(verdicts.data?.snapshots) ? verdicts.data.snapshots : null
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

  return (
    <>
      <PageHeader
        title="Verdicts"
        sub="Advantage adjudication bound to the claim ladder. Diagnostics are labeled as diagnostics — never advantage — and promotion is human-gated."
      />

      {storeMissing || (!snapshots && !isLoading) ? (
        <div className="notice" style={{ marginTop: 14 }}>
          The persistent verdict store (<span className="mono">/api/verdicts</span>) isn't reachable on this branch yet —
          showing verdicts <b>derived on the fly</b> from runs that have a matched comparison. They become append-only
          snapshots once the backend merges to <span className="mono">main</span>.
        </div>
      ) : null}

      {snapshots ? (
        snapshots.length === 0 ? (
          <div className="state" style={{ marginTop: 14 }}>No verdict snapshots recorded yet.</div>
        ) : (
          <div className="card scroll-x" style={{ marginTop: 14 }}>
            <table className="data">
              <thead>
                <tr>
                  <th>Verdict</th><th>Claim level</th><th>Claim status</th><th>Replication</th>
                  <th>Assessment</th><th className="right-td">Rev</th><th className="right-td" />
                </tr>
              </thead>
              <tbody>
                {snapshots.map((s) => (
                  <tr key={s.id} className="click">
                    <td className="mono">{s.verdict_key || s.claim_id || `#${s.id}`}</td>
                    <td><span className="tag q">{s.claim_level}</span></td>
                    <td>{s.claim_status}</td>
                    <td><span className="tag plain">{s.replication_status}</span></td>
                    <td className="hint">{s.assessment_level || DASH}{s.assessment_status ? ` · ${s.assessment_status}` : ''}</td>
                    <td className="right-td num">{s.revision}</td>
                    <td className="right-td"><Link className="btn sm" to={`/verdicts/${s.id}`}>Open →</Link></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      ) : isError ? (
        <ErrorState error={error} />
      ) : isLoading ? (
        <Loading label="Loading verdicts…" />
      ) : derived.length === 0 ? (
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
              {derived.map((j) => (
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

// ---- Shared bits ---------------------------------------------------------

function LadderRungs({ steps, claimLevel }) {
  if (!steps.length) return <span className="hint">No ladder rungs reported.</span>
  const lastHit = steps.reduce((acc, s, i) => (s.ok ? i : acc), -1)
  return (
    <div className="ladder">
      {steps.map((s, i) => (
        <span key={s.key || i} className={`rung ${s.ok ? 'hit' : ''} ${i === lastHit ? 'cur' : ''}`}
          title={s.detail || s.caution || s.label}>
          {s.label}
        </span>
      ))}
      {claimLevel ? <span className="rung level">claim: {claimLevel}</span> : null}
    </div>
  )
}

function CheckList({ rows }) {
  if (!rows.length) return <p className="hint" style={{ margin: 0 }}>None recorded.</p>
  return rows.map((f) => (
    <div className="check" key={f.key}>
      <span className={`ok ${f.ok === true ? '' : f.ok === false ? 'bad' : 'plan'}`}>
        {f.ok === true ? '✓' : f.ok === false ? '✕' : '·'}
      </span>
      {f.label}
    </div>
  ))
}

function CaveatList({ notes }) {
  if (!notes.length) return <p className="hint" style={{ margin: 0 }}>No caveats recorded.</p>
  return notes.map((w, i) => (
    <p style={{ margin: '0 0 8px' }} key={w.code || i}>• <b>{w.title || w.code}</b>{w.message ? ` — ${w.message}` : ''}</p>
  ))
}

// ---- Persistent snapshot detail -----------------------------------------

function SnapshotDetail({ data }) {
  const snapshot = data.snapshot
  const history = Array.isArray(data.history) ? data.history : []
  const claim = snapshotClaim(snapshot)
  const scorecard = snapshotScorecardRows(snapshot)
  const metricType = snapshotMetricType(snapshot)
  const ladder = ladderView(snapshot.evidence)
  const fairness = booleanChecks(snapshot.fairness)
  const controls = booleanChecks(snapshot.controls)
  const notes = snapshotCaveats(snapshot)
  const revisions = revisionHistory(history, claim.revision)

  return (
    <>
      <div className="banner">
        <div className="row">
          <span className="tag good">{claim.claimLevel || 'unadjudicated'}</span>
          <span className="tag plain">status: {claim.claimStatus || DASH}</span>
          <span className="tag plain">replication: {claim.replicationStatus || DASH}</span>
          <span className="tag warn">awaiting human review</span>
          <span className="spacer" />
          <button className="btn" type="button">Reject with note</button>
          <button className="btn primary" type="button" disabled title="Claim promotion is human-gated">
            Promote to claim ladder…
          </button>
        </div>
        <h1 className="h1" style={{ marginTop: 12 }}>{claim.verdictKey || claim.claimId || `Verdict #${snapshot.id}`}</h1>
        <p className="sub">
          Canonical claim level, status, and replication come from the checked claim ledger and are kept distinct from the
          dashboard's derived assessment
          {claim.assessmentLevel ? <> (<b>{claim.assessmentLevel}{claim.assessmentStatus ? ` · ${claim.assessmentStatus}` : ''}</b>)</> : ''}.
          No composite advantage score is produced. Revision {claim.revision ?? DASH}
          {history.length ? ` · ${history.length} revision${history.length === 1 ? '' : 's'} on record` : ''}.
        </p>
        <LadderRungs steps={ladder.steps} claimLevel={claim.claimLevel} />
      </div>

      <div className="grid2" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd"><h3>Scorecard</h3><span className="hint">{metricType ? `metric: ${metricType} · ` : ''}per dimension · no total</span></div>
          <div className="bd scroll-x" style={{ padding: '4px 16px 8px' }}>
            <table className="data">
              <thead><tr><th>Dimension</th><th className="right-td">Δ (candidate − control)</th></tr></thead>
              <tbody>
                {scorecard.map((r) => (
                  <tr key={r.key}><td>{r.label}</td><td className="right-td num">{fmtNum(r.delta, 3)}</td></tr>
                ))}
                {scorecard.length === 0 ? (
                  <tr><td colSpan="2" className="hint">No named scorecard dimensions on this snapshot.</td></tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <div className="hd"><h3>Honest caveats</h3></div>
          <div className="bd" style={{ fontSize: 13, color: 'var(--ink2)' }}>
            <CaveatList notes={notes} />
            <p className="hint" style={{ marginTop: 6 }}>Simulator timings only — never a QPU cost claim.</p>
          </div>
        </div>
      </div>

      <div className="grid2" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd"><h3>Fairness</h3></div>
          <div className="bd" style={{ paddingTop: 8 }}><CheckList rows={fairness} /></div>
        </div>
        <div className="card">
          <div className="hd"><h3>Controls</h3></div>
          <div className="bd" style={{ paddingTop: 8 }}><CheckList rows={controls} /></div>
        </div>
      </div>

      {revisions.length > 1 ? (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="hd">
            <h3>Revision history</h3>
            <span className="hint">append-only, content-addressed · newest first · corrections are recorded, never overwritten</span>
          </div>
          <div className="bd" style={{ padding: '4px 16px 12px' }}>
            <RevisionTimeline rows={revisions} />
          </div>
        </div>
      ) : null}
    </>
  )
}

// The claim ledger's history as a vertical timeline. Each entry shows the
// canonical claim_level / status / replication for that revision verbatim, and
// flags the fields that changed from the older revision — so a promotion,
// refutation, or replication change is visible without any snapshot being
// rewritten. The current revision is marked; nothing here derives a verdict.
function RevisionTimeline({ rows }) {
  return (
    <ol className="rev-timeline">
      {rows.map((r) => (
        <li key={r.revision ?? r.contentHash ?? r.createdTs} className={`rev-item${r.isCurrent ? ' current' : ''}`}>
          <span className="rev-dot" aria-hidden="true" />
          <div className="rev-body">
            <div className="rev-head">
              <b>rev {r.revision ?? DASH}</b>
              {r.isCurrent ? <span className="tag good sm">current</span> : null}
              {r.changedAny ? <span className="tag warn sm">changed</span> : <span className="hint">no claim change</span>}
              {r.createdTs ? <span className="hint mono" style={{ marginLeft: 'auto' }}>{r.createdTs}</span> : null}
            </div>
            <div className="rev-fields">
              <span className={r.changed.level ? 'chg' : ''}>level: <b>{r.claimLevel ?? DASH}</b></span>
              <span className={r.changed.status ? 'chg' : ''}>status: <b>{r.claimStatus ?? DASH}</b></span>
              <span className={r.changed.replication ? 'chg' : ''}>replication: <b>{r.replicationStatus ?? DASH}</b></span>
              {r.contentHash ? <span className="hint mono">#{r.contentHash}</span> : null}
            </div>
          </div>
        </li>
      ))}
    </ol>
  )
}

// ---- Comparison fallback (run without a persisted snapshot) --------------

function ScoreDelta({ row }) {
  if (row.favors === 'tie' || row.favors == null) return <span className="delta tie">matched</span>
  const arm = row.favors === 'quantum' ? 'q' : 'c'
  const label = row.favors === 'quantum' ? 'Q' : 'C'
  const pct = row.deltaPct != null ? ` ${fmtPct(row.deltaPct)}` : ''
  return <span className={`delta ${arm}`}>{label}{pct}</span>
}

function ComparisonVerdict({ id }) {
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
          {comparison?.reason || 'This run has no matched comparison or persisted verdict yet — queue its classical twin to adjudicate.'}
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
          Derived on the fly from this pair (no persisted snapshot). Diagnostics are labeled diagnostics; no composite
          advantage score, and a strong diagnostic can never raise the claim level. <b>Promotion is yours alone.</b>
        </p>
        <LadderRungs steps={ladder.steps} claimLevel={ladder.claimLevel} />
      </div>

      <div className="grid32" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd">
            <h3>Perplexity vs steps</h3>
            <Legend items={[{ label: 'candidate', color: ARM.quantum }, { label: 'control', color: ARM.classical }]} />
          </div>
          <div className="bd chart-wrap"><ComparisonCurve rows={pplRows} metricLabel="val_ppl" height={260} /></div>
          <p className="hint" style={{ padding: '0 16px 12px' }}>
            Single seed per arm — a min–max seed band needs multi-seed data (a full study), not this pairwise comparison.
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
          <div className="bd" style={{ paddingTop: 8 }}><CheckList rows={fairness} /></div>
        </div>
        <div className="card">
          <div className="hd"><h3>Honest caveats</h3><span className="hint">auto-generated</span></div>
          <div className="bd" style={{ fontSize: 13, color: 'var(--ink2)' }}>
            <CaveatList notes={notes} />
            <p className="hint" style={{ marginTop: 6 }}>Simulator timings only — never a QPU cost claim.</p>
          </div>
        </div>
      </div>
    </>
  )
}

// ---- Detail dispatch -----------------------------------------------------

function VerdictDetail({ id }) {
  // Prefer a persisted snapshot (list links pass a snapshot id); fall back to a
  // derived comparison (run-detail links pass a job id, or the store is absent).
  const verdict = useVerdict(id)
  if (verdict.isLoading) return <Loading label="Loading verdict…" />
  if (verdict.data?.snapshot) return <SnapshotDetail data={verdict.data} />
  return <ComparisonVerdict id={id} />
}

export default function Verdicts() {
  const { id } = useParams()
  return id ? <VerdictDetail id={id} /> : <VerdictsList />
}
