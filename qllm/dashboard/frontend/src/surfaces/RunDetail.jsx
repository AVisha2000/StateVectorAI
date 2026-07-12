import { useMemo } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api.js'
import { useWorkspace, useModelTests, useDiagnostics } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState, StatusTag } from '../lib/ui.jsx'
import { ComparisonCurve, MetricCurve, TrainabilityChart, Legend, ARM } from '../components/charts.jsx'
import { mergeComparison, mergeCurve } from '../lib/curves.js'
import { diagnosticValues, hasAnyDiagnostic, DIAGNOSTIC_KPIS } from '../lib/diagnostics.js'
import { fmtNum, fmtSci, fmtPct, DASH } from '../lib/format.js'

// Which trainability series to draw, best-first — grad_norm_ratio is the
// per-step circuit-vs-classical gradient signal already in the curve.
const TRAINABILITY_KEYS = ['grad_norm_ratio', 'grad_norm_circuit', 'grad_norm_classical']

function kpiValue(entry, kind) {
  if (!entry || !entry.available) return { text: DASH, title: entry?.reason || 'unavailable' }
  return { text: kind === 'sci' ? fmtSci(entry.value) : fmtNum(entry.value, 3), title: null }
}

export default function RunDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data, isLoading, isError, error } = useWorkspace(id)
  const { data: tests } = useModelTests(id)
  const { data: diag } = useDiagnostics(id)

  const cancel = useMutation({
    mutationFn: () => api.cancelJob(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workspace', id] })
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })

  const job = data?.job
  const comparison = data?.comparison
  const curve = data?.curve || {}
  const finalRun = data?.final_run || {}

  const pplRows = useMemo(() => {
    if (comparison?.available) {
      return mergeComparison(comparison.candidate?.curve, comparison.baseline?.curve, 'val_ppl')
    }
    return mergeCurve(curve, ['val_ppl']).map((r) => ({ step: r.step, candidate: r.val_ppl, baseline: null }))
  }, [comparison, curve])

  const trainKey = useMemo(() => TRAINABILITY_KEYS.find((k) => Array.isArray(curve[k]) && curve[k].length), [curve])
  const trainRows = useMemo(() => (trainKey ? mergeCurve(curve, [trainKey]) : []), [curve, trainKey])

  const summaryDiag = tests?.summary?.quantum_diagnostics
  const diagValues = useMemo(() => diagnosticValues(diag?.diagnostics, summaryDiag), [diag, summaryDiag])

  // Merge run warnings with the diagnostics endpoint's own non-advantage notes,
  // de-duplicated by code. Hooks must run before any early return below.
  const warnings = useMemo(() => {
    const all = [...(data?.interpretation_warnings || []), ...(diag?.interpretation_warnings || [])]
    const seen = new Set()
    return all.filter((w) => {
      const key = w?.code || w?.title
      if (!key || seen.has(key)) return false
      seen.add(key)
      return true
    })
  }, [data, diag])

  if (isError) return <ErrorState error={error} label="Could not load this run." />
  if (isLoading) return <Loading label="Loading run…" />
  if (!job) return <ErrorState error={{ message: `Run ${id} not found.` }} label="Run not found." />

  const isQuantum = job.uses_quantum
  const currentStep = pplRows.length ? pplRows[pplRows.length - 1].step : null
  const totalSteps = job.steps
  const live = job.status === 'running' || job.status === 'queued'

  // vs-twin headline from the paired final metrics.
  const q = comparison?.candidate?.final_run?.val_ppl
  const c = comparison?.baseline?.final_run?.val_ppl
  const vsTwin = typeof q === 'number' && typeof c === 'number' && c !== 0 ? (q - c) / Math.abs(c) : null

  return (
    <>
      <PageHeader
        title={<span className="mono" style={{ fontSize: 17 }}>#{job.id} {job.run_name}</span>}
        actions={
          <>
            {live ? (
              <button className="btn" type="button" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
                {cancel.isPending ? 'Cancelling…' : 'Cancel'}
              </button>
            ) : null}
            <Link className="btn primary" to={`/verdicts/${job.id}`}>Compare with twin →</Link>
          </>
        }
      />

      <div className="row" style={{ gap: 6, marginTop: 10 }}>
        <span className={`tag ${isQuantum ? 'q' : 'c'}`}>{isQuantum ? 'QUANTUM' : 'CLASSICAL'}</span>
        {job.dataset_name ? <span className="tag plain">{job.dataset_name}</span> : null}
        <span className="tag plain">seed {job.seed}</span>
        {job.preset_id ? <span className="tag plain">{job.preset_id}</span> : null}
        {job.model_family ? <span className="tag plain">{job.model_family}</span> : null}
        {job.device_target ? <span className="tag plain">{job.device_target}</span> : null}
        <StatusTag status={job.status} />
        {currentStep != null && totalSteps ? (
          <span className="tag plain num">step {fmtNum(currentStep, 0)} / {fmtNum(totalSteps, 0)}</span>
        ) : null}
      </div>

      <div className="kpis" style={{ marginTop: 16 }}>
        <div className="kpi">
          <span className="microlabel">val_ppl</span>
          <div className="v num">{fmtNum(finalRun.val_ppl ?? q, 2)}</div>
          <div className="s">final validation perplexity</div>
        </div>
        <div className="kpi">
          <span className="microlabel">vs twin</span>
          <div className="v num" style={{ color: vsTwin != null ? (vsTwin < 0 ? 'var(--good)' : 'var(--crit)') : 'var(--ink)' }}>
            {vsTwin != null ? fmtPct(vsTwin) : DASH}
          </div>
          <div className="s">{vsTwin != null ? 'final val_ppl vs control' : 'no matched control yet'}</div>
        </div>
        {DIAGNOSTIC_KPIS.map((k) => {
          const cell = kpiValue(diagValues[k.key], k.kind)
          return (
            <div className="kpi" key={k.key} title={cell.title || undefined}>
              <span className="microlabel">{k.label}</span>
              <div className="v num">{cell.text}</div>
              <div className="s">{k.hint}</div>
            </div>
          )
        })}
      </div>

      <div className="grid2" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd">
            <h3>Validation perplexity</h3>
            <Legend items={[
              { label: 'candidate', color: ARM.quantum },
              { label: 'control', color: ARM.classical },
            ]} />
          </div>
          <div className="bd chart-wrap">
            <ComparisonCurve rows={pplRows} metricLabel="val_ppl" />
          </div>
        </div>
        <div className="card">
          <div className="hd">
            <h3>Gradient trace <span className="hint" style={{ display: 'inline' }}>— barren-plateau watch</span></h3>
          </div>
          <div className="bd chart-wrap">
            <TrainabilityChart rows={trainRows} metricKey={trainKey || 'grad_norm_ratio'} label={trainKey || 'grad norm ratio'} />
          </div>
        </div>
      </div>

      {!hasAnyDiagnostic(diagValues) ? (
        <p className="hint" style={{ marginTop: 10 }}>
          Per-run diagnostics (expressibility, gradient variance/SNR, entanglement) come from{' '}
          <span className="mono">quantum/metrics.py</span>; they appear here once a quantum run finishes and writes its
          <span className="mono"> summary.json</span>, or when the backend ships <span className="mono">/jobs/{'{id}'}/diagnostics</span>.
        </p>
      ) : (
        <p className="hint" style={{ marginTop: 10 }}>
          Values above are <b>diagnostics / mechanism candidates</b> from <span className="mono">quantum/metrics.py</span> —
          never advantage on their own.
        </p>
      )}

      {warnings.length ? (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="hd"><h3>Interpretation warnings</h3><span className="hint">from the backend</span></div>
          <div className="bd">
            {warnings.map((w, i) => (
              <div className="check" key={w.code || i}>
                <span className={`ok ${w.severity === 'error' ? 'bad' : 'warn'}`}>!</span>
                <span><b>{w.title}</b> — {w.message}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </>
  )
}
