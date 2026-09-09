import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { isNotYetBuilt, useDecision, useResolveDecision } from '../lib/hooks.js'
import { ErrorState, Loading, PageHeader } from '../lib/ui.jsx'

function key() {
  return globalThis.crypto?.randomUUID?.() || `decision-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export default function DecisionDetail() {
  const { id } = useParams()
  const query = useDecision(id)
  const mutation = useResolveDecision(id)
  const idempotencyKey = useMemo(key, [id, query.data?.revision])
  const [answer, setAnswer] = useState('')
  const [rationale, setRationale] = useState('')
  const errorRef = useRef(null)
  const decision = query.data
  const playground = import.meta.env.VITE_PLAYGROUND_MODE === '1'

  useEffect(() => {
    if (mutation.isError) errorRef.current?.focus()
  }, [mutation.isError])

  if (query.isLoading) return <Loading label="Loading decision…" />
  if (query.isError) return isNotYetBuilt(query.error)
    ? <Missing /> : <ErrorState error={query.error} label="Could not load this decision." />
  if (!decision) return <Missing />

  const canResolve = decision.status === 'open' && decision.allowed_actions?.includes('resolve')
  const isStale = mutation.isError && mutation.error?.status === 409
  const refreshDecision = async () => {
    mutation.reset()
    await query.refetch()
  }
  const submit = (event) => {
    event.preventDefault()
    if (!answer || !rationale.trim()) return
    mutation.mutate({ expected_revision: decision.revision, idempotency_key: idempotencyKey, answer, rationale: rationale.trim() })
  }

  return <>
    <PageHeader title="Human decision" sub="Provide research taste for this bounded coordination question. This response does not make or promote a scientific claim." actions={<Link className="btn sm" to="/decisions">All decisions</Link>} />
    <section className="card decision-detail" aria-labelledby="decision-question">
      <div className="hd"><span className={`tag ${decision.status === 'open' ? 'warn' : 'plain'}`}>{decision.status}</span><span className="mono id-wrap">{decision.id}</span></div>
      <div className="bd">
        <h2 id="decision-question" className="detail-title">{decision.question}</h2>
        <dl className="decision-context">
          <div><dt>Human authority boundary</dt><dd>{decision.required_human_capability || decision.request_payload?.required_human_capability || 'Not recorded'}</dd></div>
          <div><dt>Agent recommendation</dt><dd>{decision.request_payload?.recommendation || 'Not recorded'}</dd></div>
          <div><dt>Supporting evidence</dt><dd>{decision.request_payload?.supporting_evidence_ids?.length ? <ul>{decision.request_payload.supporting_evidence_ids.map((id) => <li key={id} className="mono id-wrap">{id}</li>)}</ul> : 'Not recorded'}</dd></div>
          <div><dt>Uncertainty</dt><dd>{decision.request_payload?.uncertainty || 'Not recorded'}</dd></div>
          <div><dt>Consequences</dt><dd>{decision.request_payload?.consequences?.length ? <ul>{decision.request_payload.consequences.map((item) => <li key={item}>{item}</li>)}</ul> : 'Not recorded'}</dd></div>
          <div><dt>Expires at</dt><dd>{decision.request_payload?.expires_at || 'No expiry recorded'}</dd></div>
          <div><dt>Reversible</dt><dd>{decision.request_payload?.reversible === true ? 'Yes' : decision.request_payload?.reversible === false ? 'No' : 'Not recorded'}</dd></div>
        </dl>
        {mutation.isSuccess ? <div className="notice" role="status">{playground ? 'Your guidance was simulated in this playground and was not persisted.' : 'Your guidance was recorded for this coordination decision.'}</div> : null}
        {mutation.isError ? <div ref={errorRef} className="notice crit" role="alert" tabIndex="-1">Could not record your guidance: {mutation.error.message}{isStale ? <button className="btn sm" type="button" onClick={refreshDecision}>Refresh decision</button> : null}</div> : null}
        {canResolve ? <form className="decision-form" onSubmit={submit}>
          <fieldset disabled={mutation.isPending}>
            <legend>Your choice</legend>
            {decision.options.map((option) => <label className="choice" key={option}>
              <input type="radio" name="answer" value={option} checked={answer === option} onChange={(event) => setAnswer(event.target.value)} required />
              <span>{option}</span>
            </label>)}
          </fieldset>
          <label htmlFor="decision-rationale">Reasoning</label>
          <textarea id="decision-rationale" className="mini block" value={rationale} onChange={(event) => setRationale(event.target.value)} disabled={mutation.isPending} required maxLength="4000" aria-describedby="decision-rationale-help" />
          <p id="decision-rationale-help" className="hint">Explain the research tradeoff behind your choice. This guides coordination only.</p>
          <div className="footer-bar"><button className="btn primary" type="submit" disabled={mutation.isPending || !answer || !rationale.trim()}>{mutation.isPending ? 'Recording…' : 'Record guidance'}</button><Link className="btn ghost" to="/decisions">Decide later</Link><Link className="btn ghost" to={`/research/${encodeURIComponent(decision.work_id)}`}>View research record</Link></div>
        </form> : <div className="notice">{decision.status === 'resolved' ? <>{playground ? 'Simulated answer:' : 'Recorded answer:'} <b>{decision.answer}</b>{decision.rationale ? ` — ${decision.rationale}` : ''}</> : 'This decision is not currently available for resolution.'}</div>}
      </div>
    </section>
  </>
}

function Missing() {
  return <div className="state err" role="alert">This decision was not found or is unavailable on this backend. <Link to="/decisions">Return to the inbox</Link>.</div>
}
