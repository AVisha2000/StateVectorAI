import { Link } from 'react-router-dom'
import { useDecisions } from '../lib/hooks.js'
import { ErrorState, Loading, PageHeader } from '../lib/ui.jsx'

export default function DecisionInbox() {
  const query = useDecisions()
  const decisions = Array.isArray(query.data) ? query.data : []
  const open = decisions.filter((item) => item.status === 'open')
  const completed = decisions.filter((item) => item.status !== 'open')

  return <>
    <PageHeader title="Decision inbox" sub="Questions where agent work needs human research judgment. A response guides coordination; it is not scientific acceptance." />
    {query.isLoading ? <Loading label="Loading decisions…" /> : query.isError ? <ErrorState error={query.error} label="Could not load decisions." /> : decisions.length === 0 ? (
      <div className="state" role="status">No decisions have been requested.</div>
    ) : <>
      <DecisionSection id="open-decisions" title="Open first" count={`${open.length} awaiting review`} decisions={open} empty="No open decisions need input." />
      {completed.length ? <DecisionSection id="completed-decisions" title="Decision history" count={`${completed.length} recorded`} decisions={completed} /> : null}
    </>}
  </>
}

function DecisionSection({ id, title, count, decisions, empty }) {
  return <section className="card decision-inbox" aria-labelledby={id}>
    <div className="hd"><h2 id={id}>{title}</h2><span className="hint">{count}</span></div>
    {decisions.length ? <ul className="decision-list bd">
      {decisions.map((item) => <li key={item.id}>
        <div className="row"><span className={`tag ${item.status === 'open' ? 'warn' : 'plain'}`}>{item.status}</span><span className="mono id-wrap">{item.id}</span></div>
        <Link className="decision-link" to={`/decisions/${encodeURIComponent(item.id)}`}>{item.question}</Link>
        <p className="hint">Recommendation: {item.request_payload?.recommendation || 'not recorded'} · uncertainty: {item.request_payload?.uncertainty || 'not recorded'}</p>
        <Link className="btn sm" to={`/decisions/${encodeURIComponent(item.id)}`}>{item.status === 'open' ? 'Review question' : 'View record'}</Link>
      </li>)}
    </ul> : <div className="state" role="status">{empty}</div>}
  </section>
}
