import { Link } from 'react-router-dom'
import { useDecisions, useWork } from '../lib/hooks.js'
import { Loading, PageHeader } from '../lib/ui.jsx'

function countWork(work, state) {
  return work.filter((item) => item.state === state).length
}

function countOpen(decisions) {
  return decisions.filter((item) => item.status === 'open').length
}

function CoordinationState({ workError, decisionError }) {
  const disabled = workError?.status === 503 || decisionError?.status === 503
  if (disabled) {
    return (
      <div className="state" role="status">
        Agent coordination is disabled on this backend. The existing <Link to="/lab">Lab overview</Link> remains available.
      </div>
    )
  }
  const detail = workError?.message || decisionError?.message
  return (
    <div className="state err" role="alert">
      Could not load the Workboard.{detail ? <div className="hint" style={{ marginTop: 6 }}>{detail}</div> : null}
    </div>
  )
}

function StateTag({ state }) {
  const className = state === 'running' ? 'good' : state === 'claimed' ? 'warn' : 'plain'
  return <span className={`tag ${className}`}>{state}</span>
}

function Blockers({ item }) {
  const overlaps = item.blocking_overlap_ids || []
  const decisions = item.blocking_decision_ids || []
  if (!overlaps.length && !decisions.length) return <>None</>
  return <>
    {overlaps.length ? <div><span className="hint">Overlap: </span>{overlaps.map((id) => <span className="mono id-wrap" key={id}>{id} </span>)}</div> : null}
    {decisions.length ? <div><span className="hint">Decision: </span>{decisions.map((id) => <Link className="mono id-wrap" key={id} to={`/decisions/${encodeURIComponent(id)}`}>{id}</Link>)}</div> : null}
  </>
}

function NextState({ item }) {
  const status = []
  if (item.blocking_overlap_ids?.length) status.push('Blocked: resolve overlap warning')
  if (item.blocking_decision_ids?.length) status.push('Blocked: human decision due')
  if (item.state === 'ready_for_human') status.push('Evidence reviewed for human handoff')
  if (item.state === 'revision_requested') status.push('Revision requested')
  if (status.length) return <>{status.join(' · ')}</>
  return item.actionable_next_state ? <>Actionable next: {item.actionable_next_state}</> : <>No action currently available</>
}

export default function Workboard() {
  const workQuery = useWork()
  const decisionsQuery = useDecisions()
  const work = Array.isArray(workQuery.data) ? workQuery.data : []
  const decisions = Array.isArray(decisionsQuery.data) ? decisionsQuery.data : []
  const openDecisions = decisions.filter((item) => item.status === 'open')
  const loading = workQuery.isLoading || decisionsQuery.isLoading
  const error = workQuery.isError || decisionsQuery.isError
  const simulated = import.meta.env.VITE_PLAYGROUND_MODE === '1'

  return (
    <>
      <PageHeader title="Workboard" sub="Read-only coordination for quantum research. Work is visible here; agents and authorized humans act through the API boundary." />
      {simulated ? <p className="notice workboard-notice">UAT playground data is synthetic and non-persistent.</p> : null}
      {loading ? <Loading label="Loading agent work and human decisions…" /> : error ? <CoordinationState workError={workQuery.error} decisionError={decisionsQuery.error} /> : (
        <>
          <div className="workboard-counts" aria-label="Workboard counts">
            <Count label="Available" value={countWork(work, 'available')} description="ready to claim" />
            <Count label="Claimed" value={countWork(work, 'claimed')} description="held by an agent" />
            <Count label="Running" value={countWork(work, 'running')} description="active research" />
            <Count label="Open decisions" value={countOpen(decisions)} description="need human input" />
          </div>
          {work.length === 0 && decisions.length === 0 ? (
            <div className="state" role="status">No agent work or human decisions are open yet.</div>
          ) : (
            <div className="workboard-grid">
              <section className="card" aria-labelledby="work-queue-title">
                <div className="hd"><h2 id="work-queue-title">Research queue</h2><span className="hint">read-only</span></div>
                <div className="scroll-x">
                  {work.length ? <table className="data">
                    <thead><tr><th>Question</th><th>State</th><th>Owner / agent</th><th>Blockers</th><th>Next state</th><th>Scope</th><th>Stop rule</th></tr></thead>
                    <tbody>{work.map((item) => <tr key={item.id}>
                      <td><Link to={`/research/${encodeURIComponent(item.id)}`}><b>{item.payload?.title || item.id}</b></Link><div className="hint">{item.payload?.question}</div></td>
                      <td><StateTag state={item.state} /></td>
                      <td>{item.owner_actor_id || 'Unassigned'}</td>
                      <td><Blockers item={item} /></td>
                      <td><NextState item={item} /></td>
                      <td>{item.payload?.scope || '—'}</td>
                      <td className="hint">{item.payload?.stop_rule || '—'}</td>
                    </tr>)}</tbody>
                  </table> : <div className="state">No work packages are available.</div>}
                </div>
              </section>
              <section className="card" aria-labelledby="decision-list-title">
                <div className="hd"><h2 id="decision-list-title">Human decisions</h2><span className="hint">input, not a command</span></div>
                <div className="bd">
                  {openDecisions.length ? <ul className="decision-list">{openDecisions.map((item) => (
                    <li key={item.id}>
                      <div className="row"><StateTag state="open" /><span className="microlabel">human review</span></div>
                      <Link to={`/decisions/${encodeURIComponent(item.id)}`}><b>{item.question}</b></Link>
                      <p className="hint">Agent recommendation: {item.request_payload?.recommendation || 'not recorded'}. Options: {(item.options || []).join(' · ') || 'not recorded'}.</p>
                    </li>
                  ))}</ul> : <div className="state">No open human decisions.</div>}
                </div>
              </section>
            </div>
          )}
          <section className="notice workboard-explainer" aria-label="Agent interface">
            <b>Agent interface</b><br />Agents read the shared queue, claim bounded work, and request explicit human judgment when uncertainty needs research taste. This page never submits a claim, decision, or experiment.
          </section>
        </>
      )}
    </>
  )
}

function Count({ label, value, description }) {
  return <div className="workboard-count"><div className="microlabel">{label}</div><strong className="num">{value}</strong><span>{description}</span></div>
}
