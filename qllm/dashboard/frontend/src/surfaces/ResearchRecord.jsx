import { Link, useParams } from 'react-router-dom'
import { isNotYetBuilt, useClaimProposal, useEvidence, useLineage, useReview, useWorkDetail, useWorkOverlaps } from '../lib/hooks.js'
import { ErrorState, Loading, PageHeader } from '../lib/ui.jsx'

function ValueList({ values }) {
  return Array.isArray(values) && values.length ? <ul>{values.map((value, index) => <li key={`${index}:${value}`}>{value}</li>)}</ul> : 'Not recorded'
}

function DetailList({ entries }) {
  return <dl className="research-meta">{entries.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value ?? 'Not recorded'}</dd></div>)}</dl>
}

function flag(value) {
  return typeof value === 'boolean' ? String(value) : null
}

function ObjectCard({ node }) {
  const evidence = useEvidence(node.type === 'submission' ? node.id : '')
  const claim = useClaimProposal(node.type === 'claim' ? node.id : '')
  const review = useReview(node.type === 'review' ? node.id : '')
  const record = evidence.data || claim.data || review.data
  const loading = evidence.isLoading || claim.isLoading || review.isLoading
  const error = evidence.isError || claim.isError || review.isError
  const status = record?.outcome || record?.status
  const detail = !record ? null : node.type === 'submission' ? <>
    <p>{record.summary}</p>
    <DetailList entries={[
      ['Submission ID', record.id], ['Revision', record.revision], ['Content hash', <span className="mono id-wrap">{record.content_hash}</span>],
      ['Outcome', record.outcome], ['Authority', record.authority], ['Scientific acceptance', flag(record.scientific_acceptance)],
      ['Methods', record.methods], ['Controls', record.controls], ['Regime', record.regime], ['Uncertainty', record.uncertainty],
      ['Limitations', <ValueList values={record.limitations} />], ['Gaps', <ValueList values={record.gaps} />],
      ['Previous submission hash', record.predecessor_content_hash ? <span className="mono id-wrap">{record.predecessor_content_hash}</span> : 'None'],
      ['Correction reason', record.correction_reason], ['Submitted by', record.submitted_by_actor_id], ['Work revision', record.work_revision],
    ]} />
    <h3>Evidence items</h3>
    {record.items?.length ? <ul className="evidence-items">{record.items.map((item, index) => <li data-evidence-kind={item.kind} key={`${item.kind}:${item.anchor_type}:${item.anchor_id}:${index}`}><b>{item.kind}</b>: {item.statement} <span className="hint">anchor {item.anchor_type}: </span><span className="mono id-wrap">{item.anchor_id}</span></li>)}</ul> : <p className="hint">No evidence items recorded.</p>}
  </> : node.type === 'claim' ? <>
    <p>{record.statement}</p>
    <DetailList entries={[
      ['Claim ID', record.id], ['Revision', record.revision], ['Claim hash', <span className="mono id-wrap">{record.content_hash}</span>],
      ['Submission ID', record.submission_id], ['Submission revision', record.submission_revision], ['Submission hash', <span className="mono id-wrap">{record.submission_hash}</span>],
      ['Authority', record.authority], ['Scientific acceptance', flag(record.scientific_acceptance)], ['Canonical claim updated', flag(record.canonical_claim_updated)],
      ['Scope', record.scope], ['Comparator', record.comparator], ['Regime', record.regime], ['Evidence level', record.intended_evidence_level],
      ['Uncertainty', record.uncertainty], ['Limitations', <ValueList values={record.limitations} />], ['Correction', record.correction_reason],
      ['Previous claim hash', record.predecessor_claim_content_hash ? <span className="mono id-wrap">{record.predecessor_claim_content_hash}</span> : 'None'],
      ['Proposed by', record.proposed_by_actor_id], ['Work revision', record.work_revision],
    ]} />
  </> : node.type === 'review' ? <>
    <p>{record.rationale}</p>
    <DetailList entries={[
      ['Review ID', record.id], ['Revision', record.revision], ['Review hash', <span className="mono id-wrap">{record.content_hash}</span>],
      ['Submission ID', record.submission_id], ['Submission hash', <span className="mono id-wrap">{record.submission_hash}</span>],
      ['Claim ID', record.claim_id], ['Claim hash', <span className="mono id-wrap">{record.claim_hash}</span>],
      ['Authority', record.authority], ['Scientific acceptance', flag(record.scientific_acceptance)], ['Canonical claim updated', flag(record.canonical_claim_updated)],
      ['Outcome', record.outcome], ['Checks', <ValueList values={record.checks} />], ['Limitations', <ValueList values={record.limitations} />], ['Reviewer', record.reviewed_by_actor_id],
      ['Work revision', record.work_revision],
    ]} />
  </> : null
  return <li>
    <div className="row"><span className="tag plain">{node.type}</span>{status ? <span className="tag plain">{status}</span> : null}<span className="mono id-wrap">{node.id}</span></div>
    <div className="hint">revision {node.revision} · hash <span className="mono id-wrap">{node.hash}</span></div>
    {loading ? <span className="hint">Loading object…</span> : error ? <span className="hint">Object detail is unavailable.</span> : detail}
    {node.type === 'decision' ? <Link to={`/decisions/${encodeURIComponent(node.id)}`}>Open decision record</Link> : null}
  </li>
}

function RelationshipList({ edges }) {
  return edges.length ? <ul className="relationship-list">{edges.map((edge) => <li key={`${edge.from}:${edge.relation}:${edge.to}`}>
    <span className="mono id-wrap">{edge.from}</span>
    <span className="tag plain">{edge.relation.replaceAll('_', ' ')}</span>
    <span className="mono id-wrap">{edge.to}</span>
  </li>)}</ul> : <p className="hint">No lineage relationships have been recorded.</p>
}

function Timeline({ events = [] }) {
  return <ol className="rev-timeline">{events.map((event) => <li className="rev-item" key={event.id}><span className="rev-dot" /><div className="rev-head"><b>{event.event_type.replaceAll('_', ' ')}</b><span className="hint">revision {event.revision}</span></div><div className="hint">{event.actor_id || 'system'}</div></li>)}</ol>
}

export default function ResearchRecord() {
  const { workId } = useParams()
  const work = useWorkDetail(workId)
  const overlaps = useWorkOverlaps(workId)
  const lineage = useLineage(workId)
  if (work.isLoading) return <Loading label="Loading research record…" />
  if (work.isError) return isNotYetBuilt(work.error) ? <Missing /> : <ErrorState error={work.error} label="Could not load this research record." />
  const record = work.data
  const signature = record.payload?.research_signature
  const nodes = lineage.data?.nodes || []
  const edges = lineage.data?.edges || []
  return <>
    <PageHeader title={record.payload?.title || 'Research record'} sub="Coordination record for a bounded research task. All evidence, claims, and reviews shown here are provisional and are not scientific acceptance." actions={<Link className="btn sm" to="/">Workboard</Link>} />
    <div className="notice research-authority" role="note"><b>Provisional coordination record.</b> This view does not update a canonical claim or represent scientific acceptance.</div>
    <div className="research-grid">
      <section className="card"><div className="hd"><h2>Scope and state</h2><span className="tag plain">{record.state}</span></div><div className="bd detail-copy"><p>{record.payload?.question}</p><dl className="research-meta"><div><dt>Scope</dt><dd>{record.payload?.scope}</dd></div><div><dt>Stop rule</dt><dd>{record.payload?.stop_rule}</dd></div><div><dt>Work ID</dt><dd className="mono id-wrap">{record.id}</dd></div><div><dt>Human input</dt><dd>{record.blocking_decision_ids?.length ? record.blocking_decision_ids.map((id) => <div key={id}><Link to={`/decisions/${encodeURIComponent(id)}`}>{id}</Link></div>) : 'No open decision is attached.'}</dd></div></dl>{signature ? <details><summary>Research signature</summary><dl className="research-meta">{Object.entries(signature).map(([name, value]) => <div key={name}><dt>{name.replaceAll('_', ' ')}</dt><dd>{value}</dd></div>)}</dl></details> : null}</div></section>
      <section className="card"><div className="hd"><h2>Activity</h2></div><div className="bd">{record.events?.length ? <Timeline events={record.events} /> : <div className="state">No recorded activity.</div>}</div></section>
    </div>
    <section className="card research-overlap"><div className="hd"><h2>Overlap explanations</h2></div><div className="bd">{overlaps.isLoading ? <Loading label="Loading overlaps…" /> : overlaps.isError ? <p className="hint">Overlap explanations are unavailable.</p> : overlaps.data?.candidates?.length ? <ul className="object-list">{overlaps.data.candidates.map((item) => <li key={item.id}><b>{item.title}</b><div className="hint">{item.reasons?.join(' · ') || 'No explanation recorded'}{item.classification ? ` · ${item.classification}` : ''}</div></li>)}</ul> : <p className="hint">No overlap candidates recorded.</p>}</div></section>
    <section className="card lineage-card"><div className="hd"><h2>Complete lineage</h2></div><div className="bd">{lineage.isLoading ? <Loading label="Loading lineage…" /> : lineage.isError ? <p className="hint">Lineage is unavailable.</p> : <><DetailList entries={[["Authority", lineage.data?.authority], ["Scientific acceptance", flag(lineage.data?.scientific_acceptance)], ["Canonical claim updated", flag(lineage.data?.canonical_claim_updated)]]} /><h3 className="lineage-subhead">Objects</h3>{nodes.length ? <ul className="object-list">{nodes.map((node) => <ObjectCard key={`${node.type}:${node.id}`} node={node} />)}</ul> : <p className="hint">No provisional objects have been recorded.</p>}<h3 className="lineage-subhead">Relationships</h3><RelationshipList edges={edges} /></>}</div></section>
  </>
}

function Missing() {
  return <div className="state err" role="alert">This research record was not found or is unavailable on this backend. <Link to="/">Return to the Workboard</Link>.</div>
}
