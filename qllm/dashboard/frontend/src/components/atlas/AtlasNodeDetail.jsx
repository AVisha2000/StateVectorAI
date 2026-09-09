import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../api.js'
import { OUTCOME_LABELS } from '../../lib/atlasModel.js'

function atlasProposal(cell) {
  const target = (cell.advantage_target || 'the intended outcome').replaceAll('_', ' ')
  return `Evaluate whether ${cell.label} changes ${target} under matched controls.`
}

// Side panel for a selected cell. Claim level and replication status are shown
// as SEPARATE labeled rows and never combined; a matched verdict's canonical
// claim (a different vocabulary) is surfaced distinctly. Null / classical-holds
// cells read with the same weight as positive ones.
export default function AtlasNodeDetail({ cell }) {
  const isTrackB = cell?.area_id === 'monitored_quantum_memory'
  const trackBQuery = useQuery({
    queryKey: ['atlas-track-b-readiness'],
    queryFn: api.trackBReadiness,
    enabled: isTrackB,
    retry: false,
  })
  if (!cell) {
    return (
      <div className="card">
        <div className="hd"><h3>Select a cell</h3></div>
        <div className="bd">
          <p className="hint" style={{ margin: 0 }}>
            Pick any cell to see its head-to-head status, claim level, replication, and the latest verdict — or a
            "no advantage found" result, shown with equal prominence.
          </p>
        </div>
      </div>
    )
  }
  const seed = cell.provenance === 'seed'
  return (
    <div className="card">
      <div className="hd">
        <span className={`atlas-oc atlas-oc-${cell.outcome_class}`}>{OUTCOME_LABELS[cell.outcome_class]}</span>
        <span className="spacer" />
        <span className={`tag ${seed ? 'plain' : 'good'}`}>{seed ? 'seed · unverified' : 'derived verdict'}</span>
      </div>
      <div className="bd">
        <div style={{ fontWeight: 600, marginBottom: 8 }}>{cell.label}</div>
        <div className="atlas-kv">
          <div><span className="microlabel">Claim level (map)</span><div>{cell.claim_level || '—'}</div></div>
          <div><span className="microlabel">Replication</span><div>{cell.replication_status || '—'}</div></div>
          <div><span className="microlabel">Status</span><div>{cell.status || '—'}</div></div>
          <div><span className="microlabel">Pipeline stage</span><div>{cell.pipeline_stage || '—'}</div></div>
          <div><span className="microlabel">Quantum resource</span><div>{cell.quantum_resource || '—'}</div></div>
          <div><span className="microlabel">Advantage target</span><div>{cell.advantage_target || '—'}</div></div>
        </div>

        {cell.note ? (
          <div className="notice" style={{ marginTop: 12 }}>
            <span className="microlabel">Current evidence summary</span>
            <div style={{ marginTop: 4 }}>{cell.note}</div>
          </div>
        ) : null}

        {cell.blockers?.length ? (
          <div className="notice warn" style={{ marginTop: 12 }}>
            <span className="microlabel">Current blockers</span>
            <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
              {cell.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}
            </ul>
          </div>
        ) : (
          <div className="notice" style={{ marginTop: 12 }}>
            <span className="microlabel">Current blockers</span>
            <div style={{ marginTop: 4 }}>None recorded in the canonical research map.</div>
          </div>
        )}

        {cell.next_decisive_test ? (
          <div className="notice" style={{ marginTop: 12 }}>
            <span className="microlabel">Next decisive test</span>
            <div style={{ marginTop: 4 }}>{cell.next_decisive_test}</div>
          </div>
        ) : null}

        {cell.verdict_claim_level || cell.verdict_claim_status ? (
          <div className="notice" style={{ marginTop: 12 }}>
            <span className="microlabel">Latest verdict (ledger)</span>
            <div style={{ marginTop: 4 }}>
              claim level <b>{cell.verdict_claim_level || '—'}</b>
              {cell.verdict_claim_status ? <> · status <b>{cell.verdict_claim_status}</b></> : null}
              {' '}— canonical vocabulary, kept distinct from the map's claim ladder above.
            </div>
          </div>
        ) : null}

        {isTrackB ? (
          <div className="notice warn" style={{ marginTop: 12 }}>
            <span className="microlabel">Track B proposal readiness</span>
            {trackBQuery.isLoading ? <div style={{ marginTop: 4 }}>Loading the manifest-bound readiness projection…</div> : null}
            {trackBQuery.isError ? <div style={{ marginTop: 4 }}>Could not load proposal readiness: {trackBQuery.error?.message || 'unknown error'}</div> : null}
            {trackBQuery.data ? <TrackBReadiness data={trackBQuery.data} /> : null}
          </div>
        ) : null}

        <div className="row" style={{ marginTop: 12, gap: 8 }}>
          {cell.verdict_id != null ? (
            <Link className="btn sm" to={`/verdicts/${cell.verdict_id}`}>Open verdict →</Link>
          ) : null}
          {!isTrackB && (cell.outcome_class === 'unexplored' || cell.outcome_class === 'suggested') ? (
            <Link
              className="btn sm primary"
              to={`/bench?source=atlas&area_id=${encodeURIComponent(cell.area_id)}&hypothesis=${encodeURIComponent(atlasProposal(cell))}`}
            >
              Design a test →
            </Link>
          ) : null}
        </div>

        <p className="hint" style={{ marginTop: 10 }}>
          Area <span className="mono">{cell.area_id}</span>. Colors reflect the curated RESEARCH_MAP status; no composite
          advantage score is computed. Simulator results only.
        </p>
      </div>
    </div>
  )
}

function TrackBReadiness({ data }) {
  const implemented = data.implemented_challenger_ids || []
  const missing = data.missing_challenger_ids || []
  const campaign = data.campaign || {}
  const contract = data.resource_contract || {}
  const candidateContract = data.candidate_contract || {}
  return (
    <div style={{ marginTop: 4 }}>
      <div><b>Proposal only.</b> No runnable Bench action is available for this blocked area.</div>
      <div className="hint" style={{ marginTop: 6 }}>
        {campaign.planned_instances ?? '—'} planned instances · {campaign.trajectory_count_per_instance ?? '—'} trajectories/instance · execution {campaign.execution_status || '—'}
      </div>
      <div className="hint" style={{ marginTop: 6 }}>
        Implemented bindings: {implemented.length ? implemented.join(', ') : 'none'}.
      </div>
      <div className="hint" style={{ marginTop: 4 }}>
        Missing bindings: {missing.length ? missing.join(', ') : 'none'}.
      </div>
      <div className="hint" style={{ marginTop: 4 }}>
        Resource contract: {contract.status || 'unavailable'}; {contract.profiles_recorded ?? 0} profiles recorded.
      </div>
      <div className="hint" style={{ marginTop: 4 }}>
        Candidate contract: {candidateContract.status || 'unavailable'}; {candidateContract.candidates_recorded ?? 0} candidates recorded; capacity ladder must be supplied.
      </div>
      <div className="hint" style={{ marginTop: 4 }}>
        Authority: execution, oracle, quantum, material execution, test access, and claim promotion are all closed.
      </div>
    </div>
  )
}
