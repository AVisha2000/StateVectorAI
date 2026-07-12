import { useMemo } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useJobs, useVerdicts, isNotYetBuilt } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState, StatusTag } from '../lib/ui.jsx'

// Recent snapshots first (higher id = newer append-only revision).
function latestVerdicts(data, n = 5) {
  const snaps = Array.isArray(data?.snapshots) ? data.snapshots : []
  return [...snaps].sort((a, b) => (b.id ?? 0) - (a.id ?? 0)).slice(0, n)
}

function LatestVerdicts() {
  const { data, isLoading, isError, error } = useVerdicts()
  const rows = useMemo(() => latestVerdicts(data), [data])
  const storeMissing = isError && isNotYetBuilt(error)

  return (
    <div className="card">
      <div className="hd">
        <h3>Latest verdicts</h3>
        <Link className="more" to="/verdicts">Verdicts →</Link>
      </div>
      <div className="bd scroll-x" style={{ padding: '4px 16px 8px' }}>
        {isLoading ? (
          <Loading label="Loading verdicts…" />
        ) : storeMissing || rows.length === 0 ? (
          <div className="hint" style={{ padding: '12px 0' }}>
            {storeMissing
              ? 'The verdict store isn’t reachable yet — adjudications will appear here.'
              : 'No verdicts yet. Queue a matched pair on the Bench to produce one.'}
          </div>
        ) : (
          <table className="data">
            <thead>
              <tr><th>Verdict</th><th>Claim level</th><th>Replication</th><th className="right-td" /></tr>
            </thead>
            <tbody>
              {rows.map((s) => (
                <tr key={s.id} className="click">
                  <td className="mono">{s.verdict_key || s.claim_id || `#${s.id}`}</td>
                  <td><span className="tag plain">{s.claim_level}</span></td>
                  <td className="hint">{s.replication_status}</td>
                  <td className="right-td"><Link className="btn sm" to={`/verdicts/${s.id}`}>Open →</Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {rows.length ? (
        <p className="hint" style={{ padding: '0 16px 12px' }}>
          Claim level and replication are shown distinctly; positive claims are <b>candidates</b>, not established results,
          until human-promoted on the Verdict.
        </p>
      ) : null}
    </div>
  )
}

export default function Overview() {
  const navigate = useNavigate()
  const { data: jobs = [], isLoading, isError, error } = useJobs()

  const counts = useMemo(() => ({
    running: jobs.filter((j) => j.status === 'running').length,
    queued: jobs.filter((j) => j.status === 'queued').length,
    done: jobs.filter((j) => j.status === 'done').length,
    failed: jobs.filter((j) => j.status === 'error').length,
  }), [jobs])

  const running = jobs.filter((j) => j.status === 'running').slice(0, 6)

  return (
    <>
      <PageHeader
        title="Overview"
        sub="Live lab state. Every tile links to where the work is."
      />

      {isError ? (
        <ErrorState error={error} />
      ) : isLoading ? (
        <Loading label="Loading lab state…" />
      ) : (
        <>
          <div className="tiles" style={{ marginTop: 18 }}>
            <button className="tile" onClick={() => navigate('/runs')}>
              <div className="microlabel">Running</div>
              <div className="v num">{counts.running}</div>
              <div className="d">active experiments</div>
            </button>
            <button className="tile" onClick={() => navigate('/runs')}>
              <div className="microlabel">Queued</div>
              <div className="v num">{counts.queued}</div>
              <div className="d">waiting on the worker</div>
            </button>
            <button className="tile" onClick={() => navigate('/verdicts')}>
              <div className="microlabel">Completed</div>
              <div className="v num">{counts.done}</div>
              <div className="d">finished runs</div>
            </button>
            <button className="tile" onClick={() => navigate('/runs')}>
              <div className="microlabel">Failed</div>
              <div className="v num">{counts.failed}</div>
              <div className="d">kept with logs, never discarded</div>
            </button>
          </div>

          <div className="grid2" style={{ marginTop: 14 }}>
            <div className="card">
              <div className="hd">
                <h3>Now running</h3>
                <button className="more" onClick={() => navigate('/runs')}>All runs →</button>
              </div>
              <div className="bd scroll-x" style={{ padding: '4px 16px 8px' }}>
                <table className="data">
                  <thead>
                    <tr><th>Run</th><th>Preset</th><th>Dataset</th><th className="right-td">Seed</th><th>Status</th></tr>
                  </thead>
                  <tbody>
                    {running.map((j) => (
                      <tr key={j.id} className="click" onClick={() => navigate(`/runs/${j.id}`)}>
                        <td className="mono">#{j.id} {j.run_name}</td>
                        <td>{j.preset_id}</td>
                        <td>{j.dataset_name}</td>
                        <td className="right-td num">{j.seed}</td>
                        <td><StatusTag status={j.status} /></td>
                      </tr>
                    ))}
                    {running.length === 0 && (
                      <tr><td colSpan="5" className="hint" style={{ padding: '14px 12px' }}>
                        Nothing running right now. Queue an experiment from the Bench.
                      </td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <LatestVerdicts />
          </div>
        </>
      )}
    </>
  )
}
