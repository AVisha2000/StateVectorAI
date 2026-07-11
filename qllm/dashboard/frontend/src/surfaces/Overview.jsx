import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useJobs } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState, StatusTag } from '../lib/ui.jsx'

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

          <div className="card" style={{ marginTop: 14 }}>
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
                    <tr key={j.id}>
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
        </>
      )}
    </>
  )
}
