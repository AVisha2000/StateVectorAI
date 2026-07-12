import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useJobs } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState, StatusTag, rowActivation } from '../lib/ui.jsx'
import { filterRuns, uniqueDatasets } from '../lib/runsFilter.js'

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'running', label: 'Running' },
  { key: 'queued', label: 'Queued' },
  { key: 'done', label: 'Done' },
  { key: 'error', label: 'Failed' },
]

export default function Runs() {
  const { data: jobs = [], isLoading, isError, error } = useJobs()
  const [filter, setFilter] = useState('all')
  const [dataset, setDataset] = useState('all')
  const [search, setSearch] = useState('')
  const navigate = useNavigate()

  const datasets = useMemo(() => uniqueDatasets(jobs), [jobs])
  const rows = useMemo(
    () => filterRuns(jobs, { status: filter, dataset, search }),
    [jobs, filter, dataset, search],
  )

  return (
    <>
      <PageHeader
        title="Runs"
        sub="Every run in one table — live, queued, finished, failed. No separate results silos."
      />

      {isError ? (
        <ErrorState error={error} />
      ) : isLoading ? (
        <Loading label="Loading runs…" />
      ) : (
        <>
          <div className="row" style={{ margin: '16px 0 12px', gap: 10 }}>
            <div className="seg" role="tablist">
              {FILTERS.map((f) => (
                <button
                  key={f.key}
                  className={filter === f.key ? 'on' : undefined}
                  onClick={() => setFilter(f.key)}
                >
                  {f.label}
                </button>
              ))}
            </div>
            <label className="row" style={{ gap: 6 }}>
              <span className="microlabel">Dataset</span>
              <select className="mini" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                <option value="all">all</option>
                {datasets.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </label>
            <input
              className="mini"
              type="search"
              placeholder="Search runs…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ minWidth: 180 }}
              aria-label="Search runs"
            />
            <span className="hint spacer">{rows.length} run{rows.length === 1 ? '' : 's'}</span>
          </div>

          <div className="card scroll-x">
            <table className="data">
              <thead>
                <tr>
                  <th>Run</th><th>Role</th><th>Preset</th><th>Dataset</th>
                  <th className="right-td">Seed</th><th className="right-td">Steps</th>
                  <th>Analogue</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((j) => (
                  <tr key={j.id} className="click" aria-label={`Open run ${j.run_name}`} {...rowActivation(() => navigate(`/runs/${j.id}`))}>
                    <td className="mono">#{j.id} {j.run_name}</td>
                    <td><span className="tag plain">{j.comparison_role || 'primary'}</span></td>
                    <td>{j.preset_id}</td>
                    <td>{j.dataset_name}</td>
                    <td className="right-td num">{j.seed}</td>
                    <td className="right-td num">{j.steps}</td>
                    <td><span className="tag plain">{j.analogue_state || 'none'}</span></td>
                    <td><StatusTag status={j.status} /></td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr><td colSpan="8" className="hint" style={{ padding: '16px 12px' }}>
                    No runs match these filters.
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="hint" style={{ marginTop: 10 }}>
            Row → run detail with diagnostics and the twin comparison. Failed runs keep their logs and are never silently discarded.
          </p>
        </>
      )}
    </>
  )
}
