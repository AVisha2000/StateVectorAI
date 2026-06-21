import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'

function canCancel(job) {
  return job.status === 'queued' || job.status === 'running'
}

export default function Jobs() {
  const [jobs, setJobs] = useState([])
  const [error, setError] = useState('')
  const [status, setStatus] = useState('all')
  const [dataset, setDataset] = useState('all')
  const [preset, setPreset] = useState('all')
  const [family, setFamily] = useState('all')
  const [group, setGroup] = useState('all')

  const refresh = () => api.jobs().then(setJobs).catch((e) => setError(e.message))

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 2000)
    return () => clearInterval(id)
  }, [])

  const counts = useMemo(() => ({
    queued: jobs.filter((j) => j.status === 'queued').length,
    running: jobs.filter((j) => j.status === 'running').length,
    done: jobs.filter((j) => j.status === 'done').length,
    error: jobs.filter((j) => j.status === 'error').length,
    cancelled: jobs.filter((j) => j.status === 'cancelled').length,
  }), [jobs])

  const options = (key) => Array.from(new Set(jobs.map((j) => j[key]).filter(Boolean))).sort()
  const filtered = jobs.filter((j) => (
    (status === 'all' || j.status === status)
    && (dataset === 'all' || j.dataset_name === dataset)
    && (preset === 'all' || j.preset_id === preset)
    && (family === 'all' || j.model_family === family)
    && (group === 'all' || j.group_id === group)
  ))
  const groupCounts = useMemo(() => jobs.reduce((acc, job) => {
    if (!job.group_id) return acc
    acc[job.group_id] = (acc[job.group_id] || 0) + 1
    return acc
  }, {}), [jobs])
  const isScaled = (job) => job.config?.['lab.quantum_override.n_qubits'] && job.config?.['lab.quantum_override.n_circuit_layers']

  const cancel = async (jobId) => {
    setError('')
    try {
      await api.cancelJob(jobId)
      refresh()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div>
      <h1>Experiments</h1>
      <h2>{counts.running} running - {counts.queued} queued - {counts.done} done - {counts.error} failed - {counts.cancelled} cancelled</h2>
      {error && <div className="alert error">{error}</div>}

      <div className="action-grid">
        <Link className="action-card" to="/launch">
          <b>Run classical baseline</b>
          <span>Start with classical-small to establish a dataset floor.</span>
        </Link>
        <Link className="action-card" to="/launch">
          <b>Queue comparison</b>
          <span>Pick a quantum preset; the classical twin is enabled by default.</span>
        </Link>
        <Link className="action-card" to="/datasets">
          <b>Import dataset</b>
          <span>Load a public Hugging Face text corpus into the local registry.</span>
        </Link>
        <Link className="action-card" to="/gpu">
          <b>Run GPU check</b>
          <span>Confirm JAX can see CUDA before requesting GPU runs.</span>
        </Link>
      </div>

      <div className="panel">
        <div className="chips">
          {['all', 'queued', 'running', 'done', 'error', 'cancelled'].map((item) => (
            <button key={item} className={`chip ${status === item ? 'on' : ''}`} onClick={() => setStatus(item)}>{item}</button>
          ))}
        </div>
        <div className="form-grid">
          <label>Dataset
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              <option value="all">all</option>
              {options('dataset_name').map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label>Preset
            <select value={preset} onChange={(e) => setPreset(e.target.value)}>
              <option value="all">all</option>
              {options('preset_id').map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label>Model family
            <select value={family} onChange={(e) => setFamily(e.target.value)}>
              <option value="all">all</option>
              {options('model_family').map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label>Group
            <select value={group} onChange={(e) => setGroup(e.target.value)}>
              <option value="all">all</option>
              {options('group_id').map((item) => <option key={item} value={item}>{item.slice(0, 8)} ({groupCounts[item]})</option>)}
            </select>
          </label>
        </div>
      </div>

      <div className="panel table-panel">
        <table>
          <thead>
            <tr>
              <th>Job</th>
              <th>Status</th>
              <th>Role</th>
              <th>Preset</th>
              <th>Dataset</th>
              <th className="num">Seed</th>
              <th className="num">Steps</th>
              <th>Target</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((j) => (
              <tr key={j.id}>
                <td>
                  <Link to={`/jobs/${j.id}`}>#{j.id} {j.run_name}</Link>
                  {j.group_id && <div className="muted mono">group {j.group_id.slice(0, 8)}</div>}
                </td>
                <td><span className={`badge ${j.status}`}>{j.status}</span></td>
                <td><span className="badge">{j.comparison_role || 'primary'}</span></td>
                <td>{j.preset_id}</td>
                <td>{j.dataset_name}</td>
                <td className="num">{j.seed}</td>
                <td className="num">{j.steps}</td>
                <td>{j.device_target || 'auto'}</td>
                <td className="num">
                  {j.compare_to_job_id && <Link className="small-link" to={`/comparisons/${j.id}`}>Compare</Link>}
                  {' '}
                  {j.group_id && groupCounts[j.group_id] > 1 && isScaled(j) && (
                    <Link className="small-link" to={`/scaling/${j.group_id}`}>Scaling</Link>
                  )}
                  {' '}
                  <Link className="small-link" to={`/launch?preset=${encodeURIComponent(j.preset_id)}&dataset=${encodeURIComponent(j.dataset_name)}&seed=${j.seed}&steps=${j.steps}&eval_every=${j.eval_every}`}>Rerun</Link>
                  {' '}
                  {canCancel(j) && (
                    <button className="small" onClick={() => cancel(j.id)}>Cancel</button>
                  )}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr><td colSpan="9">No lab jobs yet. Queue one from Run.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
