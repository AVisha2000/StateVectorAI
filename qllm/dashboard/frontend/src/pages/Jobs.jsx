import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import EvidenceWarnings from '../components/EvidenceWarnings'

function canCancel(job) {
  return job.status === 'queued' || job.status === 'running'
}

export default function Jobs() {
  const [jobs, setJobs] = useState([])
  const [loaded, setLoaded] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('all')
  const [dataset, setDataset] = useState('all')
  const [preset, setPreset] = useState('all')
  const [family, setFamily] = useState('all')
  const [group, setGroup] = useState('all')
  const [notice, setNotice] = useState('')

  const refresh = () => api.jobs()
    .then((payload) => {
      setJobs(payload)
      setLoaded(true)
      setError('')
    })
    .catch((e) => setError(e.message))
    .finally(() => setLoading(false))

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
  const filtersActive = status !== 'all' || dataset !== 'all' || preset !== 'all' || family !== 'all' || group !== 'all'
  const groupCounts = useMemo(() => jobs.reduce((acc, job) => {
    if (!job.group_id) return acc
    acc[job.group_id] = (acc[job.group_id] || 0) + 1
    return acc
  }, {}), [jobs])
  const activeGpuReservations = jobs.filter((j) => (
    ['queued', 'running'].includes(j.status) && j.gpu_reservation?.required
  ))
  const activeHighMemory = jobs.filter((j) => (
    ['queued', 'running'].includes(j.status) && j.gpu_reservation?.high_memory
  ))
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

  const queueAnalogue = async (jobId) => {
    setError(''); setNotice('')
    try {
      const job = await api.queueClassicalAnalogue(jobId)
      setNotice(`Queued classical analogue for job #${job.id}.`)
      refresh()
    } catch (e) {
      setError(e.message)
    }
  }

  const queueGroupAnalogues = async () => {
    if (group === 'all') return
    setError(''); setNotice('')
    try {
      const payload = await api.queueGroupClassicalAnalogues(group)
      setNotice(`Queued ${payload.count} classical analogue job(s) for group ${group.slice(0, 8)}.`)
      refresh()
    } catch (e) {
      setError(e.message)
    }
  }

  if (loading) return <div className="loading">Loading experiments...</div>
  if (!loaded) return <div><h1>Experiments</h1><div className="alert error">{error}</div></div>

  return (
    <div>
      <h1>Experiments</h1>
      <h2>{counts.running} running - {counts.queued} queued - {counts.done} done - {counts.error} failed - {counts.cancelled} cancelled</h2>
      {error && <div className="alert error">{error}</div>}
      {notice && <div className="alert good">{notice}</div>}

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

      <section className="panel">
        <div className="workspace-header">
          <div>
            <h3>GPU Reservation and Memory Warnings</h3>
            <p className="panel-copy">
              GPU-targeted jobs reserve the repo-managed exclusive execution lane. High-memory quantum simulations are flagged before they run.
            </p>
          </div>
          <span className={`badge ${activeGpuReservations.some((j) => j.status === 'running') ? 'running' : activeGpuReservations.length ? 'queued' : 'done'}`}>
            {activeGpuReservations.length ? `${activeGpuReservations.length} reserved/waiting` : 'lane idle'}
          </span>
        </div>
        {activeHighMemory.length > 0 && (
          <div className="alert">
            {activeHighMemory.length} active high-memory quantum job(s). Keep batch size and sequence length conservative before scaling qubits or depth.
          </div>
        )}
      </section>

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
        {group !== 'all' && (
          <button className="small" type="button" onClick={queueGroupAnalogues}>
            Queue missing classical analogues for this group
          </button>
        )}
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
              <th>Analogue</th>
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
                  <EvidenceWarnings warnings={j.interpretation_warnings} compact />
                </td>
                <td><span className={`badge ${j.status}`}>{j.status}</span></td>
                <td><span className="badge">{j.comparison_role || 'primary'}</span></td>
                <td>{j.preset_id}</td>
                <td>{j.dataset_name}</td>
                <td className="num">{j.seed}</td>
                <td className="num">{j.steps}</td>
                <td>
                  <span className={`badge ${j.analogue_state === 'missing' ? 'error' : j.analogue_state === 'done' ? 'done' : ''}`}>
                    {j.analogue_state || 'none'}
                  </span>
                  {j.analogue_job_id && <div><Link className="small-link" to={`/jobs/${j.analogue_job_id}`}>analogue #{j.analogue_job_id}</Link></div>}
                  {j.analogue_state === 'missing' && (
                    <button className="small" type="button" onClick={() => queueAnalogue(j.id)}>Queue analogue</button>
                  )}
                </td>
                <td>
                  {j.device_target || 'auto'}
                  {j.gpu_reservation?.required && (
                    <div><span className={`badge ${j.gpu_reservation.state === 'active' ? 'running' : 'queued'}`}>gpu lane {j.gpu_reservation.state}</span></div>
                  )}
                  {j.gpu_reservation?.high_memory && (
                    <div><span className={`badge quantum-band ${j.gpu_reservation.resource_band || 'high'}`}>memory {j.gpu_reservation.resource_band || 'high'}</span></div>
                  )}
                </td>
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
              <tr><td colSpan="10">
                {filtersActive
                  ? 'No experiments match the active filters. Clear or change a filter to see other jobs.'
                  : 'No lab jobs yet. Queue one from Run.'}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
