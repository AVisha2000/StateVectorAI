import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import StatusPanel from '../components/StatusPanel'

function quantumDefaults(preset) {
  const fields = preset?.quantum_controls?.fields || []
  return Object.fromEntries(fields.map((field) => [field.key, String(field.default)]))
}

function costBand(qubits, depth) {
  const score = qubits * depth
  if (score <= 12) return 'light'
  if (score <= 28) return 'medium'
  if (score <= 48) return 'heavy'
  return 'research'
}

function resourceEstimate(selected, qubits, depth, batchSize, seqLen) {
  const cfg = selected?.config || {}
  const attnType = cfg['model.attn_type']
  const ffnType = cfg['model.ffn_type']
  const arch = cfg['model.arch'] || 'transformer'
  const blocks = Number(cfg['model.n_blocks'] || 1)
  let multiplier = 0
  if (attnType === 'quantum_proj' || attnType === 'quantum_qkv') multiplier += 2 * blocks
  if (ffnType === 'quantum' || ffnType === 'quantum_linear') multiplier += blocks
  if (arch === 'qrnn') multiplier += 2
  if (arch === 'two_stream' && cfg['model.encoder_kind'] === 'quantum') multiplier += 1
  const score = (2 ** Number(qubits || 0)) * Number(depth || 1) * Number(batchSize || 1) * Number(seqLen || 1) * Math.max(multiplier, 1)
  const band = score >= 16000000 ? 'extreme' : score >= 6000000 ? 'high' : score >= 1500000 ? 'medium' : 'low'
  return { score, band, quantumAttention: multiplier > 0 && (attnType === 'quantum_proj' || attnType === 'quantum_qkv') }
}

function parseGrid(value) {
  return Array.from(new Set(String(value)
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isInteger(item) && item > 0))).sort((a, b) => a - b)
}

export default function Launch() {
  const [params] = useSearchParams()
  const [presets, setPresets] = useState([])
  const [datasets, setDatasets] = useState([])
  const [preset, setPreset] = useState('classical-small')
  const [dataset, setDataset] = useState('default-text')
  const [runName, setRunName] = useState('')
  const [seed, setSeed] = useState(0)
  const [steps, setSteps] = useState(50)
  const [evalEvery, setEvalEvery] = useState(10)
  const [batchSize, setBatchSize] = useState(16)
  const [seqLen, setSeqLen] = useState(64)
  const [deviceTarget, setDeviceTarget] = useState('auto')
  const [queueComparison, setQueueComparison] = useState(false)
  const [quantumOverrides, setQuantumOverrides] = useState({})
  const [status, setStatus] = useState(null)
  const [queuedJob, setQueuedJob] = useState(null)
  const [queuedSweep, setQueuedSweep] = useState(null)
  const [sweepQubits, setSweepQubits] = useState('4,6,8')
  const [sweepDepths, setSweepDepths] = useState('2,4,6')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.presets().then((items) => {
      setPresets(items)
      const requested = params.get('preset')
      const first = items.find((p) => p.id === requested) || items.find((p) => p.id === preset) || items[0]
      if (first) {
        setPreset(first.id)
        setRunName(first.defaults.run_name || first.id)
        setSteps(Number(params.get('steps') || (first.id === 'classical-small' ? 50 : 25)))
        setEvalEvery(Number(params.get('eval_every') || (first.id === 'classical-small' ? 10 : 5)))
        setSeed(Number(params.get('seed') || 0))
        setQuantumOverrides(quantumDefaults(first))
      }
    }).catch((e) => setError(e.message))
    api.datasets().then((items) => {
      setDatasets(items)
      const requested = params.get('dataset')
      if (requested && items.some((d) => d.name === requested)) setDataset(requested)
    }).catch((e) => setError(e.message))
    api.status().then(setStatus).catch((e) => setError(e.message))
  }, [])

  const selected = useMemo(
    () => presets.find((p) => p.id === preset),
    [presets, preset],
  )

  const choosePreset = (p) => {
    setPreset(p.id)
    setRunName(p.defaults.run_name || p.id)
    setSteps(p.id === 'classical-small' ? 50 : 25)
    setEvalEvery(p.id === 'classical-small' ? 10 : 5)
    setQueueComparison(Boolean(p.classical_twin_id))
    setQuantumOverrides(quantumDefaults(p))
  }

  useEffect(() => {
    if (selected) {
      setQueueComparison(Boolean(selected.classical_twin_id))
      setQuantumOverrides(quantumDefaults(selected))
    }
  }, [selected?.id])

  const gpuBlocked = deviceTarget === 'gpu' && status?.gpu && !status.gpu.ready
  const quantumFields = selected?.quantum_controls?.fields || []
  const fieldMeta = Object.fromEntries(quantumFields.map((field) => [field.key, field]))
  const liveQubits = Number(
    quantumOverrides.n_qubits || quantumFields.find((field) => field.key === 'n_qubits')?.default || 0,
  )
  const liveDepth = Number(
    quantumOverrides.n_circuit_layers || quantumFields.find((field) => field.key === 'n_circuit_layers')?.default || 0,
  )
  const quantumCost = costBand(liveQubits, liveDepth)
  const resource = resourceEstimate(selected, liveQubits, liveDepth, batchSize, seqLen)
  const qubitGrid = parseGrid(sweepQubits)
  const depthGrid = parseGrid(sweepDepths)
  const sweepCount = qubitGrid.length * depthGrid.length
  const gpuReady = Boolean(status?.gpu?.ready)

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError(''); setQueuedJob(null); setQueuedSweep(null)
    try {
      const payload = {
        preset_id: preset,
        dataset_name: dataset,
        run_name: runName,
        seed: Number(seed),
        steps: Number(steps),
        eval_every: Number(evalEvery),
        batch_size: Number(batchSize),
        seq_len: Number(seqLen),
        device_target: deviceTarget,
        queue_classical_comparison: queueComparison,
      }
      if (selected?.quantum_controls?.enabled) {
        payload.quantum_overrides = Object.fromEntries(
          quantumFields.map((field) => [field.key, Number(quantumOverrides[field.key])]),
        )
      }
      const job = await api.createJob({
        ...payload,
      })
      setQueuedJob(job)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const submitSweep = async () => {
    setBusy(true); setError(''); setQueuedJob(null); setQueuedSweep(null)
    try {
      const job = await api.createSweep({
        preset_id: preset,
        dataset_name: dataset,
        run_name: runName || `${preset}-scale`,
        seed: Number(seed),
        steps: Number(steps),
        eval_every: Number(evalEvery),
        batch_size: Number(batchSize),
        seq_len: Number(seqLen),
        device_target: deviceTarget === 'cpu' ? 'cpu' : 'gpu',
        qubits: qubitGrid,
        depths: depthGrid,
      })
      setQueuedSweep(job)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <h1>Run an experiment</h1>
      <h2>Select a dataset, choose a safe preset, and queue local training.</h2>
      <StatusPanel />

      {error && <div className="alert error">{error}</div>}
      {queuedJob && (
        <div className="alert good">
          Queued <Link to={`/jobs/${queuedJob.id}`}>job #{queuedJob.id}: {queuedJob.run_name}</Link>
          {queuedJob.comparison_job && (
            <> with classical comparison <Link to={`/jobs/${queuedJob.comparison_job.id}`}>#{queuedJob.comparison_job.id}</Link></>
          )}
        </div>
      )}
      {queuedSweep && (
        <div className="alert good">
          Queued scaling sweep group <span className="mono">{queuedSweep.group_id.slice(0, 8)}</span> with {queuedSweep.count} jobs.
          {' '}Open <Link to="/experiments">Experiments</Link> to monitor the batch.
        </div>
      )}

      <form onSubmit={submit}>
        <div className="panel">
          <h3>Dataset</h3>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} ({d.source_type})
              </option>
            ))}
          </select>
          <p className="pill">Need another corpus? Import a public Hugging Face text dataset on the Datasets tab.</p>
        </div>

        <div className="panel">
          <h3>Model preset</h3>
          <div className="grid">
            {presets.map((p) => (
              <button
                type="button"
                key={p.id}
                className={`card preset ${preset === p.id ? 'selected' : ''}`}
                onClick={() => choosePreset(p)}
              >
                <h3>{p.label}</h3>
                <div className="badge">{p.kind}</div>
                <p>{p.summary}</p>
                <p className="pill">{p.cost}</p>
              </button>
            ))}
          </div>
          {selected && (
            <div className="selected-preset">
              <div>
                <h3>{selected.label}</h3>
                <p>{selected.description}</p>
                {selected.quantum_controls?.enabled && (
                  <div className="quantum-summary">
                    <div className="metric-card">
                      <div className="metric-label">Current quantum setup</div>
                      <div className="metric-value">{liveQubits} qubits x depth {liveDepth}</div>
                    </div>
                    <div className={`badge quantum-band ${quantumCost}`}>cost: {quantumCost}</div>
                    <div className={`badge quantum-band ${resource.band}`}>memory: {resource.band}</div>
                  </div>
                )}
              </div>
              <div className="kv compact">
                <div className="k">architecture</div><div className="v">{selected.architecture}</div>
                <div className="k">quantum role</div><div className="v">{selected.quantum_role}</div>
                <div className="k">recommended</div><div className="v">{selected.recommended_use}</div>
                <div className="k">classical twin</div><div className="v">{selected.classical_twin_id || 'none'}</div>
              </div>
            </div>
          )}
        </div>

        {selected?.quantum_controls?.enabled && (
          <div className="panel">
            <h3>Quantum controls</h3>
            <p className="panel-copy">{selected.quantum_controls.summary}</p>
            <div className="form-grid">
              {quantumFields.map((field) => (
                <label key={field.key}>
                  {field.label}
                  <input
                    type="number"
                    min={field.min}
                    max={deviceTarget === 'gpu' ? field.gpu_max : field.max}
                    step={field.step}
                    value={quantumOverrides[field.key] ?? ''}
                    onChange={(e) => setQuantumOverrides((prev) => ({
                      ...prev,
                      [field.key]: e.target.value,
                    }))}
                  />
                  <span className="field-help">Safe range {field.min} to {field.max}</span>
                  {field.gpu_max > field.max && (
                    <span className="field-help">GPU range up to {field.gpu_max}</span>
                  )}
                </label>
              ))}
            </div>
            <div className="quantum-advice">
              <p className="pill">
                Current plan: {liveQubits} qubits, depth {liveDepth}. Simulation cost rises much faster with
                qubits than with depth, so widen carefully.
              </p>
              <p className="pill">{selected.quantum_controls.warning}</p>
              <p className="pill">{selected.quantum_controls.comparison_note}</p>
            </div>
            {resource.band !== 'low' && (
              <div className={`alert ${resource.band === 'extreme' ? 'error' : ''}`}>
                <b>Memory preflight: {resource.band}</b>
                <p className="panel-copy">
                  Estimated circuit workload score {Math.round(resource.score).toLocaleString()}.
                  {resource.quantumAttention
                    ? ' Quantum attention runs the circuit on every token, so reduce batch size or sequence length before scaling qubits/depth.'
                    : ' Reduce batch size or sequence length if JAX reports allocator pressure.'}
                </p>
              </div>
            )}
          </div>
        )}

        {selected?.quantum_controls?.enabled && (
          <div className="panel">
            <div className="workspace-header">
              <div>
                <h3>Scaling sweep</h3>
                <p className="panel-copy">
                  Queue a performance grid over qubit count and circuit depth. Each point becomes its own job
                  and is labeled with q/depth in the run name and result variant.
                </p>
              </div>
              <span className={`badge ${gpuReady ? 'done' : 'cancelled'}`}>{gpuReady ? 'GPU ready' : 'GPU not detected'}</span>
            </div>
            <div className="form-grid">
              <label>Qubit grid
                <input value={sweepQubits} onChange={(e) => setSweepQubits(e.target.value)} placeholder="4,6,8,10" />
                <span className="field-help">
                  CPU max {fieldMeta.n_qubits?.max}; GPU max {fieldMeta.n_qubits?.gpu_max || fieldMeta.n_qubits?.max}
                </span>
              </label>
              <label>Depth grid
                <input value={sweepDepths} onChange={(e) => setSweepDepths(e.target.value)} placeholder="2,4,6,8" />
                <span className="field-help">
                  CPU max {fieldMeta.n_circuit_layers?.max}; GPU max {fieldMeta.n_circuit_layers?.gpu_max || fieldMeta.n_circuit_layers?.max}
                </span>
              </label>
              <label>Sweep target
                <select value={deviceTarget === 'cpu' ? 'cpu' : 'gpu'} onChange={(e) => setDeviceTarget(e.target.value)}>
                  <option value="gpu" disabled={status?.gpu && !status.gpu.ready}>gpu</option>
                  <option value="cpu">cpu</option>
                </select>
              </label>
              <label>Batch size
                <input type="number" min="1" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
                <span className="field-help">Use 1-4 for high-qubit quantum attention.</span>
              </label>
              <label>Sequence length
                <input type="number" min="8" value={seqLen} onChange={(e) => setSeqLen(e.target.value)} />
                <span className="field-help">Use 16-32 when exploring depth or 8+ qubits.</span>
              </label>
            </div>
            <div className="quantum-advice">
              <p className="pill">
                Preview: {qubitGrid.join(', ') || '-'} qubits x {depthGrid.join(', ') || '-'} depths = {sweepCount} jobs.
                Use short step counts first, then rerun promising points longer.
              </p>
              <p className="pill">
                The worker is still single-job local queueing; the GPU target makes each point larger/faster, not parallel.
              </p>
            </div>
            {deviceTarget !== 'cpu' && status?.gpu && !status.gpu.ready && (
              <div className="alert error">GPU sweeps are blocked until JAX reports a GPU backend.</div>
            )}
            <button
              type="button"
              className="primary"
              disabled={busy || sweepCount < 1 || sweepCount > 64 || (deviceTarget !== 'cpu' && status?.gpu && !status.gpu.ready)}
              onClick={submitSweep}
            >
              {busy ? 'Queueing...' : `Queue ${sweepCount} scaling jobs`}
            </button>
          </div>
        )}

        <div className="panel">
          <h3>Run settings</h3>
          <div className="form-grid">
            <label>Run name<input value={runName} onChange={(e) => setRunName(e.target.value)} /></label>
            <label>Seed<input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} /></label>
            <label>Steps<input type="number" min="1" value={steps} onChange={(e) => setSteps(e.target.value)} /></label>
            <label>Eval every<input type="number" min="1" value={evalEvery} onChange={(e) => setEvalEvery(e.target.value)} /></label>
            <label>Batch size<input type="number" min="1" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} /></label>
            <label>Sequence length<input type="number" min="8" value={seqLen} onChange={(e) => setSeqLen(e.target.value)} /></label>
            <label>Device target
              <select value={deviceTarget} onChange={(e) => setDeviceTarget(e.target.value)}>
                <option value="auto">auto</option>
                <option value="cpu">cpu</option>
                <option value="gpu" disabled={status?.gpu && !status.gpu.ready}>gpu</option>
              </select>
            </label>
          </div>
          {selected?.classical_twin_id && (
            <label className="check-row">
              <input
                type="checkbox"
                checked={queueComparison}
                onChange={(e) => setQueueComparison(e.target.checked)}
              />
              Queue classical comparison ({selected.classical_twin_id})
            </label>
          )}
          {deviceTarget === 'gpu' && status?.gpu && !status.gpu.ready && (
            <div className="alert error">
              GPU target is blocked because JAX currently reports {status.gpu.jax_backend || 'no'} backend.
              Open the <Link to="/gpu">GPU page</Link> for setup guidance.
            </div>
          )}
          {selected && (
            <p className="pill">
              {selected.label}: {selected.summary} Start with 5-50 steps for smoke tests,
              then increase once the curves look sane.
            </p>
          )}
          <button className="primary" disabled={busy || !preset || !dataset || gpuBlocked}>
            {busy ? 'Queueing...' : queueComparison ? 'Queue experiment + comparison' : 'Queue experiment'}
          </button>
        </div>
      </form>
    </div>
  )
}
