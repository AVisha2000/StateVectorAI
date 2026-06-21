import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'

function parseInts(value) {
  return Array.from(new Set(String(value)
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isInteger(item) && item >= 0))).sort((a, b) => a - b)
}

function parsePositiveInts(value) {
  return parseInts(value).filter((item) => item > 0)
}

function fmt(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return '-'
  return Number(value).toFixed(digits)
}

export default function Studies() {
  const [studies, setStudies] = useState([])
  const [presets, setPresets] = useState([])
  const [datasets, setDatasets] = useState([])
  const [name, setName] = useState('quantum-ffn-multiseed-study')
  const [question, setQuestion] = useState('Does the quantum candidate beat its matched classical analogue across seeds?')
  const [task, setTask] = useState('Language modelling')
  const [candidate, setCandidate] = useState('quantum-ffn-4q')
  const [selectedDatasets, setSelectedDatasets] = useState(['default-text'])
  const [controls, setControls] = useState([])
  const [seeds, setSeeds] = useState('0,1,2')
  const [steps, setSteps] = useState(50)
  const [evalEvery, setEvalEvery] = useState(10)
  const [batchSize, setBatchSize] = useState(16)
  const [seqLen, setSeqLen] = useState(64)
  const [qubits, setQubits] = useState('4')
  const [depths, setDepths] = useState('2')
  const [deviceTarget, setDeviceTarget] = useState('auto')
  const [queueNow, setQueueNow] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [created, setCreated] = useState(null)

  const refresh = () => api.studies().then(setStudies).catch((e) => setError(e.message))

  useEffect(() => {
    refresh()
    api.presets().then((items) => {
      setPresets(items)
      if (items.some((p) => p.id === 'quantum-ffn-4q')) setCandidate('quantum-ffn-4q')
      else if (items[0]) setCandidate(items[0].id)
    }).catch((e) => setError(e.message))
    api.datasets().then((items) => {
      setDatasets(items)
      if (items[0]) setSelectedDatasets([items[0].name])
    }).catch((e) => setError(e.message))
  }, [])

  const candidatePreset = useMemo(() => presets.find((p) => p.id === candidate), [presets, candidate])
  const jobCount = selectedDatasets.length
    * Math.max(parseInts(seeds).length, 1)
    * Math.max(parsePositiveInts(qubits).length || 1, 1)
    * Math.max(parsePositiveInts(depths).length || 1, 1)
  const baselineCount = candidatePreset?.classical_analogue ? jobCount : 0

  const toggleDataset = (dataset) => {
    setSelectedDatasets((prev) => (
      prev.includes(dataset)
        ? prev.filter((item) => item !== dataset)
        : [...prev, dataset]
    ))
  }

  const toggleControl = (preset) => {
    setControls((prev) => (
      prev.includes(preset)
        ? prev.filter((item) => item !== preset)
        : [...prev, preset]
    ))
  }

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError(''); setCreated(null)
    try {
      const payload = {
        name,
        research_question: question,
        task,
        dataset_names: selectedDatasets,
        candidate_preset_id: candidate,
        baseline_policy: 'analogue',
        control_preset_ids: controls,
        seeds: parseInts(seeds),
        steps: Number(steps),
        eval_every: Number(evalEvery),
        batch_size: Number(batchSize),
        seq_len: Number(seqLen),
        device_target: deviceTarget,
        queue_now: queueNow,
        queue_analogues: true,
        sweep: {
          qubits: parsePositiveInts(qubits),
          depths: parsePositiveInts(depths),
        },
      }
      const study = await api.createStudy(payload)
      setCreated(study)
      await refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <div className="workspace-header">
        <div>
          <h1>Studies</h1>
          <h2>Create multi-seed protocols with matched baselines, controls, and qubit/depth grids.</h2>
        </div>
        <Link className="small" to="/scaling">Legacy scaling tests</Link>
      </div>

      {error && <div className="alert error">{error}</div>}
      {created && (
        <div className="alert good">
          Study <Link to={`/studies/${created.id}`}>#{created.id}: {created.name}</Link> created
          {created.job_count ? ` with ${created.job_count} linked job(s).` : '.'}
        </div>
      )}

      <form onSubmit={submit} className="panel">
        <h3>Study Protocol</h3>
        <div className="form-grid">
          <label>Name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
          <label>Task<input value={task} onChange={(e) => setTask(e.target.value)} /></label>
          <label>Candidate preset
            <select value={candidate} onChange={(e) => setCandidate(e.target.value)}>
              {presets.map((preset) => <option key={preset.id} value={preset.id}>{preset.label} ({preset.kind})</option>)}
            </select>
          </label>
          <label>Device target
            <select value={deviceTarget} onChange={(e) => setDeviceTarget(e.target.value)}>
              <option value="auto">auto</option>
              <option value="cpu">cpu</option>
              <option value="gpu">gpu</option>
            </select>
          </label>
        </div>
        <label>Research question<input value={question} onChange={(e) => setQuestion(e.target.value)} /></label>

        <div className="workspace-grid">
          <section>
            <div className="pill">datasets</div>
            <div className="chips">
              {datasets.map((dataset) => (
                <label key={dataset.name} className="check-row">
                  <input
                    type="checkbox"
                    checked={selectedDatasets.includes(dataset.name)}
                    onChange={() => toggleDataset(dataset.name)}
                  />
                  {dataset.name}
                </label>
              ))}
            </div>
          </section>
          <section>
            <div className="pill">controls</div>
            <div className="chips">
              {presets.filter((preset) => preset.id !== candidate).map((preset) => (
                <label key={preset.id} className="check-row">
                  <input
                    type="checkbox"
                    checked={controls.includes(preset.id)}
                    onChange={() => toggleControl(preset.id)}
                  />
                  {preset.label}
                </label>
              ))}
            </div>
          </section>
        </div>

        <div className="form-grid">
          <label>Seeds<input value={seeds} onChange={(e) => setSeeds(e.target.value)} placeholder="0,1,2" /></label>
          <label>Qubit grid<input value={qubits} onChange={(e) => setQubits(e.target.value)} placeholder="4,6,8" /></label>
          <label>Depth grid<input value={depths} onChange={(e) => setDepths(e.target.value)} placeholder="2,4,6" /></label>
          <label>Steps<input type="number" min="1" value={steps} onChange={(e) => setSteps(e.target.value)} /></label>
          <label>Eval every<input type="number" min="1" value={evalEvery} onChange={(e) => setEvalEvery(e.target.value)} /></label>
          <label>Batch<input type="number" min="1" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} /></label>
          <label>Seq len<input type="number" min="8" value={seqLen} onChange={(e) => setSeqLen(e.target.value)} /></label>
        </div>

        {candidatePreset?.classical_analogue && (
          <div className="alert">
            Matched analogue will be queued for candidate jobs: {candidatePreset.classical_analogue.label}.
            Fairness checks preserve dataset, seed, steps, eval cadence, preprocessing, batch size, and sequence length.
          </div>
        )}
        {!candidatePreset?.classical_analogue && (
          <div className="alert error">This candidate has no detected analogue; the study can still queue, but evidence will remain incomplete.</div>
        )}

        <label className="check-row">
          <input type="checkbox" checked={queueNow} onChange={(e) => setQueueNow(e.target.checked)} />
          Queue jobs immediately
        </label>
        <p className="pill">
          Preview: {jobCount} candidate job(s), {baselineCount} baseline job(s), {controls.length * jobCount} control job(s).
        </p>
        <button className="primary" disabled={busy || !selectedDatasets.length || !parseInts(seeds).length}>
          {busy ? 'Creating...' : 'Create study'}
        </button>
      </form>

      <section className="panel table-panel">
        <h3>Existing Studies</h3>
        <table>
          <thead>
            <tr><th>Study</th><th>Candidate</th><th>Datasets</th><th>Seeds</th><th>Jobs</th><th>Evidence</th></tr>
          </thead>
          <tbody>
            {studies.map((study) => (
              <tr key={study.id}>
                <td><Link to={`/studies/${study.id}`}>#{study.id} {study.name}</Link><div className="muted">{study.research_question}</div></td>
                <td>{study.candidate_preset_id}</td>
                <td>{(study.dataset_names || []).join(', ')}</td>
                <td>{(study.seeds || []).join(', ')}</td>
                <td>{study.job_count}</td>
                <td><span className="badge">{study.evidence?.label || 'pending'}</span></td>
              </tr>
            ))}
            {studies.length === 0 && <tr><td colSpan="6">No studies yet. Create a protocol above.</td></tr>}
          </tbody>
        </table>
      </section>
    </div>
  )
}
