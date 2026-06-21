import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import ModelDiagram from '../components/ModelDiagram'

const ATTN_TYPES = ['classical', 'quantum_proj', 'quantum_qkv']
const FFN_TYPES = ['classical', 'quantum', 'quantum_linear', 'lowrank']
const ANSATZ = ['reuploading', 'hardware_efficient']
const READOUT = ['z', 'zz']
const BACKENDS = ['pennylane', 'tensorcircuit']

function flatToNested(flat = {}) {
  const out = { model: {}, train: {}, data: {}, tracking: {} }
  Object.entries(flat).forEach(([key, value]) => {
    const parts = key.split('.')
    let cursor = out
    parts.forEach((part, index) => {
      const last = index === parts.length - 1
      const next = parts[index + 1]
      if (last) {
        cursor[part] = value
      } else {
        cursor[part] = cursor[part] || (/^\d+$/.test(next) ? [] : {})
        cursor = cursor[part]
      }
    })
  })
  return out
}

function clone(value) {
  return JSON.parse(JSON.stringify(value))
}

function ensureBlocks(config) {
  const draft = clone(config)
  const model = draft.model
  const n = Number(model.n_blocks || 0)
  if (!model.blocks) {
    model.blocks = Array.from({ length: n }, () => ({
      attn_type: model.attn_type || 'classical',
      ffn_type: model.ffn_type || 'classical',
      quantum: clone(model.quantum),
    }))
  }
  return draft
}

function changeAt(config, path, value) {
  const next = clone(config)
  const parts = path.split('.')
  let cursor = next
  parts.slice(0, -1).forEach((part) => { cursor = cursor[part] })
  cursor[parts[parts.length - 1]] = value
  return next
}

function diffConfigs(current, base, prefix = '') {
  const keys = new Set([...Object.keys(current || {}), ...Object.keys(base || {})])
  const changes = []
  Array.from(keys).sort().forEach((key) => {
    const path = prefix ? `${prefix}.${key}` : key
    const a = current?.[key]
    const b = base?.[key]
    if (a && b && typeof a === 'object' && !Array.isArray(a) && !Array.isArray(b)) {
      changes.push(...diffConfigs(a, b, path))
    } else if (JSON.stringify(a) !== JSON.stringify(b)) {
      changes.push({ path, before: b, after: a })
    }
  })
  return changes
}

function quantumNodes(graph) {
  return (graph?.nodes || []).filter((node) => node.kind === 'quantum')
}

export default function Models() {
  const [presets, setPresets] = useState([])
  const [specs, setSpecs] = useState([])
  const [datasets, setDatasets] = useState([])
  const [source, setSource] = useState(null)
  const [savedSpec, setSavedSpec] = useState(null)
  const [draft, setDraft] = useState(null)
  const [baseDraft, setBaseDraft] = useState(null)
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [tab, setTab] = useState('overview')
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [validation, setValidation] = useState(null)
  const [runSettings, setRunSettings] = useState({ dataset_name: 'default-text', seed: 0, steps: 50, eval_every: 10, device_target: 'auto', batch_size: 16, seq_len: 64 })
  const [queued, setQueued] = useState(null)
  const [error, setError] = useState('')

  const refreshSpecs = () => api.modelSpecs().then(setSpecs).catch((e) => setError(e.message))

  useEffect(() => {
    api.presets().then((items) => {
      setPresets(items)
      if (items[0]) loadPreset(items[0])
    }).catch((e) => setError(e.message))
    api.datasets().then((items) => {
      setDatasets(items)
      if (items[0]) setRunSettings((prev) => ({ ...prev, dataset_name: items[0].name }))
    }).catch((e) => setError(e.message))
    refreshSpecs()
  }, [])

  useEffect(() => {
    if (!draft) return
    api.validateModelSpec({ config: draft }).then(setValidation).catch((e) => setError(e.message))
  }, [draft])

  const graph = validation?.graph
  const changes = useMemo(() => diffConfigs(draft || {}, baseDraft || {}), [draft, baseDraft])
  const qnodes = quantumNodes(graph)
  const layer = draft?.model?.blocks?.[selectedLayer]

  function loadPreset(preset) {
    const config = ensureBlocks(flatToNested(preset.config))
    setSource({ type: 'preset', id: preset.id, label: preset.label })
    setSavedSpec(null)
    setDraft(config)
    setBaseDraft(clone(config))
    setName(`${preset.label} editable`)
    setNotes('')
    setSelectedLayer(0)
    setQueued(null)
  }

  function loadSpec(spec) {
    const config = ensureBlocks(spec.config)
    setSource({ type: 'spec', id: spec.id, label: spec.name })
    setSavedSpec(spec)
    setDraft(config)
    setBaseDraft(clone(config))
    setName(spec.name)
    setNotes(spec.notes || '')
    setSelectedLayer(0)
    setQueued(null)
  }

  function edit(path, value) {
    setDraft((prev) => changeAt(prev, path, value))
  }

  function editNumber(path, value) {
    const n = Number(value)
    edit(path, Number.isNaN(n) ? 0 : n)
  }

  function editBool(path, value) {
    edit(path, Boolean(value))
  }

  async function saveSpec() {
    setError('')
    try {
      const payload = {
        name,
        notes,
        source: source ? `${source.type}:${source.id}` : undefined,
        parent_id: savedSpec?.id || undefined,
        version: savedSpec ? Number(savedSpec.version || 1) + 1 : 1,
        config: draft,
      }
      const spec = await api.createModelSpec(payload)
      setSavedSpec(spec)
      setBaseDraft(clone(spec.config))
      setSource({ type: 'spec', id: spec.id, label: spec.name })
      await refreshSpecs()
    } catch (e) {
      setError(e.message)
    }
  }

  async function updateCurrentSpec() {
    if (!savedSpec) return saveSpec()
    setError('')
    try {
      const spec = await api.updateModelSpec(savedSpec.id, { name, notes, config: draft })
      setSavedSpec(spec)
      setBaseDraft(clone(spec.config))
      await refreshSpecs()
    } catch (e) {
      setError(e.message)
    }
  }

  async function runSpec() {
    setError(''); setQueued(null)
    try {
      const spec = savedSpec || await api.createModelSpec({ name, notes, source: source ? `${source.type}:${source.id}` : undefined, config: draft })
      setSavedSpec(spec)
      const job = await api.runModelSpec(spec.id, { ...runSettings, run_name: `${spec.name}-run` })
      setQueued(job)
      await refreshSpecs()
    } catch (e) {
      setError(e.message)
    }
  }

  if (!draft) return <div className="loading">Loading model builder...</div>

  return (
    <div>
      <div className="workspace-header">
        <div>
          <h1>Model Builder</h1>
          <h2>Editable model specs, layer-level swaps, quantum inspector, and constrained circuit preview.</h2>
        </div>
        <div className="header-actions">
          <button className="small" onClick={saveSpec}>Save as version</button>
          <button className="small" onClick={updateCurrentSpec}>Update current</button>
          <button className="primary" type="button" onClick={runSpec}>Run spec</button>
        </div>
      </div>

      {error && <div className="alert error">{error}</div>}
      {queued && <div className="alert good">Queued <Link to={`/jobs/${queued.id}`}>job #{queued.id}: {queued.run_name}</Link></div>}

      <div className="builder-shell">
        <aside className="builder-sidebar">
          <section className="panel">
            <h3>Saved specs</h3>
            <div className="model-list">
              {specs.map((spec) => (
                <button key={spec.id} className={`model-list-item ${savedSpec?.id === spec.id ? 'selected' : ''}`} onClick={() => loadSpec(spec)}>
                  <b>{spec.name}</b>
                  <span>v{spec.version} - {spec.source || 'custom'}</span>
                </button>
              ))}
              {specs.length === 0 && <p className="muted">No saved specs yet.</p>}
            </div>
          </section>
          <section className="panel">
            <h3>Start from preset</h3>
            <div className="model-list">
              {presets.map((preset) => (
                <button key={preset.id} className="model-list-item" onClick={() => loadPreset(preset)}>
                  <b>{preset.label}</b>
                  <span>{preset.kind} - {preset.cost}</span>
                </button>
              ))}
            </div>
          </section>
        </aside>

        <main className="builder-main">
          <section className="panel">
            <div className="form-grid">
              <label>Spec name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
              <label>Notes<input value={notes} onChange={(e) => setNotes(e.target.value)} /></label>
              <label>Architecture
                <select value={draft.model.arch} onChange={(e) => edit('model.arch', e.target.value)}>
                  <option value="transformer">transformer</option>
                  <option value="gru">gru</option>
                  <option value="qrnn">qrnn</option>
                  <option value="two_stream">two_stream</option>
                </select>
              </label>
              <label>Blocks<input type="number" min="1" value={draft.model.n_blocks} onChange={(e) => {
                const count = Number(e.target.value)
                const next = ensureBlocks(changeAt(draft, 'model.n_blocks', count))
                next.model.blocks = Array.from({ length: count }, (_, i) => next.model.blocks[i] || { attn_type: next.model.attn_type, ffn_type: next.model.ffn_type, quantum: clone(next.model.quantum) })
                setDraft(next)
              }} /></label>
            </div>
            <div className="chips">
              {['overview', 'layers', 'quantum', 'circuit', 'runs', 'diff', 'config'].map((item) => (
                <button key={item} className={`chip ${tab === item ? 'on' : ''}`} onClick={() => setTab(item)}>{item}</button>
              ))}
            </div>
            <div className="builder-status">
              <span className={`badge ${validation?.ok ? 'done' : 'error'}`}>{validation?.ok ? 'valid' : 'invalid'}</span>
              <span className={`badge quantum-band ${validation?.resource?.band || 'low'}`}>memory {validation?.resource?.band || 'unknown'}</span>
              <span className="badge">{changes.length} pending changes</span>
              <span className="badge">{qnodes.length} quantum components</span>
            </div>
            {validation?.errors?.length > 0 && <div className="alert error">{validation.errors.join(' ')}</div>}
            {validation?.warnings?.length > 0 && <div className="alert">{validation.warnings.join(' ')}</div>}
          </section>

          {tab === 'overview' && (
            <section className="panel">
              <ModelDiagram graph={graph} title="Editable architecture" />
            </section>
          )}

          {tab === 'layers' && (
            <section className="panel table-panel">
              <table>
                <thead><tr><th>Layer</th><th>Attention</th><th>FFN</th><th>Quantum</th></tr></thead>
                <tbody>
                  {(draft.model.blocks || []).map((block, index) => (
                    <tr key={index} className={selectedLayer === index ? 'selected-row' : ''} onClick={() => setSelectedLayer(index)}>
                      <td>Block {index + 1}</td>
                      <td>
                        <select value={block.attn_type} onChange={(e) => edit(`model.blocks.${index}.attn_type`, e.target.value)}>
                          {ATTN_TYPES.map((item) => <option key={item} value={item}>{item}</option>)}
                        </select>
                      </td>
                      <td>
                        <select value={block.ffn_type} onChange={(e) => edit(`model.blocks.${index}.ffn_type`, e.target.value)}>
                          {FFN_TYPES.map((item) => <option key={item} value={item}>{item}</option>)}
                        </select>
                      </td>
                      <td>{block.attn_type.startsWith('quantum') || block.ffn_type.startsWith('quantum') ? <span className="badge quantum">quantum</span> : <span className="badge">classical</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          {tab === 'quantum' && (
            <section className="panel">
              <h3>Quantum Inspector</h3>
              <p className="panel-copy">Editing the selected layer updates its block-level quantum config. Use this for controlled hybrid model edits.</p>
              <div className="form-grid">
                <label>Selected layer
                  <select value={selectedLayer} onChange={(e) => setSelectedLayer(Number(e.target.value))}>
                    {(draft.model.blocks || []).map((_, i) => <option key={i} value={i}>Block {i + 1}</option>)}
                  </select>
                </label>
                <label>Qubits<input type="number" min="2" value={layer?.quantum?.n_qubits || draft.model.quantum.n_qubits} onChange={(e) => editNumber(`model.blocks.${selectedLayer}.quantum.n_qubits`, e.target.value)} /></label>
                <label>Depth<input type="number" min="1" value={layer?.quantum?.n_circuit_layers || draft.model.quantum.n_circuit_layers} onChange={(e) => editNumber(`model.blocks.${selectedLayer}.quantum.n_circuit_layers`, e.target.value)} /></label>
                <label>Ansatz<select value={layer?.quantum?.ansatz || draft.model.quantum.ansatz} onChange={(e) => edit(`model.blocks.${selectedLayer}.quantum.ansatz`, e.target.value)}>{ANSATZ.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
                <label>Readout<select value={layer?.quantum?.readout || draft.model.quantum.readout} onChange={(e) => edit(`model.blocks.${selectedLayer}.quantum.readout`, e.target.value)}>{READOUT.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
                <label>Backend<select value={layer?.quantum?.backend || draft.model.quantum.backend} onChange={(e) => edit(`model.blocks.${selectedLayer}.quantum.backend`, e.target.value)}>{BACKENDS.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
                <label>Shots<input value={layer?.quantum?.shots ?? ''} placeholder="analytic" onChange={(e) => edit(`model.blocks.${selectedLayer}.quantum.shots`, e.target.value ? Number(e.target.value) : null)} /></label>
                <label>Trainable
                  <select value={String(layer?.quantum?.trainable ?? true)} onChange={(e) => editBool(`model.blocks.${selectedLayer}.quantum.trainable`, e.target.value === 'true')}>
                    <option value="true">trainable</option>
                    <option value="false">frozen control</option>
                  </select>
                </label>
              </div>
            </section>
          )}

          {tab === 'circuit' && (
            <section className="panel">
              <h3>Constrained Circuit Composer</h3>
              <p className="panel-copy">This v1 composer previews and edits supported ansatz templates. Freeform arbitrary gates come after the compiler path exists.</p>
              <div className="circuit-list">
                {(qnodes[0]?.circuit?.template || []).map((gate, index) => (
                  <div key={index} className="circuit-gate">
                    <b>{gate.gate}</b>
                    <span>{gate.layer ? `layer ${gate.layer}` : 'input'} {gate.wires ? `wires ${gate.wires.join(',')}` : gate.pattern || ''}</span>
                    <span className={`badge ${gate.trainable ? 'done' : ''}`}>{gate.trainable ? 'trainable' : 'fixed'}</span>
                  </div>
                ))}
                {!qnodes.length && <p className="muted">Add a quantum attention or FFN component to inspect the circuit template.</p>}
              </div>
            </section>
          )}

          {tab === 'runs' && (
            <section className="panel">
              <h3>Run From Spec</h3>
              <div className="form-grid">
                <label>Dataset<select value={runSettings.dataset_name} onChange={(e) => setRunSettings((p) => ({ ...p, dataset_name: e.target.value }))}>{datasets.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}</select></label>
                <label>Seed<input type="number" value={runSettings.seed} onChange={(e) => setRunSettings((p) => ({ ...p, seed: Number(e.target.value) }))} /></label>
                <label>Steps<input type="number" min="1" value={runSettings.steps} onChange={(e) => setRunSettings((p) => ({ ...p, steps: Number(e.target.value) }))} /></label>
                <label>Eval every<input type="number" min="1" value={runSettings.eval_every} onChange={(e) => setRunSettings((p) => ({ ...p, eval_every: Number(e.target.value) }))} /></label>
                <label>Target<select value={runSettings.device_target} onChange={(e) => setRunSettings((p) => ({ ...p, device_target: e.target.value }))}><option value="auto">auto</option><option value="cpu">cpu</option><option value="gpu">gpu</option></select></label>
                <label>Batch<input type="number" min="1" value={runSettings.batch_size} onChange={(e) => setRunSettings((p) => ({ ...p, batch_size: Number(e.target.value) }))} /></label>
                <label>Seq len<input type="number" min="8" value={runSettings.seq_len} onChange={(e) => setRunSettings((p) => ({ ...p, seq_len: Number(e.target.value) }))} /></label>
              </div>
              <button className="primary" onClick={runSpec}>Save if needed and queue run</button>
            </section>
          )}

          {tab === 'diff' && (
            <section className="panel">
              <h3>Pending Changes</h3>
              <div className="change-list">
                {changes.map((change) => (
                  <div key={change.path} className="change-row">
                    <span className="mono">{change.path}</span>
                    <span>{String(change.before)} -&gt; {String(change.after)}</span>
                  </div>
                ))}
                {changes.length === 0 && <p className="muted">No changes from the loaded version.</p>}
              </div>
            </section>
          )}

          {tab === 'config' && (
            <section className="panel">
              <pre className="code-block">{JSON.stringify(draft, null, 2)}</pre>
            </section>
          )}
        </main>
      </div>
    </div>
  )
}
