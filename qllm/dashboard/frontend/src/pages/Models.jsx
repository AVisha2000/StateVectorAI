import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import ModelDiagram from '../components/ModelDiagram'
import { changeArchitecture, changeQuantumBackend, cloneConfig, ensureBlocks } from '../modelConfig'

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

const clone = cloneConfig

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

function applyRange(config, start, end, transform) {
  const next = clone(config)
  const lo = Math.max(0, Math.min(Number(start), Number(end)))
  const hi = Math.min((next.model.blocks || []).length - 1, Math.max(Number(start), Number(end)))
  for (let index = lo; index <= hi; index += 1) {
    const block = next.model.blocks[index]
    next.model.blocks[index] = transform(clone(block), index)
  }
  return next
}

export default function Models() {
  const [choices, setChoices] = useState(null)
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
  const [rangeStart, setRangeStart] = useState(0)
  const [rangeEnd, setRangeEnd] = useState(0)
  const [bulkAttnType, setBulkAttnType] = useState('classical')
  const [bulkFfnType, setBulkFfnType] = useState('classical')
  const [bulkTrainable, setBulkTrainable] = useState('true')
  const [validation, setValidation] = useState(null)
  const [runSettings, setRunSettings] = useState({ dataset_name: 'default-text', seed: 0, steps: 50, eval_every: 10, device_target: 'auto', batch_size: 16, seq_len: 64 })
  const [queueAnalogue, setQueueAnalogue] = useState(false)
  const [queued, setQueued] = useState(null)
  const [error, setError] = useState('')

  const refreshSpecs = () => api.modelSpecs().then(setSpecs).catch((e) => setError(e.message))

  useEffect(() => {
    api.configChoices().then(setChoices).catch((e) => setError(e.message))
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
  const activeQuantum = layer?.quantum || draft?.model?.quantum || {}
  const analogue = validation?.classical_analogue
  const blockCount = draft?.model?.blocks?.length || 0

  useEffect(() => {
    setQueueAnalogue(Boolean(analogue))
  }, [analogue?.reason])

  useEffect(() => {
    const last = Math.max(blockCount - 1, 0)
    setSelectedLayer((prev) => Math.min(prev, last))
    setRangeStart((prev) => Math.min(prev, last))
    setRangeEnd((prev) => Math.min(prev, last))
  }, [blockCount])

  function loadPreset(preset) {
    const config = ensureBlocks(flatToNested(preset.config))
    setSource({ type: 'preset', id: preset.id, label: preset.label })
    setSavedSpec(null)
    setDraft(config)
    setBaseDraft(clone(config))
    setName(`${preset.label} editable`)
    setNotes('')
    setSelectedLayer(0)
    setRangeStart(0)
    setRangeEnd(Math.max((config.model.blocks?.length || 0) - 1, 0))
    setTab('overview')
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
    setRangeStart(0)
    setRangeEnd(Math.max((config.model.blocks?.length || 0) - 1, 0))
    setTab('overview')
    setQueued(null)
  }

  function edit(path, value) {
    setDraft((prev) => changeAt(prev, path, value))
  }

  function editQuantum(field, value) {
    setDraft((previous) => {
      const next = clone(previous)
      const block = next.model.blocks?.[selectedLayer]
      if (block) {
        block.quantum = {
          ...(block.quantum || next.model.quantum || {}),
          [field]: value,
        }
      } else {
        next.model.quantum = {
          ...(next.model.quantum || {}),
          [field]: value,
        }
      }
      return next
    })
  }

  function editQuantumBackend(backend) {
    setDraft((previous) => {
      const next = clone(previous)
      const block = next.model.blocks?.[selectedLayer]
      if (block) {
        block.quantum = changeQuantumBackend(
          block.quantum || next.model.quantum || {},
          backend,
        )
      } else {
        next.model.quantum = changeQuantumBackend(next.model.quantum || {}, backend)
      }
      return next
    })
  }

  function applyBulkSwap() {
    setDraft((prev) => applyRange(prev, rangeStart, rangeEnd, (block) => ({
      ...block,
      attn_type: bulkAttnType,
      ffn_type: bulkFfnType,
    })))
  }

  function applyBulkQuantumSettings() {
    setDraft((prev) => applyRange(prev, rangeStart, rangeEnd, (block) => ({
      ...block,
      quantum: {
        ...(block.quantum || clone(prev.model.quantum)),
        trainable: bulkTrainable === 'true',
      },
    })))
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
      const job = await api.runModelSpec(spec.id, {
        ...runSettings,
        run_name: `${spec.name}-run`,
        queue_classical_comparison: Boolean(queueAnalogue && analogue),
      })
      setQueued(job)
      await refreshSpecs()
    } catch (e) {
      setError(e.message)
    }
  }

  if (!draft || !choices) {
    return <div className="loading">{error || 'Loading model builder...'}</div>
  }

  const attnTypes = choices.attention
  const archTypes = choices.architecture
  const ffnTypes = choices.feed_forward
  const ansatzTypes = choices.circuit_ansatz
  const readoutTypes = choices.readout
  const backendTypes = choices.backend

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
      {queued && (
        <div className="alert good">
          Queued <Link to={`/jobs/${queued.id}`}>job #{queued.id}: {queued.run_name}</Link>
          {queued.comparison_job && (
            <> with classical analogue <Link to={`/jobs/${queued.comparison_job.id}`}>#{queued.comparison_job.id}</Link></>
          )}
        </div>
      )}

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
                <select value={draft.model.arch} onChange={(e) => {
                  const arch = e.target.value
                  setDraft((current) => changeArchitecture(current, arch, {
                    quantumArchitectures: choices.quantum_architecture,
                    quantumDefault: choices.quantum_default,
                  }))
                  if (arch !== 'transformer' && tab === 'layers') setTab('overview')
                }}>
                  {archTypes.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
              </label>
              <label>Blocks<input type="number" min="1" disabled={draft.model.arch !== 'transformer'} value={draft.model.n_blocks} onChange={(e) => {
                if (draft.model.arch !== 'transformer') return
                const count = Number(e.target.value)
                const next = ensureBlocks(changeAt(draft, 'model.n_blocks', count))
                next.model.blocks = Array.from({ length: count }, (_, i) => next.model.blocks[i] || { attn_type: next.model.attn_type, ffn_type: next.model.ffn_type, quantum: clone(next.model.quantum) })
                setDraft(next)
              }} /></label>
            </div>
            <div className="chips">
              {['overview', 'layers', 'quantum', 'circuit', 'runs', 'diff', 'config'].map((item) => (
                <button key={item} disabled={item === 'layers' && draft.model.arch !== 'transformer'} className={`chip ${tab === item ? 'on' : ''}`} onClick={() => setTab(item)}>{item}</button>
              ))}
            </div>
            <div className="builder-status">
              <span className={`badge ${validation?.ok ? 'done' : 'error'}`}>{validation?.ok ? 'valid' : 'invalid'}</span>
              <span className={`badge quantum-band ${validation?.resource?.band || 'low'}`}>memory {validation?.resource?.band || 'unknown'}</span>
              <span className="badge">{changes.length} pending changes</span>
              <span className="badge">{validation?.layer_summary?.quantum_layers ?? qnodes.length} quantum layers</span>
              <span className={`badge ${analogue ? 'done' : ''}`}>{analogue ? 'analogue available' : 'no analogue needed'}</span>
            </div>
            {validation?.errors?.length > 0 && <div className="alert error">{validation.errors.join(' ')}</div>}
            {validation?.warnings?.length > 0 && <div className="alert">{validation.warnings.join(' ')}</div>}
          </section>

          <section className="panel">
            <div className="workspace-header">
              <div>
                <h3>Review Before Save/Run</h3>
                <p className="panel-copy">Use this step to sanity-check layer coverage, resource cost, and whether a matched analogue exists before queueing experiments.</p>
              </div>
              <span className={`badge ${validation?.fairness_review?.analogue_available ? 'done' : 'error'}`}>
                {validation?.fairness_review?.claim_readiness || 'review'}
              </span>
            </div>
            <div className="stat-row">
              <div className="metric-card">
                <div className="metric-label">Layers</div>
                <div className="metric-value">{validation?.layer_summary?.count ?? blockCount}</div>
                <div className="muted">{validation?.layer_summary?.quantum_layers ?? 0} quantum / {validation?.layer_summary?.frozen_quantum_layers ?? 0} frozen</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Resource band</div>
                <div className="metric-value">{validation?.resource_review?.band || '-'}</div>
                <div className="muted">score {validation?.resource_review?.score ? Math.round(validation.resource_review.score).toLocaleString() : '-'}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Fair comparison</div>
                <div className="metric-value">{validation?.fairness_review?.analogue_available ? 'ready' : 'missing'}</div>
                <div className="muted">{(validation?.fairness_review?.requirements || []).length} protocol checks</div>
              </div>
            </div>
            {validation?.resource_review?.summary && (
              <div className={`alert ${validation?.resource_review?.high_memory ? 'error' : ''}`}>
                {validation.resource_review.summary}
              </div>
            )}
            {validation?.fairness_review?.summary && (
              <div className={`alert ${validation?.fairness_review?.analogue_available ? '' : 'error'}`}>
                {validation.fairness_review.summary}
              </div>
            )}
            {(validation?.fairness_review?.requirements || []).length > 0 && (
              <div className="chips">
                {validation.fairness_review.requirements.map((item) => (
                  <span className="badge" key={item}>{item.replaceAll('_', ' ')}</span>
                ))}
              </div>
            )}
          </section>

          {tab === 'overview' && (
            <section className="panel">
              <ModelDiagram graph={graph} title="Editable architecture" />
            </section>
          )}

          {tab === 'layers' && (
            <section className="panel table-panel">
              <div className="workspace-header">
                <div>
                  <h3>Layer Range Editor</h3>
                  <p className="panel-copy">Select a contiguous layer range and apply controlled component swaps before fine-tuning a specific block.</p>
                </div>
              </div>
              <div className="form-grid">
                <label>Range start
                  <select value={rangeStart} onChange={(e) => setRangeStart(Number(e.target.value))}>
                    {(draft.model.blocks || []).map((_, i) => <option key={i} value={i}>Block {i + 1}</option>)}
                  </select>
                </label>
                <label>Range end
                  <select value={rangeEnd} onChange={(e) => setRangeEnd(Number(e.target.value))}>
                    {(draft.model.blocks || []).map((_, i) => <option key={i} value={i}>Block {i + 1}</option>)}
                  </select>
                </label>
                <label>Bulk attention
                  <select value={bulkAttnType} onChange={(e) => setBulkAttnType(e.target.value)}>
                    {attnTypes.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                </label>
                <label>Bulk FFN
                  <select value={bulkFfnType} onChange={(e) => setBulkFfnType(e.target.value)}>
                    {ffnTypes.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                </label>
              </div>
              <div className="header-actions">
                <button className="small" type="button" onClick={applyBulkSwap}>Apply component swap to range</button>
                <button className="small" type="button" onClick={() => { setRangeStart(0); setRangeEnd(Math.max(blockCount - 1, 0)) }}>Select all layers</button>
              </div>
              <table>
                <thead><tr><th>Layer</th><th>Attention</th><th>FFN</th><th>Quantum</th></tr></thead>
                <tbody>
                  {(draft.model.blocks || []).map((block, index) => (
                    <tr key={index} className={selectedLayer === index ? 'selected-row' : ''} onClick={() => setSelectedLayer(index)}>
                      <td>Block {index + 1}</td>
                      <td>
                        <select value={block.attn_type} onChange={(e) => edit(`model.blocks.${index}.attn_type`, e.target.value)}>
                          {attnTypes.map((item) => <option key={item} value={item}>{item}</option>)}
                        </select>
                      </td>
                      <td>
                        <select value={block.ffn_type} onChange={(e) => edit(`model.blocks.${index}.ffn_type`, e.target.value)}>
                          {ffnTypes.map((item) => <option key={item} value={item}>{item}</option>)}
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
              <p className="panel-copy">{blockCount > 0 ? 'Editing the selected layer updates its block-level quantum config. Use this for controlled hybrid model edits.' : 'Editing the architecture-level quantum configuration for this recurrent or encoder model.'}</p>
              <div className="form-grid">
                {blockCount > 0 && <>
                  <label>Selected layer
                    <select value={selectedLayer} onChange={(e) => setSelectedLayer(Number(e.target.value))}>
                      {(draft.model.blocks || []).map((_, i) => <option key={i} value={i}>Block {i + 1}</option>)}
                    </select>
                  </label>
                  <label>Range trainable
                    <select value={bulkTrainable} onChange={(e) => setBulkTrainable(e.target.value)}>
                      <option value="true">trainable</option>
                      <option value="false">frozen control</option>
                    </select>
                  </label>
                  <label className="check-row">
                    <input type="checkbox" checked={rangeStart === 0 && rangeEnd === Math.max(blockCount - 1, 0)} readOnly />
                    Layer range currently covers {Math.abs(rangeEnd - rangeStart) + 1} block(s)
                  </label>
                  <div className="header-actions">
                    <button className="small" type="button" onClick={applyBulkQuantumSettings}>Apply trainable/frozen to range</button>
                  </div>
                </>}
                <label>Qubits<input type="number" min="2" value={activeQuantum.n_qubits ?? ''} onChange={(e) => editQuantum('n_qubits', Number(e.target.value))} /></label>
                <label>Depth<input type="number" min="1" value={activeQuantum.n_circuit_layers ?? ''} onChange={(e) => editQuantum('n_circuit_layers', Number(e.target.value))} /></label>
                <label>Ansatz<select value={activeQuantum.ansatz || ansatzTypes[0]} onChange={(e) => editQuantum('ansatz', e.target.value)}>{ansatzTypes.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
                <label>Readout<select value={activeQuantum.readout || readoutTypes[0]} onChange={(e) => editQuantum('readout', e.target.value)}>{readoutTypes.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
                <label>Backend<select value={activeQuantum.backend || backendTypes[0]} onChange={(e) => editQuantumBackend(e.target.value)}>{backendTypes.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
                <label>Shots<input value={activeQuantum.shots ?? ''} placeholder="analytic" onChange={(e) => editQuantum('shots', e.target.value ? Number(e.target.value) : null)} /></label>
                {activeQuantum.backend === 'tensorcircuit_mps' && (
                  <label>Maximum bond dimension<input type="number" min="1" value={activeQuantum.mps_max_bond_dimension ?? ''} onChange={(e) => editQuantum('mps_max_bond_dimension', Number(e.target.value))} /></label>
                )}
                <label>Trainable
                  <select value={String(activeQuantum.trainable ?? true)} onChange={(e) => editQuantum('trainable', e.target.value === 'true')}>
                    <option value="true">trainable</option>
                    <option value="false">frozen control</option>
                  </select>
                </label>
              </div>
              {activeQuantum.backend === 'tensorcircuit_mps' && (
                <div className="alert">
                  MPS is approximate and supports a fixed bond limit only. Error-threshold rank selection is incompatible with QLLM's JIT/vmap path; realized error, convergence, and peak memory remain unmeasured.
                </div>
              )}
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
              {analogue && (
                <div className="analogue-panel">
                  <div className="comparison-grid">
                    <div>
                      <div className="pill">automatic classical analogue</div>
                      <h3>{analogue.label}</h3>
                      <p className="panel-copy">{analogue.reason}</p>
                    </div>
                    <div>
                      <div className="pill">fairness checks</div>
                      <div className="chips">
                        {(analogue.fairness_requirements || []).map((item) => (
                          <span className="badge" key={item}>{item.replaceAll('_', ' ')}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={queueAnalogue}
                      onChange={(e) => setQueueAnalogue(e.target.checked)}
                    />
                    Run with generated classical analogue
                  </label>
                  {!queueAnalogue && (
                    <div className="alert">
                      This custom quantum/hybrid run will need a matched analogue before it can support an advantage claim.
                    </div>
                  )}
                </div>
              )}
              <div className="form-grid">
                <label>Dataset<select value={runSettings.dataset_name} onChange={(e) => setRunSettings((p) => ({ ...p, dataset_name: e.target.value }))}>{datasets.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}</select></label>
                <label>Seed<input type="number" value={runSettings.seed} onChange={(e) => setRunSettings((p) => ({ ...p, seed: Number(e.target.value) }))} /></label>
                <label>Steps<input type="number" min="1" value={runSettings.steps} onChange={(e) => setRunSettings((p) => ({ ...p, steps: Number(e.target.value) }))} /></label>
                <label>Eval every<input type="number" min="1" value={runSettings.eval_every} onChange={(e) => setRunSettings((p) => ({ ...p, eval_every: Number(e.target.value) }))} /></label>
                <label>Target<select value={runSettings.device_target} onChange={(e) => setRunSettings((p) => ({ ...p, device_target: e.target.value }))}><option value="auto">auto</option><option value="cpu">cpu</option><option value="gpu">gpu</option></select></label>
                <label>Batch<input type="number" min="1" value={runSettings.batch_size} onChange={(e) => setRunSettings((p) => ({ ...p, batch_size: Number(e.target.value) }))} /></label>
                <label>Seq len<input type="number" min="8" value={runSettings.seq_len} onChange={(e) => setRunSettings((p) => ({ ...p, seq_len: Number(e.target.value) }))} /></label>
              </div>
              <button className="primary" onClick={runSpec}>
                {queueAnalogue && analogue ? 'Save and queue run + analogue' : 'Save if needed and queue run'}
              </button>
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
