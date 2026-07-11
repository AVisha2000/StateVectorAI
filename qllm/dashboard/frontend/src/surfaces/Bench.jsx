import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api.js'
import { usePresets, useDatasets } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState } from '../lib/ui.jsx'
import {
  RIGOR_LEVELS,
  rigorLevel,
  buildJobPayloads,
  buildSweepPayload,
  estimateRuns,
  parsePositiveIntList,
  requiresGpuGate,
} from '../lib/benchConfig.js'

const STEPS = ['Hypothesis', 'Test plan', 'Execution', 'Verdict']

// Prefer a quantum/hybrid preset as the default candidate.
function defaultPreset(presets) {
  return presets.find((p) => p.kind === 'quantum' || p.kind === 'hybrid') || presets[0]
}

export default function Bench() {
  const { data: presets = [], isLoading: pLoading, isError: pError, error: pErr } = usePresets()
  const { data: datasets = [] } = useDatasets()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [presetId, setPresetId] = useState('')
  const [datasetName, setDatasetName] = useState('')
  const [hypothesis, setHypothesis] = useState('')
  const [rigor, setRigor] = useState('standard')
  const [steps, setSteps] = useState(2000)
  const [evalEvery, setEvalEvery] = useState(100)
  const [batchSize, setBatchSize] = useState(16)
  const [seqLen, setSeqLen] = useState(64)
  const [deviceTarget, setDeviceTarget] = useState('cpu')
  const [qubits, setQubits] = useState('4, 6, 8')
  const [depths, setDepths] = useState('2')

  // Resolve the selected preset, defaulting once presets load.
  const preset = useMemo(() => {
    if (presetId) return presets.find((p) => p.id === presetId)
    return defaultPreset(presets)
  }, [presets, presetId])

  const level = rigorLevel(rigor)
  const analogue = preset?.classical_analogue
  const seeds = level.seeds
  const runName = preset?.defaults?.run_name || preset?.id || 'bench-run'

  const config = {
    presetId: preset?.id,
    datasetName: datasetName || datasets[0]?.name,
    runName,
    seeds,
    steps,
    evalEvery,
    batchSize,
    seqLen,
    deviceTarget,
    queueComparison: level.queueComparison,
    qubits: parsePositiveIntList(qubits),
    depths: parsePositiveIntList(depths),
  }
  const estimate = estimateRuns({ ...config, rigor })
  const gpuGated = requiresGpuGate(deviceTarget)
  const canQueue = Boolean(preset && config.datasetName) && !gpuGated

  const queue = useMutation({
    mutationFn: async () => {
      if (level.sweep) return [await api.createSweep(buildSweepPayload(config))]
      const payloads = buildJobPayloads(config)
      // One POST per seed; the backend links the matched control when requested.
      return Promise.all(payloads.map((p) => api.createJob(p)))
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      navigate('/runs')
    },
  })

  if (pError) return <ErrorState error={pErr} />
  if (pLoading) return <Loading label="Loading presets…" />

  return (
    <>
      <PageHeader
        title="Bench — hypothesis to verdict, one path"
        sub="State the claim; the Bench composes a fair test around it. A quick probe and a full study are the same object — promote later, never re-enter the config."
      />

      <div className="stepper" style={{ marginTop: 16 }}>
        {STEPS.map((s, i) => (
          <span className="step-item" key={s}>
            <span className={`step ${i === 1 ? 'cur' : i < 1 ? 'done' : ''}`}>
              <span className="n">{i < 1 ? '✓' : i + 1}</span>
              <span className="t">{s}</span>
            </span>
            {i < STEPS.length - 1 ? <span className="step-line" /> : null}
          </span>
        ))}
      </div>

      {/* Hypothesis */}
      <div className="card" style={{ marginTop: 14 }}>
        <div className="bd">
          <span className="microlabel">Hypothesis</span>
          <textarea
            className="bench-hyp"
            rows={2}
            placeholder="e.g. A quantum two-stream encoder improves sample efficiency on contextual sequence tasks."
            value={hypothesis}
            onChange={(e) => setHypothesis(e.target.value)}
          />
          <p className="hint" style={{ margin: '6px 0 0' }}>
            Free text for now — Discover ideas and paper citations wire in later. The hypothesis frames the test; it is not itself evidence.
          </p>
        </div>
      </div>

      {/* Candidate + matched control */}
      <div className="grid2" style={{ marginTop: 14 }}>
        <div className="card" style={{ borderTop: '3px solid var(--q)' }}>
          <div className="hd"><h3>Candidate</h3><span className="tag q">{(preset?.kind || 'quantum').toUpperCase()}</span></div>
          <div className="bd">
            <label className="microlabel">Preset</label>
            <select className="mini block" value={preset?.id || ''} onChange={(e) => setPresetId(e.target.value)}>
              {presets.map((p) => <option key={p.id} value={p.id}>{p.label} ({p.id})</option>)}
            </select>
            <div style={{ fontWeight: 560, marginTop: 10 }}>{preset?.summary || preset?.label}</div>
            <div className="row" style={{ gap: 6, marginTop: 10 }}>
              {preset?.architecture ? <span className="tag plain">{preset.architecture}</span> : null}
              {preset?.quantum_role ? <span className="tag plain">{preset.quantum_role}</span> : null}
              <span className="quantum-band light">{preset?.cost || 'cost n/a'}</span>
            </div>
          </div>
        </div>
        <div className="card" style={{ borderTop: '3px solid var(--c)' }}>
          <div className="hd"><h3>Matched control</h3><span className="tag c">CLASSICAL</span></div>
          <div className="bd">
            {analogue ? (
              <>
                <div style={{ fontWeight: 560 }}>{analogue.label}</div>
                <p className="hint" style={{ marginTop: 8 }}>{analogue.reason}</p>
                <div style={{ marginTop: 10 }}>
                  <span className="tag warn">proposal — fairness is gated after runs, not now</span>
                </div>
              </>
            ) : (
              <p className="hint">
                This preset has no curated classical twin. A matched control must be chosen before a verdict can be drawn.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Protocol */}
      <div className="grid2" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd"><h3>Protocol</h3></div>
          <div className="bd field-grid">
            <label><span className="microlabel">Dataset</span>
              <select className="mini block" value={config.datasetName || ''} onChange={(e) => setDatasetName(e.target.value)}>
                {datasets.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
              </select>
            </label>
            <label><span className="microlabel">Device</span>
              <select className="mini block" value={deviceTarget} onChange={(e) => setDeviceTarget(e.target.value)}>
                <option value="cpu">cpu</option>
                <option value="auto">auto (may use GPU — gated)</option>
                <option value="gpu">gpu (gated)</option>
              </select>
            </label>
            <label><span className="microlabel">Steps</span>
              <input className="mini block num" type="number" value={steps} min={1} onChange={(e) => setSteps(Number(e.target.value))} />
            </label>
            <label><span className="microlabel">Eval every</span>
              <input className="mini block num" type="number" value={evalEvery} min={1} onChange={(e) => setEvalEvery(Number(e.target.value))} />
            </label>
            <label><span className="microlabel">Batch size</span>
              <input className="mini block num" type="number" value={batchSize} min={1} onChange={(e) => setBatchSize(Number(e.target.value))} />
            </label>
            <label><span className="microlabel">Seq len</span>
              <input className="mini block num" type="number" value={seqLen} min={1} onChange={(e) => setSeqLen(Number(e.target.value))} />
            </label>
            <div style={{ gridColumn: '1 / -1' }}>
              <span className="microlabel">Seeds</span>
              <div className="num" style={{ marginTop: 3 }}>{seeds.length} × {`{${seeds.join(', ')}}`}</div>
            </div>
            {level.sweep ? (
              <>
                <label><span className="microlabel">Qubit grid</span>
                  <input className="mini block" value={qubits} onChange={(e) => setQubits(e.target.value)} />
                </label>
                <label><span className="microlabel">Depth grid</span>
                  <input className="mini block" value={depths} onChange={(e) => setDepths(e.target.value)} />
                </label>
              </>
            ) : null}
          </div>
        </div>

        <div className="card">
          <div className="hd"><h3>Fairness gate</h3><span className="tag plain">evaluated after runs</span></div>
          <div className="bd" style={{ paddingTop: 8 }}>
            <p className="hint" style={{ marginTop: 0 }}>
              The gate — parameter match, identical data/steps/eval schedule, shared seed set, controls present — is checked by the
              backend from the finished runs and shown on the Verdict. Here it is only a plan, never a passed gate.
            </p>
            <div className="check"><span className="ok plan">·</span>Same dataset, steps, eval schedule across both arms</div>
            <div className="check"><span className="ok plan">·</span>Shared seed set {`{${seeds.join(', ')}}`}</div>
            <div className="check"><span className="ok plan">·</span>{level.queueComparison ? 'Matched control queued alongside the candidate' : 'No control (quick probe) — cannot produce a verdict'}</div>
          </div>
        </div>
      </div>

      {/* Rigor */}
      <div className="card" style={{ marginTop: 14 }}>
        <div className="hd"><h3>Rigor</h3><span className="hint">same experiment object at every level — promote without re-entry</span></div>
        <div className="bd">
          <div className="rigor">
            {RIGOR_LEVELS.map((r) => (
              <button key={r.key} type="button" className={`rig ${rigor === r.key ? 'on' : ''}`} onClick={() => setRigor(r.key)}>
                <b>{r.label}</b><span>{r.detail}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="footer-bar">
          <span className="hint num">
            {estimate.total} run{estimate.total === 1 ? '' : 's'} · {estimate.candidate} candidate
            {estimate.control ? ` + ${estimate.control} control` : ''} · device {deviceTarget}
            {gpuGated ? ' · GPU is human-gated' : ' · CPU, no GPU gate'}
          </span>
          <span className="spacer" />
          {queue.isError ? <span className="tag crit">{queue.error?.message || 'Queue failed'}</span> : null}
          <button
            className="btn primary"
            type="button"
            disabled={!canQueue || queue.isPending}
            onClick={() => queue.mutate()}
          >
            {queue.isPending ? 'Queueing…' : `Queue ${estimate.total} run${estimate.total === 1 ? '' : 's'} →`}
          </button>
        </div>
        {gpuGated ? (
          <p className="hint" style={{ padding: '0 16px 14px', color: 'var(--warn)' }}>
            GPU/auto targets are a human gate. Switch device to <b>cpu</b> to queue here, or get explicit approval for GPU work.
          </p>
        ) : null}
      </div>
    </>
  )
}
