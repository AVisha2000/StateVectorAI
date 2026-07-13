import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { api } from '../api.js'
import { useDesignerCapabilities, isNotYetBuilt } from '../lib/hooks.js'
import { PageHeader } from '../lib/ui.jsx'
import CircuitSvg from '../components/CircuitSvg.jsx'
import {
  ansatzCircuit, ANSATZE, BACKENDS, READOUTS,
  designerConstraints,
  gateCount, paramCount, entanglingCount, circuitDepth,
  toBenchSpec, toQuantumOverrides,
} from '../lib/circuitModel.js'

const ZOOM = ['Machine learning', 'LLMs', 'Attention', 'Encoder', 'quantum-encoder']

export default function Designer() {
  const navigate = useNavigate()
  const { data: caps } = useDesignerCapabilities()
  const [mode, setMode] = useState('quantum') // quantum | classical
  const [ansatz, setAnsatz] = useState('hardware_efficient')
  const [nQubits, setNQubits] = useState(4)
  const [depth, setDepth] = useState(2)
  const [backend, setBackend] = useState('pennylane')
  const [readout, setReadout] = useState('z')
  const [bondDim, setBondDim] = useState(32)

  // Live registry choices/bounds from GET /designer/circuit; static fallbacks
  // (mirroring registry.py) keep the surface usable against an older backend.
  const choices = caps?.choices
  const ansatze = useMemo(
    () => (choices ? [...(choices.circuit_ansatz || []), ...(choices.qrnn_only_ansatz || [])] : ANSATZE),
    [choices],
  )
  const backends = choices?.backend?.length ? choices.backend : BACKENDS
  const readouts = choices?.readout?.length ? choices.readout : READOUTS
  const qubitBounds = caps?.constraints?.n_qubits ?? { minimum: 1, maximum: 12 }
  const layerBounds = caps?.constraints?.n_circuit_layers ?? { minimum: 1, maximum: 8 }

  // Contract rules: ising is QRNN-only (architecture='qrnn', pennylane/z
  // compatibility values); tensorcircuit_mps requires an explicit bond dimension.
  const rules = designerConstraints({ ansatz, backend })
  const effBackend = rules.backendLocked ?? backend
  const effReadout = rules.readoutLocked ?? readout

  const circuit = useMemo(() => ansatzCircuit(ansatz, nQubits, depth), [ansatz, nQubits, depth])
  const spec = useMemo(
    () => toBenchSpec(circuit, { backend, readout, mpsMaxBondDimension: bondDim }),
    [circuit, backend, readout, bondDim],
  )

  // Live POST /designer/circuit round-trip: registry-backed, side-effect-free
  // validation. The response's derived values are authoritative; the client's
  // gate counts are illustrative estimates only.
  const validate = useMutation({ mutationFn: () => api.designerCircuit(spec) })
  const derived = validate.data?.derived
  const missingBondDim = rules.needsBondDim && !(bondDim >= 1)

  const sendToBench = () => navigate('/bench', {
    state: { designer: { ansatz, backend: effBackend, readout: effReadout, overrides: toQuantumOverrides(circuit) } },
  })

  return (
    <>
      <PageHeader
        title="Designer — from the field down to the gates"
        sub="Pick a quantum ansatz family and its size; the circuit renders live and round-trips to a runnable Bench experiment with a matched classical control."
        actions={
          <>
            <div className="seg" role="tablist">
              <button className={mode === 'quantum' ? 'on' : undefined} onClick={() => setMode('quantum')}>Quantum</button>
              <button className={mode === 'classical' ? 'on' : undefined} onClick={() => setMode('classical')}>Classical</button>
            </div>
            <button className="btn primary" type="button" onClick={sendToBench}>Send to Bench →</button>
          </>
        }
      />

      <div className="zoombar" style={{ marginTop: 14 }}>
        {ZOOM.map((z, i) => (
          <span key={z} className="row" style={{ gap: 6 }}>
            <span className={`zlevel ${i === ZOOM.length - 1 ? 'on' : ''}`}>{z}</span>
            {i < ZOOM.length - 1 ? <span className="zsep">›</span> : null}
          </span>
        ))}
        <span className="hint spacer">illustrative zoom · continuous with the Atlas</span>
      </div>

      <div className="grid32" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd">
            <h3>{mode === 'quantum' ? 'Circuit' : 'Classical twin'}</h3>
            <span className={`tag ${mode === 'quantum' ? 'q' : 'c'}`}>{mode.toUpperCase()}</span>
          </div>
          <div className="bd">
            {mode === 'quantum' ? (
              <>
                <div className="row" style={{ gap: 10, flexWrap: 'wrap', marginBottom: 12 }}>
                  <label className="row" style={{ gap: 6 }}><span className="microlabel">Ansatz</span>
                    <select className="mini" value={ansatz} onChange={(e) => setAnsatz(e.target.value)}>
                      {ansatze.map((a) => <option key={a} value={a}>{a}</option>)}
                    </select>
                  </label>
                  <label className="row" style={{ gap: 6 }}><span className="microlabel">Qubits</span>
                    <input className="mini num" type="number" min={qubitBounds.minimum} max={qubitBounds.maximum} value={nQubits}
                      onChange={(e) => setNQubits(Math.min(qubitBounds.maximum, Math.max(qubitBounds.minimum, Number(e.target.value))))} style={{ width: 60 }} />
                  </label>
                  <label className="row" style={{ gap: 6 }}><span className="microlabel">Depth</span>
                    <input className="mini num" type="number" min={layerBounds.minimum} max={layerBounds.maximum} value={depth}
                      onChange={(e) => setDepth(Math.min(layerBounds.maximum, Math.max(layerBounds.minimum, Number(e.target.value))))} style={{ width: 60 }} />
                  </label>
                  <label className="row" style={{ gap: 6 }}><span className="microlabel">Backend</span>
                    <select className="mini" value={effBackend} disabled={!!rules.backendLocked} onChange={(e) => setBackend(e.target.value)}>
                      {backends.map((b) => <option key={b} value={b}>{b}</option>)}
                    </select>
                  </label>
                  <label className="row" style={{ gap: 6 }}><span className="microlabel">Readout</span>
                    <select className="mini" value={effReadout} disabled={!!rules.readoutLocked} onChange={(e) => setReadout(e.target.value)}>
                      {readouts.map((r) => <option key={r} value={r}>{r}</option>)}
                    </select>
                  </label>
                  {rules.needsBondDim ? (
                    <label className="row" style={{ gap: 6 }}><span className="microlabel">Max bond dim</span>
                      <input className="mini num" type="number" min={1} value={bondDim}
                        onChange={(e) => setBondDim(Math.max(1, Number(e.target.value)))} style={{ width: 70 }} />
                    </label>
                  ) : null}
                </div>
                {rules.architecture ? (
                  <p className="hint" style={{ margin: '0 0 10px' }}>
                    <b>{ansatz}</b> is a QRNN-only family — it runs with <span className="mono">architecture=qrnn</span>;
                    backend/readout are pinned to the compatibility values (<span className="mono">pennylane</span>/<span className="mono">z</span>),
                    not execution selectors.
                  </p>
                ) : null}
                {rules.needsBondDim ? (
                  <p className="hint" style={{ margin: '0 0 10px' }}>
                    <span className="mono">tensorcircuit_mps</span> is <b>approximate</b>, bounded by the max bond
                    dimension — never silently treated as an exact statevector.
                  </p>
                ) : null}
                <CircuitSvg circuit={circuit} />
              </>
            ) : (
              <p className="hint" style={{ margin: 0 }}>
                The matched classical twin is a width-matched network that replaces the quantum block while preserving the
                surrounding model and training protocol — the fair control the Bench pairs automatically. Toggle back to
                <b> Quantum</b> to edit the circuit.
              </p>
            )}
          </div>
        </div>

        <div>
          <div className="card" style={{ marginBottom: 12 }}>
            <div className="hd"><h3>Properties</h3><span className="hint">client estimates</span></div>
            <div className="bd">
              <div className="metric-row"><span>Qubits</span><b>{circuit.n_qubits}</b></div>
              <div className="metric-row"><span>Layers (depth)</span><b>{circuit.depth}</b></div>
              <div className="metric-row"><span>Circuit depth</span><b>{circuitDepth(circuit)}</b></div>
              <div className="metric-row"><span>Gates (drawn)</span><b>{gateCount(circuit)}</b></div>
              <div className="metric-row"><span>Param gates (drawn)</span><b>{paramCount(circuit)}</b></div>
              <div className="metric-row"><span>Entangling gates (drawn)</span><b>{entanglingCount(circuit)}</b></div>
              {derived?.trainable_circuit_parameters?.status === 'derived' ? (
                <div className="metric-row" style={{ borderTop: '1px solid var(--hair)', paddingTop: 6 }}>
                  <span>Trainable params <span className="tag good sm">registry</span></span>
                  <b>{derived.trainable_circuit_parameters.value}</b>
                </div>
              ) : null}
              <p className="hint" style={{ margin: '8px 0 0' }}>
                Drawn counts describe the illustration; the registry-derived parameter shape from validation is authoritative.
              </p>
            </div>
          </div>
          <div className="card" style={{ marginBottom: 12 }}>
            <div className="hd"><h3>Matched control</h3><span className="tag c">CLASSICAL</span></div>
            <div className="bd" style={{ fontSize: 12.5, color: 'var(--ink2)' }}>
              A width-matched classical twin is auto-paired on the Bench — a proposal, not a passed fairness gate. Fairness
              (param match, identical protocol) is checked from the finished runs on the Verdict.
            </div>
          </div>
          <div className="card">
            <div className="hd"><h3>Round-trip</h3><span className="tag good">live</span></div>
            <div className="bd">
              <button className="btn sm" type="button" disabled={validate.isPending || missingBondDim} onClick={() => validate.mutate()}>
                {validate.isPending ? 'Validating…' : 'Validate against registry'}
              </button>
              {missingBondDim ? (
                <p className="hint" style={{ marginTop: 8 }}>Set a max bond dimension (≥ 1) — required for tensorcircuit_mps.</p>
              ) : validate.isSuccess ? (
                <div style={{ marginTop: 8 }}>
                  <p className="hint" style={{ margin: 0 }}>
                    ✓ valid — registry-backed, side-effect-free (no circuit, model, or device constructed).
                  </p>
                  {validate.data?.ignored_fields?.length ? (
                    <p className="hint" style={{ margin: '6px 0 0' }}>
                      Compatibility-only fields: <span className="mono">{validate.data.ignored_fields.join(', ')}</span>
                    </p>
                  ) : null}
                  {(validate.data?.warnings || []).slice(0, 3).map((w, i) => (
                    <p className="hint" key={i} style={{ margin: '6px 0 0' }}>• {w}</p>
                  ))}
                </div>
              ) : validate.isError ? (
                isNotYetBuilt(validate.error) ? (
                  <p className="hint" style={{ marginTop: 8 }}>
                    This backend build doesn’t serve <span className="mono">/designer/circuit</span> yet — Send to Bench
                    still carries the ansatz/qubits/depth overrides.
                  </p>
                ) : (
                  <p className="hint" style={{ marginTop: 8, color: 'var(--crit)' }}>
                    Rejected by the registry: {validate.error?.message || 'validation failed.'}
                  </p>
                )
              ) : (
                <p className="hint" style={{ marginTop: 8 }}>
                  Validates this spec against <span className="mono">registry.py</span> via <span className="mono">/designer/circuit</span> —
                  circuit properties are diagnostics, never evidence of advantage.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
