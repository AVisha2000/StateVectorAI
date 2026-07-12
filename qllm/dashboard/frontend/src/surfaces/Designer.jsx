import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { api } from '../api.js'
import { PageHeader } from '../lib/ui.jsx'
import CircuitSvg from '../components/CircuitSvg.jsx'
import {
  ansatzCircuit, ANSATZE, BACKENDS, READOUTS,
  gateCount, paramCount, entanglingCount, circuitDepth,
  toBenchSpec, toQuantumOverrides,
} from '../lib/circuitModel.js'

const ZOOM = ['Machine learning', 'LLMs', 'Attention', 'Encoder', 'quantum-encoder']

export default function Designer() {
  const navigate = useNavigate()
  const [mode, setMode] = useState('quantum') // quantum | classical
  const [ansatz, setAnsatz] = useState('hardware_efficient')
  const [nQubits, setNQubits] = useState(4)
  const [depth, setDepth] = useState(2)
  const [backend, setBackend] = useState('pennylane')
  const [readout, setReadout] = useState('zz')

  const circuit = useMemo(() => ansatzCircuit(ansatz, nQubits, depth), [ansatz, nQubits, depth])
  const spec = useMemo(() => toBenchSpec(circuit, { backend, readout }), [circuit, backend, readout])

  // Proposed /designer/circuit round-trip validates the spec against registry.py.
  const validate = useMutation({ mutationFn: () => api.designerCircuit(spec) })

  const sendToBench = () => navigate('/bench', { state: { designer: { ansatz, backend, readout, overrides: toQuantumOverrides(circuit) } } })

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
                      {ANSATZE.map((a) => <option key={a} value={a}>{a}</option>)}
                    </select>
                  </label>
                  <label className="row" style={{ gap: 6 }}><span className="microlabel">Qubits</span>
                    <input className="mini num" type="number" min={1} max={12} value={nQubits}
                      onChange={(e) => setNQubits(Math.min(12, Math.max(1, Number(e.target.value))))} style={{ width: 60 }} />
                  </label>
                  <label className="row" style={{ gap: 6 }}><span className="microlabel">Depth</span>
                    <input className="mini num" type="number" min={1} max={8} value={depth}
                      onChange={(e) => setDepth(Math.min(8, Math.max(1, Number(e.target.value))))} style={{ width: 60 }} />
                  </label>
                  <label className="row" style={{ gap: 6 }}><span className="microlabel">Backend</span>
                    <select className="mini" value={backend} onChange={(e) => setBackend(e.target.value)}>
                      {BACKENDS.map((b) => <option key={b} value={b}>{b}</option>)}
                    </select>
                  </label>
                  <label className="row" style={{ gap: 6 }}><span className="microlabel">Readout</span>
                    <select className="mini" value={readout} onChange={(e) => setReadout(e.target.value)}>
                      {READOUTS.map((r) => <option key={r} value={r}>{r}</option>)}
                    </select>
                  </label>
                </div>
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
            <div className="hd"><h3>Properties</h3></div>
            <div className="bd">
              <div className="metric-row"><span>Qubits</span><b>{circuit.n_qubits}</b></div>
              <div className="metric-row"><span>Layers (depth)</span><b>{circuit.depth}</b></div>
              <div className="metric-row"><span>Circuit depth</span><b>{circuitDepth(circuit)}</b></div>
              <div className="metric-row"><span>Gates</span><b>{gateCount(circuit)}</b></div>
              <div className="metric-row"><span>Trainable params</span><b>{paramCount(circuit)}</b></div>
              <div className="metric-row"><span>Entangling gates</span><b>{entanglingCount(circuit)}</b></div>
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
            <div className="hd"><h3>Round-trip</h3><span className="tag plain">proposed</span></div>
            <div className="bd">
              <button className="btn sm" type="button" disabled={validate.isPending} onClick={() => validate.mutate()}>
                {validate.isPending ? 'Validating…' : 'Validate against registry'}
              </button>
              <p className="hint" style={{ marginTop: 8 }}>
                {validate.isSuccess ? '✓ valid circuit spec.'
                  : validate.isError ? 'The /designer/circuit round-trip isn’t on this branch yet — Send to Bench still carries the ansatz/qubits/depth overrides.'
                  : 'Validates ansatz/backend against registry.py when the backend ships /designer/circuit.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
