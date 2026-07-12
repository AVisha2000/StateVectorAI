import { useStatus } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState } from '../lib/ui.jsx'

// Backend capability summary (docs/UI_REDESIGN_PLAN.md § Queue & Backends).
const BACKENDS = [
  { name: 'pennylane', method: 'dense statevector', qubits: '~16', device: 'cpu / gpu', status: 'ready' },
  { name: 'tensorcircuit', method: 'dense statevector', qubits: '~20', device: 'cpu / gpu', status: 'ready' },
  { name: 'tensorcircuit_mps', method: 'matrix-product state · bond 16', qubits: '~40 (low entangle.)', device: 'mps', status: 'gated' },
]

export default function System() {
  const { data: status, isLoading, isError, error } = useStatus()

  return (
    <>
      <PageHeader
        title="Queue & Backends"
        sub="Worker, quantum backends, device readiness. GPU/QPU runs stay human-gated."
      />

      {isError ? (
        <ErrorState error={error} />
      ) : isLoading ? (
        <Loading label="Loading system status…" />
      ) : (
        <>
          <div className="kpis" style={{ marginTop: 16 }}>
            <div className="kpi">
              <span className="microlabel">Worker</span>
              <div className="v" style={{ fontSize: 16 }}>{status?.worker || 'CPU · active'}</div>
              <div className="s">single in-process worker</div>
            </div>
            <div className="kpi">
              <span className="microlabel">GPU</span>
              <div className="v" style={{ fontSize: 16 }}>{status?.gpu_available ? 'available' : 'idle · gated'}</div>
              <div className="s">JAX + CUDA readiness</div>
            </div>
            <div className="kpi">
              <span className="microlabel">Running</span>
              <div className="v num">{Number.isFinite(status?.running) ? status.running : '—'}</div>
              <div className="s">jobs in flight</div>
            </div>
            <div className="kpi">
              <span className="microlabel">Queue depth</span>
              <div className="v num">{Number.isFinite(status?.queued) ? status.queued : '—'}</div>
              <div className="s">jobs waiting</div>
            </div>
            <div className="kpi">
              <span className="microlabel">Runs recorded</span>
              <div className="v num">{Number.isFinite(status?.runs) ? status.runs : '—'}</div>
              <div className="s">in the local database</div>
            </div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <div className="hd"><h3>Quantum backends</h3></div>
            <div className="bd scroll-x" style={{ padding: '4px 16px 8px' }}>
              <table className="data">
                <thead>
                  <tr><th>Backend</th><th>Method</th><th>Max qubits</th><th>Device</th><th className="right-td">Status</th></tr>
                </thead>
                <tbody>
                  {BACKENDS.map((b) => (
                    <tr key={b.name}>
                      <td className="mono">{b.name}</td>
                      <td>{b.method}</td>
                      <td className="num">{b.qubits}</td>
                      <td>{b.device}</td>
                      <td className="right-td">
                        <span className={`tag ${b.status === 'ready' ? 'good' : 'warn'}`}>{b.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  )
}
