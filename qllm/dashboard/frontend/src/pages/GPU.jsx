import { useEffect, useState } from 'react'
import { api } from '../api'

export default function GPU() {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.status().then(setStatus).catch((e) => setError(e.message))
  }, [])

  const gpu = status?.gpu

  return (
    <div>
      <h1>GPU readiness</h1>
      <h2>GPU runs are enabled only when JAX reports a CUDA/accelerator device.</h2>
      {error && <div className="alert error">{error}</div>}
      {!status && !error && <div className="loading">Checking environment...</div>}

      {gpu && (
        <>
          <div className={`panel ${gpu.ready ? 'status ok' : 'status warn'}`}>
            <h3>Current state</h3>
            <div className="stat-row">
              <div className="metric-card">
                <div className="metric-label">JAX backend</div>
                <div className="metric-value">{gpu.jax_backend || 'unknown'}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">GPU target</div>
                <div className="metric-value">{gpu.ready ? 'enabled' : 'blocked'}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Detected devices</div>
                <div className="metric-value">{gpu.jax_devices?.length || 0}</div>
              </div>
            </div>
            {!gpu.ready && (
              <p className="pill">
                JAX is not GPU-backed yet, so the Run form will reject device target gpu.
              </p>
            )}
          </div>

          <section className="panel table-panel">
            <h3>JAX devices</h3>
            <table>
              <thead><tr><th>Device</th><th>Platform</th></tr></thead>
              <tbody>
                {(gpu.jax_devices || []).map((d, i) => (
                  <tr key={`${d.id}-${i}`}><td>{d.id}</td><td>{d.platform}</td></tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="panel">
            <h3>NVIDIA driver check</h3>
            {gpu.nvidia_smi ? (
              <>
                <span className={`badge ${gpu.nvidia_smi.ok ? 'done' : 'error'}`}>
                  nvidia-smi {gpu.nvidia_smi.ok ? 'ok' : 'error'}
                </span>
                <pre className="code-block">{gpu.nvidia_smi.output}</pre>
              </>
            ) : (
              <p className="muted">nvidia-smi was not found on PATH from the portal process.</p>
            )}
          </section>

          <section className="panel docs">
            <h3>Setup guidance</h3>
            <p>
              The portal does not install CUDA from the browser. Install a CUDA-enabled
              JAX wheel in this environment, restart the portal, then come back here.
            </p>
            <pre className="code-block">{`python -m pip install -U "jax[cuda13]"
python -c "import jax; print(jax.devices())"
python scripts/check_gpu.py`}</pre>
            <p>
              The official JAX install docs currently list NVIDIA GPU install via
              <code> jax[cuda13]</code>. On native Windows, JAX NVIDIA GPU support is
              not listed as supported; Windows WSL2 is the practical path to check
              if your local Windows Python still reports CPU only.
            </p>
            <p className="pill">Repo guide: GPU_SETUP.md. Official guide: https://docs.jax.dev/en/latest/installation.html</p>
          </section>
        </>
      )}
    </div>
  )
}
