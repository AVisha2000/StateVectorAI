import { useEffect, useState } from 'react'
import { api } from '../api'

function Item({ label, ok, hint }) {
  return (
    <div className="stat">
      <span className="k">{label}</span>
      <span className="v">{ok ? 'ready' : `missing${hint ? ` - ${hint}` : ''}`}</span>
    </div>
  )
}

export default function StatusPanel() {
  const [status, setStatus] = useState(null)
  useEffect(() => { api.status().then(setStatus).catch(console.error) }, [])
  if (!status) return null
  return (
    <div className={`panel status ${status.ok ? 'ok' : 'warn'}`}>
      <h3>Environment status</h3>
      <Item label="Training stack" ok={status.training.ok} hint={status.training.install} />
      <Item label="Hugging Face import" ok={status.huggingface.ok} hint={status.huggingface.install} />
      <Item label="Built frontend" ok={status.frontend.ok} hint={status.frontend.build} />
      <Item label="JAX GPU" ok={status.gpu?.ready} hint={status.gpu?.jax_backend || 'cpu-only'} />
      {!status.frontend.node && <p className="pill">Node.js is not on PATH, so the React UI cannot be rebuilt yet.</p>}
      <p className="pill">Python: {status.python}</p>
    </div>
  )
}
