import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import EvidenceWarnings from '../components/EvidenceWarnings'

export default function Suites() {
  const [suites, setSuites] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    api.suites()
      .then((payload) => { setSuites(payload); setError('') })
      .catch((e) => setError(e.message))
  }, [])
  if (!suites && !error) return <div className="loading">Loading suites...</div>
  if (!suites) return <div><h1>Experiment suites</h1><div className="alert error">{error}</div></div>
  return (
    <div>
      <h1>Experiment suites</h1>
      <h2>{suites.length} suites - {suites.reduce((a, s) => a + s.n, 0)} total runs</h2>
      <div className="grid">
        {suites.map((s) => (
          <Link key={s.suite} to={`/suite/${encodeURIComponent(s.suite)}`} className="card">
            <h3>{s.suite}</h3>
            <EvidenceWarnings warnings={s.interpretation_warnings} compact />
            <div className="stat"><span className="k">runs</span><span className="v">{s.n}</span></div>
            <div className="stat"><span className="k">variants</span><span className="v">{s.variants}</span></div>
            <div className="stat"><span className="k">datasets</span><span className="v">{s.datasets}</span></div>
            <div className="stat"><span className="k">metric status</span><span className="v">{s.metric_contract?.rerun_required ? 'rerun required' : (s.best_ppl != null ? `best ppl ${s.best_ppl.toFixed(3)}` : '-')}</span></div>
          </Link>
        ))}
        {suites?.length === 0 && <div className="panel muted">No experiment suites have completed yet.</div>}
      </div>
    </div>
  )
}
