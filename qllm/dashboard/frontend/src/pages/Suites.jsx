import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'

export default function Suites() {
  const [suites, setSuites] = useState(null)
  useEffect(() => { api.suites().then(setSuites).catch(console.error) }, [])
  if (!suites) return <div className="loading">Loading suites...</div>
  return (
    <div>
      <h1>Experiment suites</h1>
      <h2>{suites.length} suites - {suites.reduce((a, s) => a + s.n, 0)} total runs</h2>
      <div className="grid">
        {suites.map((s) => (
          <Link key={s.suite} to={`/suite/${encodeURIComponent(s.suite)}`} className="card">
            <h3>{s.suite}</h3>
            <div className="stat"><span className="k">runs</span><span className="v">{s.n}</span></div>
            <div className="stat"><span className="k">variants</span><span className="v">{s.variants}</span></div>
            <div className="stat"><span className="k">datasets</span><span className="v">{s.datasets}</span></div>
            <div className="stat"><span className="k">metric status</span><span className="v">{s.metric_contract?.rerun_required ? 'rerun required' : (s.best_ppl != null ? `best ppl ${s.best_ppl.toFixed(3)}` : '-')}</span></div>
          </Link>
        ))}
      </div>
    </div>
  )
}
