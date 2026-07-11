import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'

export default function ScalingTests() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    api.scalingTests()
      .then((payload) => { setItems(payload); setError('') })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div>
      <h1>Scaling Tests</h1>
      <h2>Qubit/depth sweeps for the same model, dataset, seed, and training budget.</h2>
      {error && <div className="alert error">{error}</div>}
      {loading && <div className="loading">Loading scaling tests...</div>}

      <div className="action-grid">
        <Link className="action-card" to="/launch">
          <b>Queue scaling test</b>
          <span>Choose a quantum preset, then use the Scaling sweep panel.</span>
        </Link>
        <Link className="action-card" to="/experiments">
          <b>Monitor queue</b>
          <span>Filter by group id while the scaling jobs run.</span>
        </Link>
      </div>

      <section className="panel table-panel">
        <table>
          <thead>
            <tr>
              <th>Scaling group</th>
              <th>Preset</th>
              <th>Dataset</th>
              <th>Grid</th>
              <th>Status</th>
              <th className="num">Seed</th>
              <th className="num">Steps</th>
              <th>Target</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.group_id}>
                <td><Link to={`/scaling/${item.group_id}`}>{item.group_id.slice(0, 8)}</Link></td>
                <td>{item.preset_id}</td>
                <td>{item.dataset_name}</td>
                <td className="mono">q {item.qubits.join(', ')} / d {item.depths.join(', ')}</td>
                <td>
                  {Object.entries(item.statuses).map(([status, count]) => (
                    <span key={status} className={`badge ${status}`}>{status} {count}</span>
                  ))}
                </td>
                <td className="num">{item.seed}</td>
                <td className="num">{item.steps}</td>
                <td>{item.device_target}</td>
              </tr>
            ))}
            {!loading && items.length === 0 && (
              <tr><td colSpan="8">{error ? 'Scaling tests could not be loaded.' : 'No scaling tests yet. Queue one from New Experiment.'}</td></tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  )
}
