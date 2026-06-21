import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { api } from '../api'

export default function Suite() {
  const { name } = useParams()
  const [data, setData] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [sortKey, setSortKey] = useState('val_ppl_mean')
  const [asc, setAsc] = useState(true)

  useEffect(() => { api.suite(name, dataset).then(setData).catch(console.error) }, [name, dataset])
  const lb = data?.leaderboard || []
  const sorted = useMemo(() => [...lb].sort((a, b) => {
    const x = a[sortKey], y = b[sortKey]
    if (x == null) return 1
    if (y == null) return -1
    return asc ? x - y : y - x
  }), [lb, sortKey, asc])

  if (!data) return <div className="loading">Loading {name}...</div>

  const metricCols = data.metric_names.map((m) => `metric_${m}`).filter((c) => lb.some((r) => r[c] != null))
  const best = sorted.find((r) => r.val_ppl_mean != null)?.variant
  const chartData = sorted.filter((r) => r.val_ppl_mean != null).map((r) => ({ variant: r.variant, ppl: r.val_ppl_mean, best: r.variant === best }))
  const click = (k) => { if (sortKey === k) setAsc(!asc); else { setSortKey(k); setAsc(true) } }

  return (
    <div>
      <h1>{name}</h1>
      <h2>{lb.length} variants - {data.datasets.length} dataset(s)</h2>

      {data.datasets.length > 1 && (
        <div className="chips">
          <span className={`chip ${!dataset ? 'on' : ''}`} onClick={() => setDataset(null)}>all</span>
          {data.datasets.map((d) => <span key={d} className={`chip ${dataset === d ? 'on' : ''}`} onClick={() => setDataset(d)}>{d}</span>)}
        </div>
      )}

      {chartData.length > 0 && (
        <div className="panel" style={{ height: 260 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 6, right: 10, bottom: 30, left: 0 }}>
              <XAxis dataKey="variant" angle={-30} textAnchor="end" height={60} tick={{ fill: '#8b949e', fontSize: 11 }} />
              <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, fontFamily: 'monospace', fontSize: 12 }} />
              <Bar dataKey="ppl" radius={[3, 3, 0, 0]}>
                {chartData.map((e, i) => <Cell key={i} fill={e.best ? '#a371f7' : '#2f81f7'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="panel" style={{ padding: 0, overflow: 'hidden' }}>
        <table>
          <thead>
            <tr>
              <th onClick={() => click('variant')}>variant</th>
              <th className="num" onClick={() => click('n_params')}>params</th>
              <th className="num" onClick={() => click('val_ppl_mean')}>val ppl</th>
              <th className="num">+/-</th>
              {metricCols.map((c) => <th key={c} className="num" onClick={() => click(c)}>{c.replace('metric_', '')}</th>)}
              <th className="num">runs</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.variant}>
                <td>{r.variant === best ? <span className="badge best">{r.variant}</span> : r.variant}</td>
                <td className="num">{r.n_params?.toLocaleString()}</td>
                <td className="num">{r.val_ppl_mean != null ? r.val_ppl_mean.toFixed(3) : '-'}</td>
                <td className="num" style={{ color: '#8b949e' }}>{r.val_ppl_std ? r.val_ppl_std.toFixed(3) : ''}</td>
                {metricCols.map((c) => <td key={c} className="num">{r[c] != null ? r[c].toFixed(3) : '-'}</td>)}
                <td className="num">{r.n_runs}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="pill">click a column header to sort - queue new lab runs from Run and inspect them from Jobs</p>
    </div>
  )
}
