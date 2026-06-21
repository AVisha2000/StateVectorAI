import { useEffect, useState } from 'react'
import { api } from '../api'
import StatusPanel from '../components/StatusPanel'

export default function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [form, setForm] = useState({
    source: 'roneneldan/TinyStories',
    split: 'train',
    text_column: 'text',
    display_name: 'tinystories-sample',
    row_limit: 1000,
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const refresh = () => api.datasets().then(setDatasets).catch((e) => setError(e.message))
  useEffect(() => { refresh() }, [])

  const update = (key, value) => setForm((f) => ({ ...f, [key]: value }))
  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError(''); setMessage('')
    try {
      const ds = await api.importHfDataset({
        ...form,
        row_limit: Number(form.row_limit),
      })
      setMessage(`Imported ${ds.name} (${ds.n_rows} rows, ${ds.n_chars} chars)`)
      refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <h1>Datasets</h1>
      <h2>Import public Hugging Face text datasets into local training corpora.</h2>
      <StatusPanel />
      {error && <div className="alert error">{error}</div>}
      {message && <div className="alert good">{message}</div>}

      <form className="panel" onSubmit={submit}>
        <h3>Hugging Face import</h3>
        <div className="form-grid">
          <label>Dataset id or URL<input value={form.source} onChange={(e) => update('source', e.target.value)} /></label>
          <label>Split<input value={form.split} onChange={(e) => update('split', e.target.value)} /></label>
          <label>Text column<input value={form.text_column} onChange={(e) => update('text_column', e.target.value)} /></label>
          <label>Display name<input value={form.display_name} onChange={(e) => update('display_name', e.target.value)} /></label>
          <label>Row limit<input type="number" min="1" max="200000" value={form.row_limit} onChange={(e) => update('row_limit', e.target.value)} /></label>
        </div>
        <p className="pill">Public text datasets only in v1. The importer writes a local .txt corpus under data/imported/.</p>
        <button className="primary" disabled={busy}>{busy ? 'Importing...' : 'Import dataset'}</button>
      </form>

      <div className="panel" style={{ padding: 0, overflow: 'hidden' }}>
        <table>
          <thead><tr><th>Name</th><th>Source</th><th>Split</th><th>Text column</th><th className="num">Rows</th><th className="num">Chars</th></tr></thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.name}>
                <td>{d.name}</td>
                <td>{d.source_type}: {d.source}</td>
                <td>{d.split || '-'}</td>
                <td>{d.text_column || '-'}</td>
                <td className="num">{d.n_rows ?? '-'}</td>
                <td className="num">{d.n_chars?.toLocaleString?.() ?? '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
