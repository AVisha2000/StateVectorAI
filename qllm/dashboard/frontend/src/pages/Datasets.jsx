import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import StatusPanel from '../components/StatusPanel'

const TASK_DESCRIPTIONS = {
  'Language modelling': 'General next-token modeling on a text corpus. Useful as a baseline, but not enough by itself for a quantum-advantage claim.',
  'Sequence memory': 'Stress-tests long-range memory and recurrent inductive bias. Compare QRNN-style candidates against recurrent classical controls.',
  'Contextual parity': 'A structured probe where contextual information should matter more than local token statistics.',
  'Quantum-generated sequence prediction': 'Generated sequence tasks with explicit quantum structure. Best used with cautious baselines and multi-seed studies.',
  'Interference/cancellation': 'Looks for settings where cancellation structure matters and naive local predictors should struggle.',
  'Two-stream semantic conditioning': 'Tests whether the conditioning pathway adds signal beyond the classical two-stream control.',
}

function formatBytes(value) {
  if (value == null) return '-'
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`
  return `${(value / (1024 ** 2)).toFixed(1)} MiB`
}

function shortHash(value) {
  return value ? `${value.slice(0, 10)}…` : '-'
}

export default function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [explore, setExplore] = useState(null)
  const [form, setForm] = useState({
    source: 'roneneldan/TinyStories',
    split: 'train',
    text_column: 'text',
    display_name: 'tinystories-sample',
    row_limit: 1000,
    revision: '',
    char_limit: 5000000,
    byte_limit: 10000000,
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [lastImport, setLastImport] = useState(null)

  const refresh = () => api.datasets().then(setDatasets).catch((e) => setError(e.message))
  useEffect(() => {
    refresh()
    api.explore().then(setExplore).catch((e) => setError(e.message))
  }, [])

  const taskCards = useMemo(() => (
    (explore?.tasks || []).map((task) => ({
      ...task,
      description: TASK_DESCRIPTIONS[task.name] || 'Task-level result slices help separate a dataset score from a stronger scientific claim.',
    }))
  ), [explore])

  const update = (key, value) => setForm((f) => ({ ...f, [key]: value }))
  const directUrl = /^(https?|hf):\/\//i.test(form.source.trim())
  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError(''); setMessage(''); setLastImport(null)
    try {
      const ds = await api.importHfDataset({
        ...form,
        row_limit: Number(form.row_limit),
        revision: directUrl ? null : (form.revision.trim() || null),
        char_limit: Number(form.char_limit),
        byte_limit: Number(form.byte_limit),
      })
      setLastImport(ds)
      setMessage(`Imported ${ds.name} (${ds.n_rows} rows, ${ds.n_chars} chars, ${formatBytes(ds.n_bytes)})`)
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
      <h2>Import local corpora and connect them to the task probes used for cautious quantum-advantage evaluation.</h2>
      <StatusPanel />
      {error && <div className="alert error">{error}</div>}
      {message && <div className="alert good">{message}</div>}
      {lastImport?.warnings?.map((warning, index) => (
        <div className="alert" key={`${index}-${warning}`}>{warning}</div>
      ))}

      <section className="panel">
        <div className="workspace-header">
          <div>
            <h3>Task Cards</h3>
            <p className="panel-copy">Datasets are the data source. Tasks are the scientific probe. Read task views before turning a good score into a broader claim.</p>
          </div>
          <Link className="small-link" to="/explore">Open task map</Link>
        </div>
        <div className="grid">
          {taskCards.map((task) => (
            <Link key={`${task.domain_slug}-${task.slug}`} className="card" to={`/explore/task/${task.slug}?domain=${encodeURIComponent(task.domain_slug)}`}>
              <h3>{task.name}</h3>
              <div className="muted">{task.domain}</div>
              <p className="panel-copy">{task.description}</p>
              <div className="stat"><span className="k">datasets</span><span className="v">{task.datasets.length}</span></div>
              <div className="stat"><span className="k">evidence</span><span className="v">{task.runs} runs / {task.jobs} jobs</span></div>
            </Link>
          ))}
          {taskCards.length === 0 && <p className="muted">Task cards will populate once runs or jobs establish a research map.</p>}
        </div>
      </section>

      <form className="panel" onSubmit={submit}>
        <h3>Hugging Face import</h3>
        <div className="form-grid">
          <label>Dataset id or URL<input value={form.source} onChange={(e) => update('source', e.target.value)} /></label>
          <label>Split<input value={form.split} onChange={(e) => update('split', e.target.value)} /></label>
          <label>Text column<input value={form.text_column} onChange={(e) => update('text_column', e.target.value)} /></label>
          <label>Display name<input value={form.display_name} onChange={(e) => update('display_name', e.target.value)} /></label>
          <label>Row limit<input type="number" min="1" max="200000" value={form.row_limit} onChange={(e) => update('row_limit', e.target.value)} /></label>
          <label>Requested revision (HF only)<input disabled={directUrl} placeholder={directUrl ? 'Not applicable to URLs' : 'branch, tag, or commit'} value={form.revision} onChange={(e) => update('revision', e.target.value)} /></label>
          <label>Character limit<input type="number" min="1" max="50000000" value={form.char_limit} onChange={(e) => update('char_limit', e.target.value)} /></label>
          <label>UTF-8 byte limit<input type="number" min="1" max="100000000" value={form.byte_limit} onChange={(e) => update('byte_limit', e.target.value)} /></label>
        </div>
        <p className="pill">Public text datasets only. Imported corpus output is bounded, SHA-256 hashed, and written as UTF-8 under data/imported/. Streaming may still fetch remote chunks. Direct URLs do not support revisions.</p>
        <button className="primary" type="submit" disabled={busy}>{busy ? 'Importing...' : 'Import dataset'}</button>
      </form>

      <div className="panel" style={{ padding: 0, overflow: 'auto' }}>
        <table>
          <thead><tr><th>Name</th><th>Source</th><th>Split</th><th>Text column</th><th>Revision / fingerprint</th><th className="num">Rows</th><th className="num">Size</th><th>Integrity</th><th>Import status</th></tr></thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.name}>
                <td>{d.name}</td>
                <td>{d.source_type}: {d.source}</td>
                <td>{d.split || '-'}</td>
                <td>{d.text_column || '-'}</td>
                <td>
                  <div>{d.revision_applicable === false ? 'n/a' : (d.requested_revision || 'default')}</div>
                  {d.resolved_fingerprint && <div className="muted" title={d.resolved_fingerprint}>fingerprint {shortHash(d.resolved_fingerprint)}</div>}
                </td>
                <td className="num">
                  <div>{d.n_rows ?? '-'}</div>
                  {d.rows_examined != null && <div className="muted">{d.rows_examined.toLocaleString()} examined</div>}
                </td>
                <td className="num">
                  <div>{d.n_chars?.toLocaleString?.() ?? '-'} chars</div>
                  <div className="muted">{formatBytes(d.n_bytes)}</div>
                </td>
                <td><span title={d.sha256 || ''}>{shortHash(d.sha256)}</span></td>
                <td>
                  <div>{d.truncated ? `Truncated: ${d.truncation_reason}` : 'Complete'}</div>
                  {(d.warnings || []).map((warning, index) => <div className="muted" key={`${index}-${warning}`}>{warning}</div>)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
