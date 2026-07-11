import { useDatasets } from '../lib/hooks.js'
import { PageHeader, Loading, ErrorState } from '../lib/ui.jsx'

function bytes(n) {
  if (!Number.isFinite(n)) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

export default function Datasets() {
  const { data, isLoading, isError, error } = useDatasets()
  const list = Array.isArray(data) ? data : data?.datasets ?? []

  return (
    <>
      <PageHeader
        title="Datasets"
        sub="Built-in synthetic quantum-native tasks and imported text corpora."
      />

      {isError ? (
        <ErrorState error={error} />
      ) : isLoading ? (
        <Loading label="Loading datasets…" />
      ) : (
        <div className="card scroll-x" style={{ marginTop: 16 }}>
          <table className="data">
            <thead>
              <tr>
                <th>Dataset</th><th>Source</th><th>Kind</th><th>Split</th>
                <th className="right-td">Rows</th><th className="right-td">Size</th>
              </tr>
            </thead>
            <tbody>
              {list.map((d) => (
                <tr key={d.name}>
                  <td className="mono">{d.name}</td>
                  <td>{d.source || '—'}</td>
                  <td><span className="tag plain">{d.source_type || 'built-in'}</span></td>
                  <td>{d.split || '—'}</td>
                  <td className="right-td num">{Number.isFinite(d.n_rows) ? d.n_rows : '—'}</td>
                  <td className="right-td num">{bytes(d.n_bytes)}</td>
                </tr>
              ))}
              {list.length === 0 && (
                <tr><td colSpan="6" className="hint" style={{ padding: '16px 12px' }}>
                  No datasets registered yet.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}
