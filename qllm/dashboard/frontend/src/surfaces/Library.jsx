import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api } from '../api.js'
import { useResearchCapabilities } from '../lib/hooks.js'
import { PageHeader } from '../lib/ui.jsx'

const TOPICS = [
  { value: 'quant-ph', label: 'arXiv quant-ph' },
  { value: 'cs.LG', label: 'arXiv cs.LG (QML-filtered)' },
]

function authorLine(authors) {
  const list = Array.isArray(authors) ? authors : []
  if (list.length === 0) return '—'
  return list.length <= 3 ? list.join(', ') : `${list.slice(0, 3).join(', ')} +${list.length - 3}`
}

// Renders the research-service D4 boundary honestly: which providers are on, the
// cost budget, and which capabilities stay human-gated.
function CapabilitiesPanel({ caps, notReachable }) {
  const rows = [
    ['Metadata only', caps?.metadata_only, true],
    ['Full text', caps?.full_text, false],
    ['Paid services enabled', caps?.paid_services_enabled, false],
    ['LLM provider', caps?.llm_provider ?? 'none — human-gated'],
    ['Embedding provider', caps?.embedding_provider ?? 'none — human-gated'],
    ['Vector store', caps?.vector_store_provider ?? 'none'],
    ['Daily cost budget', caps?.daily_cost_budget ?? 'unset — human-gated'],
    ['Human review required', caps?.human_review_required, true],
    ['Claim/evidence classification', caps?.claim_evidence_classification, false],
  ]
  return (
    <div className="card">
      <div className="hd"><h3>Research service</h3><span className={`tag ${notReachable ? 'plain' : 'good'}`}>{notReachable ? 'not reachable' : 'D4 boundary'}</span></div>
      <div className="bd">
        {notReachable ? (
          <p className="hint" style={{ margin: '0 0 10px' }}>
            The research service isn't reachable on this branch yet. The panel below shows the D4 gate it enforces once live —
            only bounded public metadata; no full text, no paid LLM/embedding/vector store, no claim classification, until a
            provider and per-day budget are <b>human-approved</b>.
          </p>
        ) : null}
        <div className="atlas-kv">
          {rows.map(([label, value, wantTrue]) => (
            <div key={label}>
              <span className="microlabel">{label}</span>
              <div>
                {value === true || value === false ? (
                  <span className={`tag ${value === wantTrue ? 'good' : value ? 'warn' : 'plain'}`}>{value ? 'yes' : 'no'}</span>
                ) : value == null ? (
                  <span className="hint">—</span>
                ) : (
                  <span>{String(value)}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function Library() {
  const caps = useResearchCapabilities()
  const [topic, setTopic] = useState('quant-ph')
  const [maxResults, setMaxResults] = useState(10)
  const [scan, setScan] = useState(null)

  const scanMut = useMutation({
    mutationFn: () => api.arxivScan({ topic, max_results: Number(maxResults) }),
    onSuccess: (data) => setScan(data),
  })

  const papers = scan?.papers || []
  // Show the D4 explainer whenever we don't have live capabilities — a clean 404
  // (endpoint not on this branch) or any other fetch failure both qualify.
  const notReachable = !caps.data && !caps.isLoading

  return (
    <>
      <PageHeader
        title="Library — research archive & knowledge vault"
        sub="Everything the lab reads. Auto-scanned public metadata from the field; the paper vault, synthesis, and copilot are human-gated on a provider + cost budget."
      />

      <div className="grid32" style={{ marginTop: 16 }}>
        <div className="card">
          <div className="hd"><h3>Scan arXiv</h3><span className="hint">bounded public metadata · 1–25/scan · 50/day</span></div>
          <div className="bd">
            <div className="row" style={{ gap: 10, flexWrap: 'wrap' }}>
              <label className="row" style={{ gap: 6 }}>
                <span className="microlabel">Topic</span>
                <select className="mini" value={topic} onChange={(e) => setTopic(e.target.value)}>
                  {TOPICS.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
                </select>
              </label>
              <label className="row" style={{ gap: 6 }}>
                <span className="microlabel">Max</span>
                <input className="mini num" type="number" min={1} max={25} value={maxResults}
                  onChange={(e) => setMaxResults(Math.min(25, Math.max(1, Number(e.target.value))))} style={{ width: 64 }} />
              </label>
              <button className="btn primary" type="button" disabled={scanMut.isPending} onClick={() => scanMut.mutate()}>
                {scanMut.isPending ? 'Scanning…' : 'Scan →'}
              </button>
            </div>
            {scan ? (
              <p className="hint" style={{ marginTop: 10 }}>
                Quota: {scan.quota_used} used · {scan.quota_remaining} remaining{scan.quota_limit ? ` / ${scan.quota_limit}` : ''} today.
              </p>
            ) : null}
            {scanMut.isError ? (
              <p className="hint" style={{ marginTop: 10, color: 'var(--warn)' }}>
                {scanMut.error?.message || 'Scan failed'} — the research service may not be running on this branch yet.
              </p>
            ) : null}
          </div>
        </div>

        <CapabilitiesPanel caps={caps.data} notReachable={notReachable} />
      </div>

      <div className="card scroll-x" style={{ marginTop: 14 }}>
        <table className="data">
          <thead>
            <tr><th>Paper</th><th>Authors</th><th>Categories</th><th>Published</th><th className="right-td" /></tr>
          </thead>
          <tbody>
            {papers.map((p) => (
              <tr key={p.arxiv_id}>
                <td><b>{p.title}</b><div className="hint">{p.arxiv_id}{p.version ? ` v${p.version}` : ''}</div></td>
                <td>{authorLine(p.authors)}</td>
                <td>{(p.categories || []).slice(0, 3).map((c) => <span key={c} className="tag plain" style={{ marginRight: 4 }}>{c}</span>)}</td>
                <td className="mono">{(p.published || '').slice(0, 10)}</td>
                <td className="right-td">{p.abs_url ? <a className="btn sm" href={p.abs_url} target="_blank" rel="noreferrer">arXiv →</a> : null}</td>
              </tr>
            ))}
            {papers.length === 0 ? (
              <tr><td colSpan="5" className="hint" style={{ padding: '16px 12px' }}>
                No papers yet — run a scan above. Synthesis into the knowledge vault and the copilot are human-gated on a provider + budget.
              </td></tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </>
  )
}
