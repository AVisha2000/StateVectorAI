import { Link } from 'react-router-dom'
import { useResearchCapabilities } from '../lib/hooks.js'
import { PageHeader } from '../lib/ui.jsx'

// The copilot + auto-ranked idea queue are the paid, human-gated core of the
// research loop (LLM/embedding provider + per-day cost budget). This surface is
// wired and capability-aware: it reads /research/capabilities and shows exactly
// what's gated, so it lights up the moment a provider is approved — without any
// spend happening here.
export default function Discover() {
  const caps = useResearchCapabilities()
  const enabled = caps.data?.paid_services_enabled === true && caps.data?.llm_provider

  return (
    <>
      <PageHeader
        title="Discover — always have something worth running"
        sub="A copilot grounded in the Library vault beside an auto-prioritized idea queue. The ranking weighs novelty × feasibility × expected-advantage × Atlas-gap alongside a decisiveness / falsification term, so null-producing and baseline-strengthening ideas rank on par."
        actions={<Link className="btn" to="/library">Open Library →</Link>}
      />

      {!enabled ? (
        <div className="notice" style={{ marginTop: 14 }}>
          The research <b>copilot</b> and the auto-ranked <b>idea queue</b> are <b>human-gated</b>: they need an
          LLM + embedding provider and a per-day cost budget (AGENTS.md gate), and the integration is backend-owned. Nothing
          here spends until a provider is approved and keys are configured. The <Link to="/library">Library</Link>'s bounded
          public arXiv scan works today without any of that.
        </div>
      ) : null}

      <div className="grid32" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="hd"><h3>Experiment idea queue</h3><span className="tag plain">ranked by promise</span></div>
          <div className="bd">
            {enabled ? (
              <p className="hint" style={{ margin: 0 }}>Idea generation is enabled — ideas will populate here from the vault.</p>
            ) : (
              <div className="state" style={{ padding: '22px 8px' }}>
                Ideas appear once the copilot is enabled. Until then, compose a hypothesis directly on the
                {' '}<Link to="/bench">Bench</Link> or design a circuit in the <Link to="/designer">Designer</Link>.
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="hd">
            <div className="ra" style={{ marginRight: 8 }}>◆</div>
            <h3>Research copilot</h3>
            <span className={`tag ${enabled ? 'good' : 'plain'}`} style={{ marginLeft: 'auto' }}>{enabled ? 'live' : 'gated'}</span>
          </div>
          <div className="bd">
            <div className="notice" style={{ marginBottom: 10 }}>
              {enabled
                ? 'Grounded in your Library vault. Ask about papers, ideas, or the next decisive test.'
                : 'The copilot is disabled until a provider + budget are approved. This panel is the interface it will use.'}
            </div>
            <div className="row" style={{ gap: 8 }}>
              <input className="mini" style={{ flex: 1 }} placeholder="Ask about papers, ideas, or the next path…" disabled={!enabled} />
              <button className="btn primary" type="button" disabled={!enabled}>Send</button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
