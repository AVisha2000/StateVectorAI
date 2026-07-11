import { PageHeader, Scaffold } from '../lib/ui.jsx'

export default function Verdicts() {
  return (
    <>
      <PageHeader
        title="Verdicts"
        sub="Advantage adjudication bound to the claim ladder. Diagnostics are labeled as diagnostics — never advantage — and promotion is human-gated."
      />
      <Scaffold
        phase="Phase 2 — diagnostics & verdicts depth"
        blurb="A single home for what we have learned, replacing the three overlapping results systems (ResultsHub, Suites, ResearchResults)."
        bullets={[
          'Seed-band perplexity curve for the candidate vs. its twin',
          'Per-dimension scorecard bound to the claim ladder (no composite score)',
          'Diagnostics (expressibility, entanglement, params) labeled as diagnostics',
          'Fairness & controls panel; auto-generated caveats',
          'Human-gated promotion up the claim ladder',
        ]}
      />
    </>
  )
}
