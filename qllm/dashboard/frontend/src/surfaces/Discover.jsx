import { PageHeader, Scaffold } from '../lib/ui.jsx'

export default function Discover() {
  return (
    <>
      <PageHeader
        title="Discover"
        sub="Your research copilot — always something worth running. A bilateral loop: the agent scans and presents; you explore and present back."
      />
      <Scaffold
        phase="Phase 4 — research loop (greenfield)"
        blurb="A dialogue grounded in the Library vault beside an auto-prioritized experiment idea queue. This subsystem is greenfield — the LLM/embedding provider, vector store, and per-day cost budget are human-gated decisions."
        bullets={[
          'Copilot dialogue grounded in the vault, with inline paper citations',
          'Idea queue ranked by novelty × feasibility × advantage × Atlas-gap + a falsification-value term',
          'Null-producing and baseline-strengthening ideas rank on par',
          'Top idea one click from the Bench; falsified ideas leave the queue',
        ]}
      />
    </>
  )
}
