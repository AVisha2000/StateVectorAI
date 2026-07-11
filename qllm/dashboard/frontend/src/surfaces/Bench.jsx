import { PageHeader, Scaffold } from '../lib/ui.jsx'

export default function Bench() {
  return (
    <>
      <PageHeader
        title="Bench"
        sub="Hypothesis to verdict, one path. A quick probe and a full study are the same object — promote without re-entry."
      />
      <Scaffold
        phase="Phase 1 — foundation (next increment)"
        blurb="The Bench composes a fair test around a stated hypothesis, pairing a candidate with an auto-matched control. It replaces the old Launch, Comparison, and Studies flows."
        bullets={[
          'Hypothesis statement (may cite a paper or a Discover idea)',
          'Candidate + auto-matched control (a proposal, not a passed fairness gate)',
          'Protocol: dataset, steps, seeds, extra controls',
          'Rigor selector: quick probe / standard pair / full study',
          'Fairness gate referencing the Pareto frontiers and baseline ladder',
        ]}
      />
    </>
  )
}
