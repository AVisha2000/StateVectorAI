// Pure shaping of a multi-seed study payload (GET /studies/{id}). A study tests a
// claim across seeds/cells; its evidence carries per-pair candidate−baseline
// deltas plus aggregate counts. Integrity: these are per-dimension multi-seed
// aggregates, NOT a composite advantage score, and replication (multi-seed) is
// distinct from the claim label. Framework-free for node --test.

function num(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

export function studySummary(study) {
  const e = study?.evidence || {}
  return {
    label: e.label ?? null,
    reason: e.reason ?? null,
    fairPairs: num(e.fair_pairs) ?? 0,
    completePairs: num(e.complete_pairs) ?? 0,
    wins: num(e.wins) ?? 0,
    meanDelta: num(e.mean_delta_val_ppl),
    stdDelta: num(e.std_delta_val_ppl),
    rerunRequiredPairs: num(e.rerun_required_pairs) ?? 0,
  }
}

// Per-pair candidate−baseline val_ppl deltas for the multi-seed strip. Only fair,
// non-rerun pairs are plotted; each gets a 1-based index. Negative delta = the
// candidate reached lower perplexity than its matched control on that pair.
export function deltaPairs(study) {
  const cmps = Array.isArray(study?.evidence?.comparisons) ? study.evidence.comparisons : []
  return cmps
    .filter((c) => c && num(c.delta_val_ppl) != null && c.fair && !c.rerun_required)
    .map((c, i) => ({ i: i + 1, delta: c.delta_val_ppl, cell: c.cell ?? c.label ?? null }))
}

// Fraction of fair pairs where the candidate won (delta < 0) — a consistency
// read, reported as a count/fraction, never merged into a single score.
export function winConsistency(study) {
  const pairs = deltaPairs(study)
  if (pairs.length === 0) return { wins: 0, total: 0, fraction: null }
  const wins = pairs.filter((p) => p.delta < 0).length
  return { wins, total: pairs.length, fraction: wins / pairs.length }
}

export function studyLadder(study) {
  return Array.isArray(study?.evidence?.ladder) ? study.evidence.ladder : []
}

export function studyJobs(study) {
  return Array.isArray(study?.jobs) ? study.jobs : []
}

export function studyCaveats(study) {
  return Array.isArray(study?.interpretation_warnings) ? study.interpretation_warnings : []
}
