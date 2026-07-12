// Pure config → job-payload logic for the Bench surface. Mirrors the real POST
// /api/jobs and /api/jobs/sweep payloads ported from the legacy Launch/Studies
// pages, but organized around the redesign's "one experiment object, promote
// without re-entry" rigor model. Kept framework-free for `node --test`.

// Rigor levels share one config; promoting only changes seeds/controls/sweep.
export const RIGOR_LEVELS = Object.freeze([
  {
    key: 'quick',
    label: 'Quick probe',
    detail: '1 seed · no controls · flags interest, proves nothing',
    seeds: [42],
    queueComparison: false,
    sweep: false,
  },
  {
    key: 'standard',
    label: 'Standard pair',
    detail: '5 seeds · matched control · can produce a candidate verdict',
    seeds: [11, 23, 42, 77, 101],
    queueComparison: true,
    sweep: false,
  },
  {
    key: 'full',
    label: 'Full study',
    detail: '+ qubit × depth grid · evidence for claim-ladder promotion',
    seeds: [11, 23, 42, 77, 101],
    queueComparison: true,
    sweep: true,
  },
])

export function rigorLevel(key) {
  return RIGOR_LEVELS.find((r) => r.key === key) || RIGOR_LEVELS[0]
}

export function parseIntList(text) {
  return String(text ?? '')
    .split(/[,\s]+/)
    .map((t) => t.trim())
    .filter((t) => t !== '')
    .map(Number)
    .filter((n) => Number.isInteger(n))
}

export function parsePositiveIntList(text) {
  return parseIntList(text).filter((n) => n > 0)
}

// Build one POST /api/jobs payload per seed (quick/standard). run_name is
// suffixed per seed so runs stay distinct. quantum_overrides is only attached
// when provided and non-empty.
export function buildJobPayloads(config) {
  const {
    presetId,
    datasetName,
    runName,
    seeds = [42],
    steps,
    evalEvery,
    batchSize,
    seqLen,
    deviceTarget = 'cpu',
    queueComparison = false,
    quantumOverrides = null,
  } = config
  const overrides =
    quantumOverrides && Object.keys(quantumOverrides).length ? quantumOverrides : null
  return seeds.map((seed) => {
    const payload = {
      preset_id: presetId,
      dataset_name: datasetName,
      run_name: seeds.length > 1 ? `${runName}-s${seed}` : runName,
      seed: Number(seed),
      steps: Number(steps),
      eval_every: Number(evalEvery),
      batch_size: Number(batchSize),
      seq_len: Number(seqLen),
      device_target: deviceTarget,
      queue_classical_comparison: Boolean(queueComparison),
    }
    if (overrides) payload.quantum_overrides = overrides
    return payload
  })
}

// Build the POST /api/jobs/sweep payload for a full study (qubit × depth grid).
export function buildSweepPayload(config) {
  const {
    presetId,
    datasetName,
    runName,
    seeds = [42],
    steps,
    evalEvery,
    batchSize,
    seqLen,
    deviceTarget = 'cpu',
    queueComparison = true,
    qubits = [],
    depths = [],
  } = config
  return {
    preset_id: presetId,
    dataset_name: datasetName,
    run_name: runName,
    seed: Number(seeds[0] ?? 42),
    steps: Number(steps),
    eval_every: Number(evalEvery),
    batch_size: Number(batchSize),
    seq_len: Number(seqLen),
    device_target: deviceTarget,
    qubits,
    depths,
    queue_classical_comparison: Boolean(queueComparison),
  }
}

// Honest run-count estimate for the footer ("queue N runs"). Controls double the
// candidate count only when a matched comparison is queued.
export function estimateRuns(config) {
  const { rigor, seeds = [42], qubits = [], depths = [], queueComparison = false } = config
  let candidate
  if (rigor === 'full') {
    candidate = Math.max(qubits.length, 1) * Math.max(depths.length, 1)
  } else {
    candidate = Math.max(seeds.length, 1)
  }
  const control = queueComparison ? candidate : 0
  return { candidate, control, total: candidate + control }
}

// GPU stays a human gate. A queue is CPU-safe only when every arm targets CPU.
export function requiresGpuGate(deviceTarget) {
  return deviceTarget !== 'cpu'
}
