// Browser teaching model, independent of canonical QLLM research backends.
// H = (omega / 2) sigma_axis, hbar = 1, angle = omega * duration (radians).
export const QUANTUM_MODEL = 'ideal-bloch-rotations-v1'
export const MAX_PULSES = 128
export const MAX_ATTEMPTS = 12
export const INITIAL_BLOCH = Object.freeze([0, 0, 1])
export const QUANTUM_TARGETS = Object.freeze([
  { id: 'plus', name: 'Find superposition', ket: '|+⟩', vector: [1, 0, 0], hint: 'Move the mint state to the amber marker on the equator.' },
  { id: 'one', name: 'Flip the qubit', ket: '|1⟩', vector: [0, 0, -1], hint: 'Reach the south pole. Can a different axis get you there too?' },
  { id: 'phase', name: 'Explore phase', ket: '|+i⟩', vector: [0, 1, 0], hint: 'Reach the side of the equator. Compare the order of two rotations.' },
].map((target) => Object.freeze({ ...target, vector: Object.freeze(target.vector) })))
const AXES = Object.freeze({ x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] })
export const PULSE_KEYS = Object.freeze({ w: ['x', 1], s: ['x', -1], d: ['y', 1], a: ['y', -1], e: ['z', 1], q: ['z', -1] })

export function rotateBloch(state, axis, angle) {
  if (!Array.isArray(state) || state.length !== 3 || !state.every(Number.isFinite) || Math.hypot(...state) > 1 + 1e-9)
    throw new Error('The Bloch vector must be finite and inside the unit sphere.')
  if (!Object.hasOwn(AXES, axis) || !Number.isFinite(angle) || Math.abs(angle) > 2 * Math.PI)
    throw new Error('Choose X, Y or Z and a pulse angle within one full turn.')
  const [x, y, z] = state, [u, v, w] = AXES[axis]
  const c = Math.cos(angle), s = Math.sin(angle), d = u * x + v * y + w * z
  return [
    x * c + (v * z - w * y) * s + u * d * (1 - c),
    y * c + (w * x - u * z) * s + v * d * (1 - c),
    z * c + (u * y - v * x) * s + w * d * (1 - c),
  ]
}

export function simulatePulses(pulses, fraction = pulses.length) {
  if (!Array.isArray(pulses) || pulses.length > MAX_PULSES || !Number.isFinite(fraction) || fraction < 0 || fraction > pulses.length)
    throw new Error('Invalid pulse sequence or replay position.')
  let state = [...INITIAL_BLOCH]
  const points = [[...state]], endpoints = [[...state]]
  for (let i = 0; i < pulses.length; i++) {
    const { axis, angle } = pulses[i]
    // Validate every pulse, including pulses after the current replay position.
    rotateBloch(INITIAL_BLOCH, axis, angle)
    const portion = Math.max(0, Math.min(1, fraction - i))
    if (!portion) continue
    const start = state, steps = Math.max(1, Math.ceil(Math.abs(angle * portion) / (Math.PI / 48)))
    for (let j = 1; j <= steps; j++) points.push(rotateBloch(start, axis, angle * portion * j / steps))
    state = points[points.length - 1]
    endpoints.push([...state])
  }
  return { state, points, endpoints }
}

export function targetOverlap(state, target) {
  if (!Array.isArray(target) || target.length !== 3 || !target.every(Number.isFinite) || Math.abs(Math.hypot(...target) - 1) > 1e-9)
    throw new Error('The target must be a unit Bloch vector.')
  rotateBloch(state, 'z', 0)
  return Math.max(0, Math.min(1, (1 + state.reduce((sum, value, i) => sum + value * target[i], 0)) / 2))
}

export function initialQuantumVisit() {
  return { targetId: 'plus', angleDegrees: 30, pulses: [], attempts: [], note: '' }
}

export function quantumTarget(id) {
  const target = QUANTUM_TARGETS.find((item) => item.id === id)
  if (!target) throw new Error('Unknown quantum target.')
  return target
}

export function captureQuantumAttempt(visit, recordedAt) {
  if (!visit.pulses.length) throw new Error('Apply a pulse before saving an attempt.')
  if (visit.attempts.length >= MAX_ATTEMPTS) throw new Error('This visit has 12 saved attempts. Download your notebook to keep them.')
  if (typeof recordedAt !== 'string' || !Number.isFinite(Date.parse(recordedAt))) throw new Error('An attempt needs a valid timestamp.')
  const target = quantumTarget(visit.targetId)
  const pulses = visit.pulses.map(({ axis, angle }) => ({ axis, angle }))
  const { state } = simulatePulses(pulses)
  return {
    id: visit.attempts.length + 1, recordedAt, model: QUANTUM_MODEL,
    targetId: target.id, target: [...target.vector], initialState: [...INITIAL_BLOCH],
    pulses, finalState: [...state], overlap: targetOverlap(state, target.vector), note: visit.note,
  }
}

export function quantumNotebook(visit) {
  return JSON.parse(JSON.stringify({
    schemaVersion: 1, model: QUANTUM_MODEL,
    provenance: 'Local ideal single-qubit simulation. Exact state visible. No shots, noise, hardware or live AI agents.',
    convention: 'hbar=1; H=(omega/2)*sigma_axis; angle=omega*duration in radians; standard right-handed Bloch x,y,z.',
    current: { targetId: visit.targetId, initialState: [...INITIAL_BLOCH], pulses: visit.pulses, finalState: simulatePulses(visit.pulses).state, note: visit.note },
    attempts: visit.attempts,
  }))
}
