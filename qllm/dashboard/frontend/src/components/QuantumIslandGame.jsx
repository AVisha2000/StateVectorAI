import { useEffect, useMemo, useRef, useState } from 'react'
import Icon from './WorldIcon.jsx'
import QuantumIslandCanvas from './QuantumIslandCanvas.jsx'
import {
  initialQuantumVisit, QUANTUM_TARGETS, MAX_PULSES, MAX_ATTEMPTS, PULSE_KEYS,
  simulatePulses, quantumTarget, targetOverlap, captureQuantumAttempt, quantumNotebook,
} from '../lib/quantumPilot.js'
import '../quantum-island.css'

const degree = (angle) => Math.round(angle * 180 / Math.PI)
const signed = (value) => `${value >= 0 ? '+' : ''}${value.toFixed(3)}`
export default function QuantumIslandGame({ visit, onVisitChange, onClose, reducedMotion }) {
  const initial = useRef(initialQuantumVisit()), dialog = useRef(null), flight = useRef(null)
  const value = visit || initial.current, current = useRef(value)
  current.current = value
  const [replay, setReplay] = useState(null), [feedback, setFeedback] = useState(''), [error, setError] = useState('')
  const replayGeneration = useRef(0)
  const [compare, setCompare] = useState(true)
  const target = quantumTarget(value.targetId)
  const result = useMemo(() => simulatePulses(value.pulses), [value.pulses])
  const lastAttempt = [...value.attempts].reverse().find((attempt) => attempt.targetId === target.id)
  const overlap = targetOverlap(result.state, target.vector)
  const presentation = useMemo(() => {
    const shown = replay ? simulatePulses(value.pulses, replay.position) : result
    return { ...shown, target: target.vector, ghost: compare && lastAttempt ? simulatePulses(lastAttempt.pulses).points : [] }
  }, [replay, value.pulses, result, target, compare, lastAttempt])
  const update = (patch) => {
    const next = { ...current.current, ...patch }
    current.current = next
    onVisitChange(next)
  }
  useEffect(() => {
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    dialog.current.showModal()
    flight.current?.focus()
    return () => { replayGeneration.current += 1; dialog.current?.close(); document.body.style.overflow = previousOverflow }
  }, [])
  const replayStarted = replay?.started
  useEffect(() => {
    if (reducedMotion) { replayGeneration.current += 1; setReplay(null) }
  }, [reducedMotion])
  useEffect(() => {
    if (replayStarted == null) return
    let frame, last = null, position = 0, stopped = false
    const tick = (now) => {
      if (stopped || replayGeneration.current !== replayStarted) return
      // Pause presentation time when hidden. Replays never append pulses/results.
      if (!document.hidden && last !== null) position += Math.min((now - last) / 1000, 0.05) / 0.35
      last = now
      if (position >= current.current.pulses.length) {
        setReplay(null); setFeedback('Replay complete. Your pulse sequence is unchanged.')
        return
      }
      setReplay({ started: replayStarted, position })
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => { stopped = true; cancelAnimationFrame(frame) }
  }, [replayStarted])
  const pulse = (axis, direction) => {
    if (replay) return
    const previous = current.current
    if (previous.pulses.length >= MAX_PULSES) { setError('Sequence full: save it, then start a new attempt.'); return }
    const angle = direction * previous.angleDegrees * Math.PI / 180
    update({ pulses: [...previous.pulses, { axis, angle }] })
    setFeedback(`${axis.toUpperCase()} ${direction > 0 ? '+' : '−'}${previous.angleDegrees}° pulse applied.`)
    setError('')
  }
  const startReplay = () => {
    if (!value.pulses.length || replay) return
    if (reducedMotion) { setFeedback('Reduced motion: the complete trajectory is shown without animation.'); return }
    replayGeneration.current += 1
    setReplay({ started: replayGeneration.current, position: 0 }); setFeedback('Replaying your pulse sequence…')
  }
  const stopReplay = () => {
    // Invalidate synchronously: a queued animation frame must not revive replay
    // between this click and React's effect cleanup.
    replayGeneration.current += 1
    setReplay(null); setFeedback('Replay stopped. Showing the complete sequence.')
  }
  const save = () => {
    try {
      const attempt = captureQuantumAttempt(current.current, new Date().toISOString())
      update({ attempts: [...current.current.attempts, attempt] })
      setFeedback(`Attempt ${attempt.id} saved with its pulses, target and note. Download the notebook to keep it after a refresh.`)
      setError('')
    } catch (cause) { setError(cause.message) }
  }
  const download = () => {
    try {
      const blob = new Blob([JSON.stringify(quantumNotebook(current.current), null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob), anchor = document.createElement('a')
      anchor.href = url; anchor.download = 'quantum-island-notebook.json'
      document.body.append(anchor); anchor.click(); anchor.remove()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
      setFeedback('Notebook downloaded. It includes your current sequence and saved attempts.')
    } catch { setError('The notebook could not be downloaded. Your visit is still here; try again.') }
  }
  const key = (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.target.closest('input, textarea, select, summary, a, [contenteditable="true"]')) return
    const command = PULSE_KEYS[event.key.toLowerCase()]
    if (!command) return
    event.preventDefault()
    if (!event.repeat) pulse(...command)
  }
  const guide = !value.pulses.length ? target.hint : overlap >= 0.99
    ? 'You reached the target. Save this route, then try another axis or reverse the order of two pulses.'
    : value.pulses[value.pulses.length - 1]?.axis === 'z' && Math.abs(1 - Math.abs(result.state[2])) < 1e-12
      ? 'A Z pulse at a pole changes only global phase. Try an X or Y pulse to move away from the pole.'
      : 'Trace the mint path. Undo one pulse to compare, or try a smaller pulse near the target.'
  const pulsePad = () => <div className="qg-pulse-pad">{[['x', 1, 'W'], ['x', -1, 'S'], ['y', 1, 'D'], ['y', -1, 'A'], ['z', 1, 'E'], ['z', -1, 'Q']].map(([axis, direction, shortcut]) =>
    <button key={shortcut} className="qg-pulse-button" aria-label={`Apply ${direction > 0 ? 'positive' : 'negative'} ${axis.toUpperCase()} pulse`} disabled={Boolean(replay) || value.pulses.length >= MAX_PULSES} onClick={() => pulse(axis, direction)}><span>{axis.toUpperCase()} {direction > 0 ? '+' : '−'}</span><kbd>{shortcut}</kbd></button>)}</div>
  return <dialog ref={dialog} className="qg-dialog" aria-labelledby="quantum-title" onKeyDown={key} onCancel={(event) => { event.preventDefault(); onClose() }}>
    <header className="qg-header">
      <button className="qg-button qg-back" onClick={onClose}><Icon name="back" size={17} /> Back to world</button>
      <div><span className="qg-eyebrow">QUANTUM COMPUTING ISLAND</span><h1 id="quantum-title">A small state. A new possibility.</h1></div>
      <span className="qg-mode"><span /> Ideal simulation</span>
    </header>
    <div className="qg-layout">
      <section className="qg-play" aria-label="Quantum playground">
        <div className="qg-mission">
          <div><span className="qg-eyebrow">YOUR FIRST EXPERIMENT</span><h2>{target.name}</h2><p>Steer the mint state to the amber target.</p></div>
          <label>Destination<select aria-label="Quantum target" value={value.targetId} disabled={Boolean(replay)} onChange={(event) => { update({ targetId: event.target.value }); setFeedback('Target changed. Your current pulse sequence is preserved.') }}>
            {QUANTUM_TARGETS.map((item) => <option key={item.id} value={item.id}>{item.ket} · {item.name}</option>)}
          </select></label>
        </div>
        <div ref={flight} tabIndex={0} className="qg-stage" role="region" aria-label="Quantum flight controls" aria-describedby="quantum-key-help" data-state={presentation.state.map((x) => x.toFixed(6)).join(',')} data-pulse-count={value.pulses.length} data-replaying={Boolean(replay)}>
          <QuantumIslandCanvas presentation={presentation} />
          <div className="qg-stage-caption"><span>01 / THE BLOCH SPHERE</span><span>Drag to look around</span></div>
          <div className="qg-legend"><span className="qg-state-dot" /> Your state <span className="qg-target-dot" /> Target {target.ket}{compare && lastAttempt && <><span className="qg-ghost-dot" /> Attempt {lastAttempt.id}</>}</div>
        </div>
        <div className="qg-mobile-pad"><div className="qg-section-heading"><h3>Steer the state</h3><span>{value.angleDegrees}° per tap</span></div>{pulsePad()}</div>
        <div className="qg-readout">
          <div><span className="qg-eyebrow">{replay ? 'REPLAY OVERLAP' : 'TARGET OVERLAP'}</span><strong data-testid="quantum-overlap">{(targetOverlap(presentation.state, target.vector) * 100).toFixed(1)}<small>%</small></strong></div>
          <div className="qg-overlap-track" role="progressbar" aria-label="Target overlap" aria-valuenow={Number((targetOverlap(presentation.state, target.vector) * 100).toFixed(1))} aria-valuemin={0} aria-valuemax={100}><span style={{ width: `${targetOverlap(presentation.state, target.vector) * 100}%` }} /></div>
          <div className="qg-coordinate"><span>X {signed(presentation.state[0])}</span><span>Y {signed(presentation.state[1])}</span><span>Z {signed(presentation.state[2])}</span></div>
        </div>
        <div className="qg-timeline-header"><h3>Your pulse trail <span>{value.pulses.length}/{MAX_PULSES}</span></h3><div className="qg-inline-actions">
          <button className="qg-button" disabled={!value.pulses.length || Boolean(replay)} onClick={() => { update({ pulses: current.current.pulses.slice(0, -1) }); setFeedback('Last pulse undone. Saved attempts are unchanged.'); setError('') }}>Undo</button>
          <button className="qg-button" disabled={!value.pulses.length} onClick={() => replay ? stopReplay() : startReplay()}>{replay ? 'Stop replay' : 'Replay'}</button>
          <button className="qg-button" disabled={!value.pulses.length || Boolean(replay)} onClick={() => { update({ pulses: [] }); setFeedback('New attempt starts at |0⟩. Your note and saved attempts are retained.'); setError('') }}>New attempt</button>
        </div></div>
        <ol className="qg-pulse-trail" aria-label="Recorded pulses">{value.pulses.map((item, i) => <li key={i}><small>{i + 1}</small>{item.axis.toUpperCase()} {degree(item.angle) > 0 ? '+' : '−'}{Math.abs(degree(item.angle))}°</li>)}</ol>
        {!value.pulses.length && <p className="qg-empty">Your first pulse starts a trail. Every move can be undone.</p>}
      </section>
      <aside className="qg-console" aria-label="Quantum control desk">
        <div className="qg-guide"><div className="qg-guide-heading"><Icon name="quantum-island" size={25} /><div><b>A guide at your side</b><span>Scripted guidance</span></div></div><p>{guide}</p></div>
        <section className="qg-controls" aria-labelledby="quantum-controls-title">
          <div className="qg-section-heading"><h2 id="quantum-controls-title">Steer the state</h2><span>One tap, one pulse</span></div>
          <p id="quantum-key-help">Use W/S for X, A/D for Y, Q/E for Z. Or use the buttons below.</p>
          {pulsePad()}
          <label className="qg-angle">Pulse angle <output>{value.angleDegrees}°</output><input aria-label="Pulse angle" type="range" min="15" max="90" step="15" disabled={Boolean(replay)} value={value.angleDegrees} onChange={(event) => update({ angleDegrees: Number(event.target.value) })} /></label>
          <button className="qg-primary" disabled={!value.pulses.length || Boolean(replay) || value.attempts.length >= MAX_ATTEMPTS} onClick={save}>Save this attempt <Icon name="arrow" size={18} /></button>
          {value.attempts.length >= MAX_ATTEMPTS && <p className="qg-note">All 12 notebook slots are used. Download to keep them; you can continue experimenting.</p>}
          <div className="qg-feedback" role="status">{feedback || (value.pulses.length ? 'Your sequence is restored. Keep exploring or replay this route.' : 'Start with a Y+ pulse and watch the state move.')}</div>
          {error && <p className="qg-error" role="alert">{error}</p>}
        </section>
        <section className="qg-notebook" aria-labelledby="quantum-notebook-title">
          <div className="qg-section-heading"><h2 id="quantum-notebook-title">A thought worth keeping</h2><span>{value.attempts.length} saved</span></div>
          <label htmlFor="quantum-note">What would you try next?</label><textarea id="quantum-note" rows={3} maxLength={2000} placeholder="A different path? Reverse the order?" value={value.note} onChange={(event) => update({ note: event.target.value })} />
          {lastAttempt && <label className="qg-checkbox"><input type="checkbox" checked={compare} onChange={(event) => setCompare(event.target.checked)} /> Overlay last saved route to this target</label>}
          {value.attempts.length > 0 && <details><summary>Saved attempts</summary><ol className="qg-attempts">{value.attempts.map((attempt) => <li key={attempt.id}><b>#{attempt.id} · {quantumTarget(attempt.targetId).ket}</b><span>{attempt.pulses.length} pulses · {(attempt.overlap * 100).toFixed(1)}%</span>{attempt.note && <p>{attempt.note}</p>}</li>)}</ol></details>}
          <button className="qg-button qg-download" onClick={download}><Icon name="paper" size={16} /> Download notebook</button>
          <p className="qg-note">Kept when you return to the world. Download to keep your work after a refresh. No research profile is inferred.</p>
        </section>
      </aside>
    </div>
    <footer className="qg-footer"><span>One qubit · Exact simulated state · No live AI or hardware connected</span><details><summary>Model & limits</summary><p>Ideal unitary rotations from |0⟩, with no noise or measurement shots. H = (ω/2)σ, ℏ = 1; pulse angle = ω × duration in radians. Bloch coordinates use the standard right-handed X, Y, Z convention. Target overlap = (1 + r · target)/2. This is a visual control prototype, not a QLLM research result. The notebook contains the replayable pulses and your notes.</p></details></footer>
  </dialog>
}
