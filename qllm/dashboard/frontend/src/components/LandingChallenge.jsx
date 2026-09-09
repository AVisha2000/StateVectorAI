import { LANDING_TARGETS, landingTarget, presentedLandingTarget, completedLandingTargets, targetInterval, landingFeedback, landingObservation } from "../lib/landingChallenge.js";

export default function LandingChallenge({ challenge, calculation, dispatch, paused }) {
  const target = presentedLandingTarget(challenge, calculation.busy ? calculation.context : null);
  const selected = landingTarget(challenge.targetId);
  const completed = completedLandingTargets(challenge);
  const shot = challenge.shots.at(-1);
  const action = (name) => dispatch({ type: "landing-challenge", action: name });
  return <section className="landing-challenge" aria-label="Landing challenge">
    {!selected ? <button className="landing-challenge-start" onClick={() => action("start")}>
      {challenge.shots.length ? "Resume landing challenge" : "Try landing challenge"}<span>Optional</span>
    </button> : <div className="landing-challenge-target">
      <span>{target?.id !== selected.id ? "THIS FLIGHT" : "LAND BETWEEN THE FLAGS"}</span>
      <strong>{target?.name} <small>{targetInterval(target)}</small></strong>
      {target?.id !== selected.id && <p>Next shot: {selected.name} · {targetInterval(selected)}</p>}
    </div>}
    {shot && (selected || shot.status === "flying") && <div className={`landing-challenge-feedback ${shot.status === "landed" && shot.hit ? "is-hit" : ""}`} role="status">
      <strong>{shot.status === "flying" && paused ? "Flight paused" : landingFeedback(shot)}</strong>
      <span>{landingTarget(shot.targetId).name} · {shot.angle}°{shot.status !== "flying" ? ` · ${shot.range.toFixed(2)} m` : ""}</span>
      {shot.status === "landed" && <button onClick={() => dispatch({ type: "quote", kind: "observation", text: landingObservation(shot) })}>Discuss this shot</button>}
    </div>}
    {(selected || challenge.shots.length > 0) && <details className="landing-challenge-history">
      <summary>Targets & shots · {completed.length}/{LANDING_TARGETS.length} found</summary>
      <p>Land the rocket’s centre inside the marked interval. Edges count. No timer; keep exploring.</p>
      <div className="landing-challenge-actions">
        {selected && <button onClick={() => action("next")}>Move target</button>}
        {selected && <button onClick={() => action("stop")}>Free play</button>}
      </div>
      <ul aria-label="Landing targets">{LANDING_TARGETS.map((t) => <li key={t.id}>{completed.includes(t.id) ? "✓" : "○"} {t.name} · {targetInterval(t)}</li>)}</ul>
      {challenge.shots.length > 0 && <ol aria-label="Shot history">{challenge.shots.map((s, i) => <li key={s.flightId}>
        <strong>Shot {i + 1} · {landingTarget(s.targetId).name}</strong>
        <span>{s.angle}° · {s.range.toFixed(2)} m · {landingFeedback(s)}</span>
      </li>)}</ol>}
      <small>Visit-only progress · Ideal browser toy</small>
    </details>}
  </section>;
}
