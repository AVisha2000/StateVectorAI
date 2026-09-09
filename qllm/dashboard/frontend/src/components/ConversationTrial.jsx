import { useMemo } from "react";
import { DEMO_MODELS } from "../lib/studioSimulation.js";
import { captureResultCard, resultCardView } from "../lib/resultCards.js";
import Icon from "./WorldIcon.jsx";
import "../conversation-trial.css";

export default function ConversationTrial({ trial, index, calculation, dispatch }) {
  const model = DEMO_MODELS[trial.studioId];
  const view = useMemo(() => {
    const card = captureResultCard({ result: trial.result });
    return card ? resultCardView(card) : null;
  }, [trial.result]);
  if (!model) return null;
  const running = calculation.busy && calculation.trialIndex === index;
  const error = trial.error;
  const stale = trial.result && trial.parameter !== trial.result.parameter;
  return (
    <section className="conversation-trial" aria-label={`Browser experiment ${index + 1}`}>
      <h4>{model.title}</h4>
      <label htmlFor={`trial-input-${index}`}>
        {model.parameter}<output>{trial.parameter}{model.unit}</output>
      </label>
      <input id={`trial-input-${index}`} type="range" aria-label={`${model.parameter} in conversation`}
        min={model.min} max={model.max} step={model.step} value={trial.parameter}
        onChange={(event) => dispatch({ type: "trial-input", index, parameter: Number(event.target.value) })} />
      <div className="conversation-trial-range"><span>{model.min}{model.unit}</span><span>{model.max}{model.unit}</span></div>
      <button className="conversation-trial-run" aria-disabled={calculation.busy}
        onClick={() => { if (!calculation.busy) calculation.runTrial(index); }}>
        <Icon name="code" size={16} />
        {running ? "Calculating in your browser…" : calculation.busy ? "Another calculation is running…" : error ? "Retry browser experiment" : "Run browser experiment"}
      </button>
      <div className="conversation-trial-output" role="status">
        {view ? <>
          <span>{view.label} · {view.location}</span>
          <dl>
            <div><dt>Your run</dt><dd>{view.own}</dd></div>
            <div><dt>Baseline</dt><dd>{view.baseline}</dd></div>
          </dl>
          <p>{view.parameter}; baseline input {view.baselineParameter}.</p>
          {stale && <p className="conversation-trial-stale">Showing the completed run, not the edited input. Run again to apply {trial.parameter}{model.unit}.</p>}
        </> : <p>Baseline input: {model.initial}{model.unit}. Run to compare.</p>}
      </div>
      {error && <p className="conversation-trial-error" role="alert">{error}</p>}
      {view && <div className="conversation-trial-actions">
        <button onClick={() => dispatch({ type: "quote", kind: "observation", text: view.quote })}>Discuss this result</button>
        <button onClick={() => dispatch({ type: "result-card", action: "collect", illustration: { result: trial.result } })}>
          <Icon name="paper" size={15} /> Take result card
        </button>
      </div>}
      <details><summary>Built-in model &amp; limits</summary>
        <code>{model.code}</code>
        <p>Fixed baseline {model.initial}{model.unit}. Synthetic inputs only. This does not execute your code or test an arbitrary research proposal.</p>
      </details>
      <small>Browser toy · No research model connected</small>
    </section>
  );
}
