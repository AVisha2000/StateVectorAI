import { DEMO_MODELS } from "../lib/studioSimulation.js";
import Icon from "./WorldIcon.jsx";
import LaunchBench from "./LaunchBench.jsx";
import KineticsBench from "./KineticsBench.jsx";
import { KINETIC_STUDIOS } from "../lib/kineticsBench.js";

export default function LocalIllustration({
  studioId,
  savedState,
  onParameterChange,
  calculation,
  paused = false,
  reducedMotion = false,
  flightRef,
  onSampleChange,
  onDiscuss,
  onCollect,
}) {
  const model = DEMO_MODELS[studioId];
  const parameter = savedState?.parameter ?? model?.initial;
  const result = savedState?.result;
  const { busy, error, run } = calculation;
  if (!model) return null;
  const changeParameter = (value) => {
    onParameterChange?.(value);
  };
  if (KINETIC_STUDIOS.includes(studioId))
    return (
      <div className="rw-local-demo kinetic-local-demo">
        <KineticsBench
          studioId={studioId}
          parameter={parameter}
          onChange={changeParameter}
          illustration={{ ...savedState, result }}
          onRun={run}
          busy={busy}
          paused={paused}
          reducedMotion={reducedMotion}
          onSampleChange={onSampleChange}
          onDiscuss={onDiscuss}
          onCollect={(index) =>
            onCollect?.({ ...savedState, result, sampleIndex: index })
          }
        />
        {error ? <p role="alert">{error}</p> : null}
      </div>
    );
  if (studioId === "physics")
    return (
      <div className="rw-local-demo physics-local-demo">
        <LaunchBench
          parameter={parameter}
          onChange={changeParameter}
          result={result}
          previousResult={savedState?.previousResult}
          onRun={run}
          busy={busy}
          paused={paused}
          flightRef={flightRef}
          onCollect={() => onCollect?.({ ...savedState, result })}
        />
        {error ? <p role="alert">{error}</p> : null}
      </div>
    );
  const values = result ? [...result.points, ...result.baseline] : [];
  const min = Math.min(0, ...values.map((point) => point.y)),
    max = Math.max(0.01, ...values.map((point) => point.y));
  const line = (points) =>
    points
      .map(
        (point, index) =>
          `${index ? "L" : "M"}${30 + (index / 120) * 300},${144 - ((point.y - min) / (max - min)) * 120}`,
      )
      .join(" ");
  return (
    <section className="rw-local-demo" aria-label="Local model illustration">
      <span className="rw-eyebrow">TRY A BUILT-IN ILLUSTRATION</span>
      <h4>{model.title}</h4>
      <p>
        Change one parameter and run a tiny calculation in your browser. This
        illustrates the interaction; it does not test your research proposal.
      </p>
      <label>
        {model.parameter}
        <output>
          {parameter}
          {model.unit}
        </output>
        <input
          type="range"
          aria-label={model.parameter}
          min={model.min}
          max={model.max}
          step={model.step}
          value={parameter}
          onChange={(event) => {
            const value = Number(event.target.value);
            onParameterChange?.(value);
          }}
        />
      </label>
      <button className="rw-primary" onClick={run} disabled={busy}>
        <Icon name="code" size={17} />
        {busy ? "Calculating…" : "Run browser illustration"}
      </button>
      {result ? (
        <div className="rw-local-result">
          <svg
            viewBox="0 0 350 177"
            role="img"
            aria-label={`Illustrative ${model.title}. Baseline ${result.baselineParameter}; comparison ${result.parameter}.`}
          >
            <path
              d="M30 18v126h305"
              fill="none"
              stroke="#8e9d7d"
              strokeWidth="1"
            />
            <path
              d={line(result.baseline)}
              fill="none"
              stroke="#a5ac8e"
              strokeWidth="2"
              strokeDasharray="5 4"
            />
            <path
              d={line(result.points)}
              fill="none"
              stroke="#3c7355"
              strokeWidth="2.5"
            />
            <text x="5" y="27" fill="#607252" fontSize="9">
              {max.toFixed(2)}
            </text>
            <text x="5" y="145" fill="#607252" fontSize="9">
              {min.toFixed(2)}
            </text>
            <text
              x="175"
              y="169"
              fill="#607252"
              fontSize="9"
              textAnchor="middle"
            >
              {model.axis}
            </text>
          </svg>
          <div className="rw-plot-key">
            <span>— Your value: {result.parameter}</span>
            <span>┄ Baseline: {result.baselineParameter}</span>
          </div>
          <p role="status">
            121 sample values computed locally.{" "}
            {result.parameter !== parameter
              ? "Run again to apply your new value."
              : ""}
          </p>
          {onCollect && (
            <button
              className="take-result-card"
              disabled={busy}
              onClick={() => onCollect({ ...savedState, result })}
            >
              <Icon name="paper" size={16} />
              Take result card
            </button>
          )}
          <details>
            <summary>See the calculation and limits</summary>
            <code>{model.code}</code>
            <p>{result.provenance}</p>
            <p>
              Idealized model only. The omitted assumptions and real-world
              measurements must be evaluated separately.
            </p>
          </details>
        </div>
      ) : null}
      {error ? <p role="alert">{error}</p> : null}
    </section>
  );
}
