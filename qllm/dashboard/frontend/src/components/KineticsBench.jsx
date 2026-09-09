import { useEffect, useId, useRef, useState } from "react";
import { DEMO_MODELS } from "../lib/studioSimulation.js";
import {
  kineticsFrame,
  kineticsObservation,
  startKineticReplay,
  scrubSample,
} from "../lib/kineticsBench.js";
import Icon from "./WorldIcon.jsx";
import "../kinetics-bench.css";

function Specimen({ chemistry, value, label, parameter, starting }) {
  const id = useId();
  return (
    <div className="kinetic-specimen">
      <div className="kinetic-specimen-title">
        <span>{label}</span>
        <small>rate {parameter}</small>
      </div>
      <svg
        viewBox="0 0 220 190"
        role="img"
        aria-label={`${label}: ${(value * 100).toFixed(1)}% ${chemistry ? "reactant remaining" : "of model capacity"}`}
      >
        {chemistry ? (
          <>
            <defs>
              <clipPath id={id}>
                <path d="M54 32H166V143Q166 159 150 159H70Q54 159 54 143Z" />
              </clipPath>
            </defs>
            <ellipse
              cx="110"
              cy="163"
              rx="80"
              ry="11"
              fill="#769890"
              opacity=".15"
            />
            <g clipPath={`url(#${id})`}>
              <rect x="54" y="52" width="112" height="107" fill="#75aaa0" />
              <rect
                x="54"
                y="52"
                width="112"
                height={107 * value}
                fill="#dfb578"
              />
              {Array.from({ length: 32 }, (_, i) => {
                const x = 64 + (i % 6) * 18,
                  y = 65 + Math.floor(i / 6) * 16;
                return (
                  <g key={i} opacity=".6">
                    <circle
                      cx={x}
                      cy={y}
                      r="3.5"
                      fill={i / 32 < value ? "#8c5b34" : "#235b50"}
                    />
                    <circle cx={x + 3} cy={y - 3} r="1.5" fill="#fff9dc" />
                  </g>
                );
              })}
            </g>
            <path
              d="M49 27H171M54 29V143Q54 160 70 160H150Q166 160 166 143V29"
              fill="none"
              stroke="#476f65"
              strokeWidth="4"
              strokeLinecap="round"
            />
            {[0, 1, 2, 3].map((i) => (
              <path
                key={i}
                d={`M148 ${57 + i * 24}h14`}
                stroke="#edf0d9"
                strokeWidth="2"
              />
            ))}
            <text
              x="110"
              y="182"
              textAnchor="middle"
              className="kinetic-svg-caption"
            >
              A → B
            </text>
          </>
        ) : (
          <>
            <ellipse
              cx="110"
              cy="147"
              rx="92"
              ry="27"
              fill="#739381"
              opacity=".16"
            />
            <circle
              cx="110"
              cy="89"
              r="79"
              fill="#e9ecd6"
              stroke="#8caf9b"
              strokeWidth="4"
            />
            <circle
              cx="110"
              cy="89"
              r="69"
              fill="#d3dec0"
              stroke="#bfd0ae"
              strokeWidth="1"
            />
            {Array.from({ length: 64 }, (_, i) => {
              const r = Math.sqrt((i + 0.5) / 64) * 62,
                a = i * 2.399963;
              const x = 110 + Math.cos(a) * r,
                y = 89 + Math.sin(a) * r;
              const shown = Math.max(0, Math.min(1, value * 64 - i));
              return (
                <g
                  key={i}
                  opacity={0.08 + 0.92 * shown}
                  transform={`translate(${x},${y}) rotate(${i * 37})`}
                >
                  <ellipse rx="6" ry="4" fill="#598865" />
                  <ellipse cx="2" rx="2.3" ry="1.8" fill="#bad59b" />
                </g>
              );
            })}
            <path
              d="M58 39Q109 3 163 37"
              fill="none"
              stroke="#f8faed"
              strokeWidth="4"
              opacity=".8"
              strokeLinecap="round"
            />
          </>
        )}
      </svg>
      <strong>
        {(value * 100).toFixed(1)}
        <small>%</small>
      </strong>
      <p>
        {starting
          ? "Starting condition"
          : chemistry
            ? "reactant remaining"
            : "of model capacity"}
      </p>
    </div>
  );
}

export default function KineticsBench({
  studioId,
  parameter,
  onChange,
  illustration,
  onRun,
  busy,
  paused,
  reducedMotion,
  onSampleChange,
  onDiscuss,
  onCollect,
}) {
  const model = DEMO_MODELS[studioId],
    chemistry = studioId === "chemistry";
  const frame = kineticsFrame(illustration),
    result = illustration?.result;
  const [playing, setPlaying] = useState(false);
  const latest = useRef({}),
    drag = useRef(null);
  latest.current = { onSampleChange, paused, reducedMotion };
  useEffect(() => {
    if (!playing || !frame) return;
    return startKineticReplay({
      start: frame.index,
      onSample: (index) => latest.current.onSampleChange(index),
      onStop: () => setPlaying(false),
      state: () => latest.current,
    });
  }, [playing, result]);
  const scrub = (index) => {
    setPlaying(false);
    onSampleChange(index);
  };
  const point = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    scrub(scrubSample(e.clientX, box.left, box.width));
  };
  const title = chemistry
    ? "A reaction you can rewind."
    : "A colony you can grow.";
  const starting = chemistry ? 1 : 0.1;
  const baseline = frame?.baseline ?? starting,
    own = frame?.own ?? starting;
  const difference = frame ? (frame.own - frame.baseline) * 100 : 0;
  const dirty = frame && frame.parameter !== parameter;
  return (
    <section
      className={`kinetics-bench ${chemistry ? "is-chemistry" : "is-biology"}`}
      aria-label={
        chemistry ? "Hands-on reaction bench" : "Hands-on growth bench"
      }
      data-sample={frame?.index ?? "none"}
      data-value={frame?.own ?? "none"}
    >
      <header>
        <span className="rw-eyebrow">
          {chemistry ? "THE REACTION BENCH" : "THE GROWTH GARDEN"}
        </span>
        <span className="kinetic-toy">Browser toy</span>
      </header>
      <h4>{title}</h4>
      <p className="kinetic-intro">
        {chemistry
          ? "Change the reaction rate. Watch A become B, then travel back to any moment."
          : "Change the growth rate. Watch two colonies evolve, then compare the same moment."}
      </p>
      <div className="kinetic-pair">
        <Specimen
          chemistry={chemistry}
          value={baseline}
          label="Baseline"
          parameter={model.initial}
          starting={!frame}
        />
        <Specimen
          chemistry={chemistry}
          value={own}
          label="Your comparison"
          parameter={frame?.parameter ?? parameter}
          starting={!frame}
        />
      </div>
      <div className="kinetic-parameter">
        <label>
          {model.parameter}
          <output>
            {parameter.toFixed(1)}
            {model.unit}
          </output>
          <input
            aria-label={model.parameter}
            type="range"
            min={model.min}
            max={model.max}
            step={model.step}
            value={parameter}
            onChange={(e) => onChange(Number(e.target.value))}
          />
        </label>
        <button
          className="kinetic-run"
          disabled={busy}
          onClick={() => {
            setPlaying(false);
            onRun();
          }}
        >
          <Icon name={chemistry ? "chemistry" : "biology"} size={18} />
          {busy ? "Calculating…" : chemistry ? "Run reaction" : "Grow colonies"}
        </button>
      </div>
      <div className="kinetic-presets">
        <span>Try a pace</span>
        {[
          [0.2, "Slow"],
          [0.5, "Baseline"],
          [1.2, "Fast"],
        ].map(([rate, name]) => (
          <button
            key={rate}
            aria-pressed={parameter === rate}
            onClick={() => onChange(rate)}
          >
            {name}
            <small>{rate.toFixed(1)}</small>
          </button>
        ))}
      </div>
      {frame ? (
        <>
          <div className="kinetic-time-header">
            <span>Explore the same moment</span>
            <output>
              t = {frame.time.toFixed(2)} {chemistry ? "s" : "model units"}
            </output>
          </div>
          <div
            className="kinetic-timeline"
            role="group"
            aria-label="Drag or tap the timeline to inspect a moment"
            onPointerDown={(e) => {
              if (e.button !== 0) return;
              drag.current = e.pointerId;
              e.currentTarget.setPointerCapture(e.pointerId);
              point(e);
            }}
            onPointerMove={(e) => {
              if (drag.current === e.pointerId) point(e);
            }}
            onPointerUp={(e) => {
              if (e.currentTarget.hasPointerCapture(e.pointerId))
                e.currentTarget.releasePointerCapture(e.pointerId);
              drag.current = null;
            }}
            onPointerCancel={() => {
              drag.current = null;
            }}
          >
            <div
              className="kinetic-time-fill"
              style={{ width: `${(frame.index / 120) * 100}%` }}
            />
            <span style={{ left: `${(frame.index / 120) * 100}%` }} />
            {[0, 2, 4, 6, 8, 10].map((t) => (
              <i key={t} style={{ left: `${t * 10}%` }}>
                <b>{t}</b>
              </i>
            ))}
          </div>
          <div className="kinetic-playback">
            <button
              onClick={() => setPlaying(!playing)}
              disabled={reducedMotion}
            >
              <Icon name={playing ? "pause" : "play"} size={15} />
              {playing
                ? paused
                  ? "Room paused"
                  : "Pause replay"
                : frame.index === 120
                  ? "Replay from start"
                  : "Play time"}
            </button>
            <input
              aria-label="Model time"
              type="range"
              min="0"
              max="120"
              step="1"
              value={frame.index}
              onChange={(e) => scrub(Number(e.target.value))}
            />
            <button onClick={() => scrub(0)}>Start</button>
            <button onClick={() => scrub(120)}>End</button>
          </div>
          {reducedMotion && (
            <p className="kinetic-motion-note">
              Motion is reduced. Scrub or use Start / End to inspect any moment.
            </p>
          )}
          <div
            className="kinetic-readout"
            role="status"
            aria-live={playing ? "off" : "polite"}
          >
            <strong>
              {Math.abs(difference) < 0.05
                ? "The two conditions match here."
                : `${Math.abs(difference).toFixed(1)} percentage points ${difference > 0 ? "more" : "less"} ${chemistry ? "reactant" : "population"}.`}
            </strong>
            <span>
              {chemistry
                ? `${(frame.product * 100).toFixed(1)}% has converted to B in your comparison.`
                : "The capacity is fixed at 100%; the rate changes how quickly it is approached."}
              {dirty
                ? ` Rate ${parameter.toFixed(1)} is selected; run again to apply it.`
                : ""}
            </span>
          </div>
          {frame.previous !== null && (
            <p className="kinetic-previous">
              Your previous run at this moment:{" "}
              {(frame.previous * 100).toFixed(1)}%. Current:{" "}
              {(frame.own * 100).toFixed(1)}%.
            </p>
          )}
          <button
            className="kinetic-discuss"
            onClick={() => {
              setPlaying(false);
              onDiscuss(kineticsObservation(frame));
            }}
          >
            <Icon name="send" size={15} />
            Discuss this moment
          </button>
          {onCollect && (
            <button
              className="take-result-card"
              disabled={busy}
              onClick={() => {
                setPlaying(false);
                onCollect(frame.index);
              }}
            >
              <Icon name="paper" size={16} />
              Take result card
            </button>
          )}
        </>
      ) : (
        <p className="kinetic-invitation">
          Both start from the same condition. Run a comparison to unlock time
          travel.
        </p>
      )}
      <details className="kinetic-method">
        <summary>Model and limits</summary>
        <code>{model.code}</code>
        <p>
          {chemistry
            ? "Ideal, irreversible first-order A → B in a closed normalized system. Not a recipe or molecular simulation. Vessel colors represent fractions, not physical colors."
            : "Ideal logistic growth with initial population 10% of a fixed capacity. Time is in arbitrary model units. Cell symbols are a fractional display, not a microscopy image or discrete cell simulation."}
        </p>
        <p>
          {frame ? "121 sample values computed in this browser. " : ""}Synthetic
          inputs; no research agent or external service. Dragging time selects
          an existing sample and does not start another calculation.
        </p>
      </details>
    </section>
  );
}
