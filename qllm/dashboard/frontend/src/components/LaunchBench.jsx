import { useEffect, useRef, useState } from "react";
import { aimAngle, projectile } from "../lib/projectileToy.js";
import Icon from "./WorldIcon.jsx";
import "../launch-bench.css";

const screen = (p) => [55 + p.x * 10.5, 230 - p.y * 8];
const path = (points) =>
  points.map((p, i) => `${i ? "L" : "M"}${screen(p).join(",")}`).join(" ");
const predicted = (angle) =>
  Array.from({ length: 61 }, (_, i) => projectile(angle, i / 60));

export default function LaunchBench({
  onCollect,
  parameter,
  onChange,
  result,
  previousResult,
  onRun,
  busy,
  paused,
  flightRef,
}) {
  const rocketRef = useRef(null),
    hostRef = useRef(null),
    dragRef = useRef(null);
  const [flying, setFlying] = useState(false);
  useEffect(() => {
    let frame;
    let wasFlying = false;
    const draw = () => {
      const pose = flightRef?.current;
      if (!document.hidden && pose && rocketRef.current) {
        const [x, y] = screen(pose);
        rocketRef.current.setAttribute(
          "transform",
          `translate(${x},${y}) rotate(${(-pose.heading * 180) / Math.PI})`,
        );
        hostRef.current.dataset.flightProgress = pose.fraction.toFixed(3);
        if (wasFlying !== !pose.landed) {
          wasFlying = !pose.landed;
          setFlying(wasFlying);
        }
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [flightRef]);
  const aim = (event) => {
    // Inverse SVG CTM also handles letterboxing and narrow-screen scaling.
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      matrix.inverse(),
    );
    onChange(aimAngle(point.x, point.y));
  };
  const a = (parameter * Math.PI) / 180;
  const [tipX, tipY] = [55 + Math.cos(a) * 94, 230 - Math.sin(a) * 94];
  const landed = result && projectile(result.parameter, 1);
  const baseline = projectile(45, 1);
  const previousFlight =
    previousResult && projectile(previousResult.parameter, 1);
  const different = result && result.parameter !== parameter;
  return (
    <section
      ref={hostRef}
      className="launch-bench"
      aria-label="Hands-on launch bench"
    >
      <header>
        <div>
          <span className="rw-eyebrow">THE LAUNCH BENCH</span>
          <h4>A little change. A different flight.</h4>
        </div>
        <span className="launch-toy-label">Browser toy</span>
      </header>
      <p>
        Drag the aiming handle or tap the sky. Then launch and follow the
        landing point.
      </p>
      <svg
        className="launch-playfield"
        viewBox="0 0 570 278"
        role="group"
        aria-label="Projectile playground. Drag or tap to aim; angle controls below provide an alternative."
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          dragRef.current = event.pointerId;
          event.currentTarget.setPointerCapture(event.pointerId);
          aim(event);
        }}
        onPointerMove={(event) => {
          if (dragRef.current === event.pointerId) aim(event);
        }}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            event.currentTarget.releasePointerCapture(event.pointerId);
          dragRef.current = null;
        }}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
      >
        <path
          className="launch-hills"
          d="M0 200 Q75 158 140 205 T300 190 T440 199 T570 185 V278 H0Z"
        />
        {[0, 10, 20, 30, 40].map((x) => (
          <g key={x} className="launch-grid">
            <path d={`M${55 + x * 10.5} 38 V235`} />
            <text x={55 + x * 10.5} y="252" textAnchor="middle">
              {x} m
            </text>
          </g>
        ))}
        <text className="launch-height-label" x="17" y="30">
          Height / m
        </text>
        {[0, 10, 20].map((y) => (
          <g key={y} className="launch-grid">
            <path d={`M50 ${230 - y * 8} H530`} />
            <text x="38" y={234 - y * 8} textAnchor="end">
              {y}
            </text>
          </g>
        ))}
        <path className="launch-baseline" d={path(predicted(45))} />
        <path className="launch-prediction" d={path(predicted(parameter))} />
        {previousResult && (
          <path className="launch-previous" d={path(previousResult.points)} />
        )}
        {result && <path className="launch-recorded" d={path(result.points)} />}
        <path className="launch-ground" d="M40 233 H535" />
        <path className="launch-aim-line" d={`M55 230 L${tipX} ${tipY}`} />
        <circle className="launch-aim-target" cx={tipX} cy={tipY} r="22" />
        <circle className="launch-aim-dot" cx={tipX} cy={tipY} r="6" />
        <text className="launch-angle-label" x={tipX + 27} y={tipY + 5}>
          {parameter}°
        </text>
        <g transform="translate(55,230)">
          <path fill="#a6b5a0" d="M-14 0L0-11L14 0Z" />
        </g>
        {result && (
          <g
            ref={rocketRef}
            className="launch-flying-rocket"
            transform={`translate(${screen(landed).join(",")})`}
            aria-hidden="true"
          >
            <path fill="#f2dec1" d="M-15-5L5-5L17 0L5 5L-15 5Z" />
            <path fill="#dc9c79" d="M-12-5L-20-12L-19 0L-20 12L-12 5Z" />
            <circle cx="1" cy="0" r="3" fill="#2a676c" />
          </g>
        )}
        <text className="launch-sky-note" x="550" y="29" textAnchor="end">
          20 m/s · no drag · flat ground
        </text>
      </svg>
      <div className="launch-controls">
        <label>
          Launch angle <output>{parameter}°</output>
          <input
            type="range"
            aria-label="Launch angle"
            min="20"
            max="70"
            step="1"
            value={parameter}
            onChange={(e) => onChange(Number(e.target.value))}
          />
        </label>
        <button
          className="rw-primary launch-fire"
          onClick={onRun}
          disabled={busy}
        >
          <Icon name="play" size={18} />
          {busy ? "Calculating…" : "Launch rocket"}
        </button>
      </div>
      <div className="launch-presets">
        <span>Try a different arc</span>
        {[30, 45, 60].map((angle) => (
          <button
            key={angle}
            aria-pressed={parameter === angle}
            onClick={() => onChange(angle)}
          >
            {angle}°
          </button>
        ))}
      </div>
      <div className="launch-key">
        <span>— Last launch</span>
        <span>┄ Aim preview</span>
        <span>┄ 45° baseline</span>
        {previousResult && (
          <span>··· Previous {previousResult.parameter}°</span>
        )}
      </div>
      <div className="launch-result" role="status">
        {landed ? (
          <>
            <strong>
              {flying
                ? paused
                  ? "Flight paused"
                  : "Following your flight…"
                : `Landed at ${landed.range.toFixed(1)} m`}
            </strong>
            <span>
              {result.parameter}° · {(landed.range - baseline.range).toFixed(1)}{" "}
              m versus the 45° baseline
              {different ? ` · Aim is now ${parameter}°; launch to apply.` : ""}
            </span>
          </>
        ) : (
          <>
            <strong>What changes when you aim higher?</strong>
            <span>Try 30°, then 60°. Compare where they land.</span>
          </>
        )}
      </div>
      {previousFlight && landed && (
        <p className="launch-discovery">
          {Math.abs(landed.range - previousFlight.range) < 0.05
            ? `Same landing as ${previousResult.parameter}°.`
            : `${(landed.range - previousFlight.range).toFixed(1)} m from your previous ${previousResult.parameter}° landing.`}{" "}
          Peak height changed by{" "}
          {(landed.height - previousFlight.height).toFixed(1)} m. What does that
          suggest?
        </p>
      )}
      {result && onCollect && (
        <button
          className="take-result-card"
          disabled={busy}
          onClick={onCollect}
        >
          <Icon name="paper" size={16} />
          Take result card
        </button>
      )}
      <details>
        <summary>Model, calculation and limits</summary>
        <p>
          A rocket-shaped, unpowered projectile. Speed 20 m/s, gravity 9.81
          m/s²; no thrust, drag or wind. Ground and launch height are equal.
          Playback is an authored 2.8-second illustration, not real-time
          physics.
        </p>
        <code>x = v cos(θ)t; y = v sin(θ)t − ½gt²</code>
        <p>
          {result
            ? `${result.points.length} sample values computed locally. `
            : ""}
          Synthetic inputs only. This does not test your research proposal.
        </p>
      </details>
    </section>
  );
}
