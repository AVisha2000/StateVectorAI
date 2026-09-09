import { projectile } from "../lib/projectileToy.js";
import Icon from "./WorldIcon.jsx";
import LandingChallenge from "./LandingChallenge.jsx";
import "../room-launcher.css";

export default function RoomLauncher({
  value,
  dispatch,
  calculation,
  flying,
  paused,
}) {
  const angle = value.illustration.parameter ?? 45;
  const result = value.illustration.result;
  const aim = (next) =>
    dispatch({ type: "launcher", action: "aim", angle: next });
  return (
    <section className="room-launcher" aria-label="In-room launcher controls">
      <header>
        <span>
          <Icon name="physics" size={16} /> LAUNCHER <small>Browser toy</small>
        </span>
        <button
          onClick={() => dispatch({ type: "launcher", action: "close" })}
          aria-label="Leave launcher controls"
        >
          Done <kbd>Esc</kbd>
        </button>
      </header>
      <div className="room-launcher-main">
        <div className="room-launcher-angle">
          <button
            aria-label="Lower launch angle"
            onClick={() => aim(angle - 1)}
            disabled={angle <= 20}
          >
            −
          </button>
          <output aria-label={`Room launch angle ${angle} degrees`}>
            {angle}
            <span>°</span>
          </output>
          <button
            aria-label="Raise launch angle"
            onClick={() => aim(angle + 1)}
            disabled={angle >= 70}
          >
            +
          </button>
        </div>
        <button
          className="room-launcher-fire"
          disabled={calculation.busy}
          onClick={calculation.run}
        >
          <Icon name="play" size={17} />
          {calculation.busy ? "Calculating…" : "Launch"}
          <kbd>Space</kbd>
        </button>
        <div className="room-launcher-prediction">
          <span>Aim preview</span>
          <strong>{projectile(angle, 1).range.toFixed(2)} m</strong>
          <small>Ideal range · 20 m/s · no drag</small>
        </div>
      </div>
      <LandingChallenge challenge={value.launcherChallenge} calculation={calculation} dispatch={dispatch} paused={paused} />
      <p>
        Drag up/down in the scene to aim. <kbd>↑</kbd> <kbd>↓</kbd> also work.{" "}
        <kbd>Shift</kbd> changes 5°.
      </p>
      {result && (
        <div className="room-launcher-result" role="status">
          <span>
            {flying
              ? paused
                ? "Flight paused"
                : "Following your flight…"
              : "Last calculation"}
            <strong>
              {result.parameter}° · {result.points.at(-1).x.toFixed(2)} m
            </strong>
          </span>
          <button
            disabled={calculation.busy}
            onClick={() =>
              dispatch({
                type: "result-card",
                action: "collect",
                illustration: value.illustration,
              })
            }
          >
            <Icon name="paper" size={16} />
            Take result card
          </button>
        </div>
      )}
      {calculation.error && (
        <p className="room-launcher-error" role="alert">
          {calculation.error}
        </p>
      )}
      <button
        className="room-launcher-details"
        onClick={() => dispatch({ type: "interact", id: "experiment" })}
      >
        Open detailed bench
      </button>
    </section>
  );
}
