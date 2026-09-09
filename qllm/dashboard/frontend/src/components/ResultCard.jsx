import { resultCardView } from "../lib/resultCards.js";
import Icon from "./WorldIcon.jsx";
import "../result-card.css";

export default function ResultCard({
  card,
  pinned = false,
  onDiscuss,
  onCarry,
}) {
  const view = resultCardView(card);
  const all = [...card.points, ...card.baselinePoints];
  const xmin = Math.min(...all.map((p) => p.x)),
    xmax = Math.max(...all.map((p) => p.x));
  const ymin = Math.min(0, ...all.map((p) => p.y)),
    ymax = Math.max(0.001, ...all.map((p) => p.y));
  const path = (points) =>
    points
      .map(
        (p, i) =>
          `${i ? "L" : "M"}${20 + ((p.x - xmin) / (xmax - xmin || 1)) * 280},${106 - ((p.y - ymin) / (ymax - ymin)) * 90}`,
      )
      .join(" ");
  return (
    <article className="result-card" aria-label={`Result card ${card.id}`}>
      <header>
        <span>BROWSER TOY / CARD {String(card.id).padStart(2, "0")}</span>
        <small>{pinned ? "Pinned to board" : "Collected this visit"}</small>
      </header>
      <h3>{view.title}</h3>
      <p>{view.location}</p>
      <div className="result-card-value">
        <strong>{view.own}</strong>
        <span>{view.label}</span>
      </div>
      <dl>
        <div>
          <dt>Applied input</dt>
          <dd>{view.parameter}</dd>
        </div>
        <div>
          <dt>Baseline input</dt>
          <dd>{view.baselineParameter}</dd>
        </div>
        <div>
          <dt>Baseline output</dt>
          <dd>{view.baseline}</dd>
        </div>
        <div>
          <dt>Difference</dt>
          <dd>{view.delta}</dd>
        </div>
      </dl>
      <svg
        viewBox="0 0 320 126"
        role="img"
        aria-label={`Stored ${view.title} comparison; solid candidate, dashed baseline. ${view.location}.`}
      >
        <path
          d={path(card.baselinePoints)}
          fill="none"
          stroke="#8b9777"
          strokeWidth="2"
          strokeDasharray="5 4"
        />
        <path
          d={path(card.points)}
          fill="none"
          stroke="#326e55"
          strokeWidth="2.5"
        />
        <text x="20" y="122">
          {xmin.toFixed(1)}
        </text>
        <text x="300" y="122" textAnchor="end">
          {xmax.toFixed(1)}
        </text>
      </svg>
      <p className="result-card-key">Solid: collected run · Dashed: baseline</p>
      <p className="result-card-disclosure">
        Stored browser calculation. Synthetic inputs, not a research result.
      </p>
      <footer>
        <button onClick={() => onDiscuss(card)}>
          <Icon name="send" size={15} />
          Discuss this result
        </button>
        {onCarry && (
          <button onClick={() => onCarry(card.id)}>
            <Icon name="paper" size={15} />
            Carry this card
          </button>
        )}
      </footer>
    </article>
  );
}
