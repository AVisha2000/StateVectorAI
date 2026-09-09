import { useRef, useState } from "react";
import Icon from "./WorldIcon.jsx";
import ResultCard from "./ResultCard.jsx";
import { resultCardView } from "../lib/resultCards.js";

export default function SharedBoard({ board, dispatch, resultCards = [] }) {
  const drawing = useRef(null),
    [preview, setPreview] = useState(null);
  const point = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [
      Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    ];
  };
  const finish = () => {
    if (!drawing.current) return;
    const stroke = drawing.current;
    drawing.current = null;
    setPreview(null);
    if (stroke.length > 1)
      dispatch({
        type: "board",
        value: { strokes: [...board.strokes, stroke].slice(-80) },
      });
  };
  return (
    <div className="shared-board">
      {resultCards.length > 0 && (
        <section
          className="result-card-collection"
          aria-label="Collected result cards"
        >
          <h4>Things we have tried</h4>
          <p>
            Collected snapshots stay unchanged when you run or scrub another
            experiment.
          </p>
          {[...resultCards].reverse().map((card) => (
            <ResultCard
              key={card.id}
              card={card}
              pinned={board.pinnedResultIds?.includes(card.id)}
              onDiscuss={(item) =>
                dispatch({
                  type: "quote",
                  kind: "observation",
                  text: resultCardView(item).quote,
                })
              }
              onCarry={(id) =>
                dispatch({ type: "result-card", action: "carry", id })
              }
            />
          ))}
        </section>
      )}
      <div className="shared-board-fields">
        {[
          ["question", "Our question", "What are we trying to understand?"],
          ["assumption", "The assumption", "What might not be true?"],
          ["test", "The smallest test", "What would change our mind?"],
        ].map(([id, label, placeholder], i) => (
          <label key={id}>
            <span>
              0{i + 1} / {label}
            </span>
            <textarea
              aria-label={label}
              rows={2}
              maxLength={500}
              placeholder={placeholder}
              value={board[id]}
              onChange={(event) =>
                dispatch({ type: "board", value: { [id]: event.target.value } })
              }
            />
          </label>
        ))}
      </div>
      <div className="shared-sketch">
        <div>
          <span>DRAW A CONNECTION</span>
          <button
            disabled={!board.strokes.length}
            onClick={() =>
              dispatch({
                type: "board",
                value: { strokes: board.strokes.slice(0, -1) },
              })
            }
          >
            <Icon name="reset" size={14} />
            Undo stroke
          </button>
        </div>
        <svg
          viewBox="0 0 600 200"
          preserveAspectRatio="none"
          role="img"
          aria-label="Shared sketch. Draw with a mouse or touch. The text fields provide a keyboard alternative."
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            drawing.current = [point(event)];
            setPreview(drawing.current);
          }}
          onPointerMove={(event) => {
            if (!drawing.current || drawing.current.length >= 600) return;
            drawing.current = [...drawing.current, point(event)];
            setPreview(drawing.current);
          }}
          onPointerUp={finish}
          onPointerCancel={finish}
        >
          <defs>
            <pattern
              id="meeting-dots"
              width="20"
              height="20"
              patternUnits="userSpaceOnUse"
            >
              <circle cx="2" cy="2" r="0.7" fill="#668774" />
            </pattern>
          </defs>
          <rect width="600" height="200" fill="url(#meeting-dots)" />
          {[...board.strokes, ...(preview ? [preview] : [])].map(
            (stroke, i) => (
              <polyline
                key={i}
                points={stroke
                  .map(([x, y]) => `${x * 600},${y * 200}`)
                  .join(" ")}
                fill="none"
                stroke="#eed7a6"
                strokeWidth="2.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ),
          )}
          {!board.strokes.length && !preview && (
            <text
              x="300"
              y="108"
              textAnchor="middle"
              fill="#b1c6b4"
              fontSize="14"
            >
              An arrow, a curve, a half-formed thought. Start here.
            </text>
          )}
        </svg>
      </div>
      <footer>
        <span>Your words and sketch appear on the board in the room.</span>
        <button
          disabled={!board.assumption.trim()}
          onClick={() =>
            dispatch({
              type: "send",
              text: "Let’s challenge the assumption on our board.",
            })
          }
        >
          Challenge this assumption <Icon name="arrow" size={15} />
        </button>
      </footer>
    </div>
  );
}
