import { useEffect, useRef } from "react";

export default function WalkPad({ apiRef }) {
  const captures = useRef(new Map());
  const end = (event) => {
    apiRef.current?.endWalk(event.pointerId);
    captures.current.delete(event.pointerId);
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };
  useEffect(() => () => {
    for (const [id, button] of captures.current) {
      apiRef.current?.endWalk(id);
      if (button.hasPointerCapture(id)) button.releasePointerCapture(id);
    }
    captures.current.clear();
  }, [apiRef]);
  return <div className="meeting-touch-walk" role="group" aria-label="Movement controls. Hold an arrow to walk">
    {[
      [1, 0, "Walk forward", "↑"], [0, -1, "Walk left", "←"],
      [-1, 0, "Walk backward", "↓"], [0, 1, "Walk right", "→"],
    ].map(([forward, right, label, icon]) => <button key={label} type="button" aria-label={label}
      title={`${label} — hold to move`}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        // Avoid stealing canvas keyboard focus while a second pointer looks.
        event.preventDefault();
        try {
          event.currentTarget.setPointerCapture(event.pointerId);
        } catch {
          apiRef.current?.move(forward, right);
          return;
        }
        captures.current.set(event.pointerId, event.currentTarget);
        apiRef.current?.beginWalk(event.pointerId, forward, right);
      }}
      onPointerUp={end} onPointerCancel={end} onLostPointerCapture={end}
      onContextMenu={(event) => event.preventDefault()}
      onClick={(event) => {
        // Pointer activation was already consumed at down; keyboard and AT
        // activation retain a single ordinary step, without a hold requirement.
        if (event.detail === 0) apiRef.current?.move(forward, right);
      }}
    >{icon}</button>)}
  </div>;
}
