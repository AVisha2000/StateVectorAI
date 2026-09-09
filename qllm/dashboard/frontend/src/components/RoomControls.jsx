import { useRef } from "react";

// Native disclosure: tab order and Enter/Space belong to the browser. Closing
// it never changes possession, camera mode, conversation or material state.
export default function RoomControls({ children, actions, instructions, apiRef }) {
  const details = useRef(null), summary = useRef(null);
  const close = (restoreSummary = false) => {
    if (!details.current?.open) return;
    details.current.open = false;
    if (restoreSummary) summary.current?.focus();
  };
  return (
    <details
      className="meeting-room-controls"
      ref={details}
      onToggle={(event) => { if (event.currentTarget.open) apiRef.current?.release(); }}
      onKeyDown={(event) => {
        if (event.key !== "Escape" || !details.current?.open) return;
        event.preventDefault();
        event.stopPropagation();
        close(true);
      }}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) close();
      }}
    >
      <summary ref={summary} aria-label="Room controls">Controls</summary>
      <div className="meeting-controls-panel" onClick={(event) => {
        if (!event.target.closest("button")) return;
        close();
        // Dispatch may schedule reader/compose focus next frame; do not
        // compete with it. This only avoids focus on a now-hidden button.
        if (details.current?.contains(document.activeElement)) apiRef.current?.focus();
      }}>
        {children}
        <p>{instructions}</p>
        <p className="meeting-controls-section">Around the room</p>
        {actions}
        <button className="meeting-controls-return" onClick={() => { close(); apiRef.current?.focus(); }}>
          Back to the room
        </button>
      </div>
    </details>
  );
}
