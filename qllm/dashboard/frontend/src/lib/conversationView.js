import { heldResultCard } from "./resultCards.js";

// Presentation only: preserve original message objects and indices so inline
// experiment actions still address the canonical visit transcript.
export function roomConversationMode(view, roomFocused, meeting, fullHistory = false) {
  if (view !== "first-person" || roomFocused || meeting.material || meeting.launcherActive) return null;
  // An editable experiment needs the workspace, including when reopening a
  // visit after walking; never squeeze its controls into the reply viewport.
  return fullHistory || conversationEntries(meeting.messages || [], true).some(({ message }) => message.trial)
    ? "history" : "dialogue";
}

export function conversationEntries(messages, latestExchange = false) {
  let start = 0;
  if (latestExchange && messages.length) {
    start = messages.findLastIndex((message) => message.role === "human");
    if (start < 0) start = messages.length - 1;
  }
  return messages.slice(start).map((message, offset) => ({ message, index: start + offset }));
}

export function conversationMaterialShortcut(meeting, inRoomConversation) {
  if (inRoomConversation && heldResultCard(meeting)) return {
    label: "Read carried result", shortLabel: "Read result",
    event: { type: "result-card", action: "read" },
  };
  if (meeting.noteHeld) return {
    label: "Read my working note", shortLabel: "Read note",
    event: { type: "note", action: "read" },
  };
  if (meeting.noteOffered) return {
    label: "Leave note on table", shortLabel: "Leave note",
    event: { type: "note", action: "decline" },
  };
  return {
    label: "Show me a working note", shortLabel: "Working note",
    event: { type: "note", action: "offer" },
  };
}
