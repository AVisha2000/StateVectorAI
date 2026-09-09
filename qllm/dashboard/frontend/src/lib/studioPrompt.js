import { ROOM_ACTIONS } from "./meetingInteraction.js";

// Presentation only. Picking, proximity and dispatch remain owned by the room.
export function studioPrompt(mode, target, noteHeld, resultHeld, seated) {
  if (mode !== "first-person") return null;
  if (target === "offered-note") return "Take offered note";
  if (target === "seat") return seated ? "Stand up" : "Take a seat";
  if (target === "board" && resultHeld) return "Pin result to board";
  if (target === "held-result") return "Read result card";
  if (target === "return-note" && noteHeld) return "Put note on table";
  if (target === "paper" && noteHeld) return "Read held note";
  if (target === "launcher") return "Use launcher";
  return ROOM_ACTIONS.find((action) => action.id === target)?.label ?? null;
}
