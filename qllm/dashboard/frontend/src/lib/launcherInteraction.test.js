import test from "node:test";
import assert from "node:assert/strict";
import {
  boundedLaunchAngle,
  draggedLaunchAngle,
  keyedLaunchAngle,
} from "./launcherInteraction.js";
import {
  initialMeeting,
  meetingEvent,
  meetingRoomFocus,
} from "./meetingInteraction.js";
import { simulateStudio } from "./studioSimulation.js";
const studio = { id: "physics", name: "Lyra Vale", title: "An arc" };
test("vertical drag is camera-independent, bounded, finite and uses the captured start angle", () => {
  assert.equal(draggedLaunchAngle(45, 200, 140), 60);
  assert.equal(draggedLaunchAngle(45, 200, 260), 30);
  assert.equal(draggedLaunchAngle(45, 200, -10000), 70);
  assert.equal(draggedLaunchAngle(45, 200, 10000), 20);
  assert.equal(draggedLaunchAngle(45, NaN, 200), 45);
  assert.equal(boundedLaunchAngle(Infinity), 45);
  for (const segments of [1, 5, 20, 100]) {
    let angle;
    for (let i = 0; i <= segments; i++)
      angle = draggedLaunchAngle(45, 200, 200 - (60 * i) / segments);
    assert.equal(angle, 60);
  }
});
test("key alternatives use the same bounds with distinct navigation/fire keys", () => {
  assert.equal(keyedLaunchAngle(45, "ArrowUp"), 46);
  assert.equal(keyedLaunchAngle(45, "ArrowDown", true), 40);
  assert.equal(keyedLaunchAngle(69, "ArrowUp", true), 70);
  assert.equal(keyedLaunchAngle(21, "ArrowDown", true), 20);
  assert.equal(keyedLaunchAngle(45, "Home"), 20);
  assert.equal(keyedLaunchAngle(45, "End"), 70);
  for (const key of ["w", "r", "q", " ", "Enter", "Escape"])
    assert.equal(keyedLaunchAngle(45, key), null);
});
test("using a launcher frees the hand without deleting artifacts; draft aim never mutates committed output", () => {
  const result = simulateStudio("physics", 30);
  let state = {
    ...initialMeeting(studio),
    draft: "unfinished",
    noteHeld: true,
    illustration: { result, parameter: 30, flightId: 4 },
  };
  state = meetingEvent(state, studio, {
    type: "result-card",
    action: "collect",
    illustration: state.illustration,
  });
  const card = state.resultCards[0];
  state = meetingEvent(state, studio, { type: "launcher", action: "open" });
  assert.equal(state.launcherActive, true);
  assert.equal(state.noteHeld, false);
  assert.equal(state.heldResultId, null);
  state = meetingEvent(state, studio, {
    type: "launcher",
    action: "aim",
    angle: 60,
  });
  assert.equal(state.illustration.parameter, 60);
  assert.equal(state.illustration.result, result);
  assert.equal(state.illustration.flightId, 4);
  assert.equal(state.resultCards[0], card);
  assert.equal(card.parameter, 30);
  assert.equal(state.draft, "unfinished");
  assert.equal(
    meetingEvent(state, studio, { type: "launcher", action: "close" })
      .launcherActive,
    false,
  );
  assert.equal(
    meetingEvent(state, studio, { type: "interact", id: "board" })
      .launcherActive,
    false,
  );
  assert.equal(
    meetingEvent(state, studio, { type: "result", result }).launcherActive,
    true,
  );
  assert.equal(
    meetingRoomFocus(false, { type: "launcher", action: "open" }),
    true,
  );
});
test("non-physics studio and non-finite aim events cannot alter the tool", () => {
  const state = initialMeeting(studio);
  assert.deepEqual(
    meetingEvent(
      state,
      { ...studio, id: "biology" },
      { type: "launcher", action: "open" },
    ),
    state,
  );
  assert.deepEqual(
    meetingEvent(state, studio, {
      type: "launcher",
      action: "aim",
      angle: NaN,
    }),
    state,
  );
});
