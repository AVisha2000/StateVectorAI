import test from "node:test";
import assert from "node:assert/strict";
import {
  initialMeeting,
  meetingEvent,
  stepVisitor,
  advanceGestureClock,
} from "./meetingInteraction.js";
const studio = {
  name: "Lyra Vale",
  title: "Compare boundaries",
  label: "Physics",
};
test("manual pause freezes gestures and resume never rewinds or replays a settled action", () => {
  let clock = { elapsed: 0, id: -1, start: 0, instant: false, age: 0 };
  for (let i = 0; i < 10; i++)
    clock = advanceGestureClock(clock, 0.05, 1, false, false);
  const age = clock.age;
  for (let i = 0; i < 20; i++)
    clock = advanceGestureClock(clock, 0.05, 1, true, false);
  assert.equal(clock.age, age);
  clock = advanceGestureClock(clock, 0.05, 1, false, false);
  assert.ok(Math.abs(clock.age - age - 0.05) < 1e-10);
  clock = advanceGestureClock(clock, 0.05, 2, true, false);
  assert.equal(clock.age, 10);
  clock = advanceGestureClock(clock, 0.05, 2, false, false);
  assert.equal(clock.age, 10);
  clock = advanceGestureClock(clock, 0.05, 3, false, true);
  assert.equal(clock.age, 10);
});
test("room actions preserve drafts, board and researcher identity", () => {
  const original = {
    ...initialMeeting(studio),
    draft: "An unfinished thought",
  };
  const changed = meetingEvent(original, studio, {
    type: "interact",
    id: "board",
  });
  assert.equal(changed.draft, original.draft);
  assert.equal(changed.material, "board");
  assert.equal(changed.board.question, studio.title);
  assert.equal(original.visited.length, 0);
});
test("taking a note carries it without forcing a reader; other actions preserve ownership", () => {
  const paper = meetingEvent(initialMeeting(studio), studio, {
    type: "interact",
    id: "paper",
  });
  const interrupted = meetingEvent(paper, studio, {
    type: "interact",
    id: "board",
  });
  assert.equal(paper.noteHeld, true);
  assert.equal(paper.material, null);
  assert.equal(interrupted.noteHeld, true);
  assert.equal(interrupted.material, "board");
  assert.equal(interrupted.noteRevision, paper.noteRevision);
  assert.equal(
    meetingEvent(interrupted, studio, {
      type: "paper-ready",
      id: paper.gesture.id,
    }).material,
    "board",
  );
});
test("quoted discussion preserves source context and unsent draft for shortcuts", () => {
  let state = { ...initialMeeting(studio), draft: "My own question" };
  state = meetingEvent(state, studio, {
    type: "quote",
    text: "Which control stays fixed?",
  });
  state = meetingEvent(state, studio, {
    type: "send",
    text: "Explain this passage",
  });
  assert.equal(state.draft, "My own question");
  assert.equal(state.messages.at(-2).quote, "Which control stays fixed?");
  assert.match(state.messages.at(-1).text, /scripted/);
  state = meetingEvent(state, studio, { type: "send" });
  assert.equal(state.draft, "");
});
test("read/lower/return/re-take preserve drafts and passages without duplicate pickup", () => {
  let state = {
    ...initialMeeting(studio),
    draft: "unfinished",
    quote: "a passage",
    experiment: "a proposal",
  };
  state = meetingEvent(state, studio, { type: "note", action: "read" });
  assert.equal(state.material, "paper");
  assert.equal(state.noteRevision, 1);
  state = meetingEvent(state, studio, { type: "note", action: "read" });
  assert.equal(state.noteRevision, 1);
  state = meetingEvent(state, studio, { type: "close-material" });
  assert.equal(state.noteHeld, true);
  assert.equal(state.material, null);
  state = meetingEvent(state, studio, { type: "note", action: "return" });
  assert.equal(state.noteHeld, false);
  assert.equal(state.noteRevision, 2);
  state = meetingEvent(state, studio, { type: "interact", id: "paper" });
  assert.equal(state.noteHeld, true);
  assert.equal(state.noteRevision, 3);
  assert.equal(state.material, null);
  assert.equal(state.draft, "unfinished");
  assert.equal(state.quote, "a passage");
  assert.equal(state.experiment, "a proposal");
});
test("WASD movement is normalized, bounded, and blocked by the table", () => {
  const start = { x: 0, z: 1.8 };
  const straight = stepVisitor(start, 0, 1, 0, 0.03);
  const diagonal = stepVisitor(start, 0, 1, 1, 0.03);
  assert.ok(
    Math.abs(
      Math.hypot(diagonal.x, diagonal.z - start.z) - (start.z - straight.z),
    ) < 1e-10,
  );
  let position = start;
  for (let i = 0; i < 1000; i++)
    position = stepVisitor(position, 0, 1, 0, 0.05);
  assert.ok(position.z >= 0.92);
  for (let i = 0; i < 1000; i++)
    position = stepVisitor(position, 0, -1, 1, 0.05);
  assert.ok(Math.hypot(position.x, position.z) <= 2.05 + 1e-10);
});
