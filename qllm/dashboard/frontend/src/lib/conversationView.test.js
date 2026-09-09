import test from "node:test";
import assert from "node:assert/strict";
import { conversationEntries, roomConversationMode, conversationMaterialShortcut } from "./conversationView.js";
import { initialMeeting, meetingEvent, meetingRoomFocus } from "./meetingInteraction.js";
import { studiosForWorld } from "./researchStudios.js";

test("empty and introductory dialogue preserve their exact contents", () => {
  assert.deepEqual(conversationEntries([], true), []);
  const message = Object.freeze({ role: "agent", text: "Scripted preview — no model connected." });
  assert.deepEqual(conversationEntries([message], true), [{ message, index: 0 }]);
});

test("latest exchange includes the human question and every following response, without shortening", () => {
  const messages = Object.freeze([
    { role: "agent", text: "Introduction" },
    { role: "human", text: "Old question" },
    { role: "agent", text: "Old answer" },
    { role: "human", text: "x".repeat(4000), quote: "Complete quotation" },
    { role: "agent", text: "Answer and limitation" },
    { role: "calculation", text: "Browser toy only", trial: { parameter: 45 } },
    { role: "agent", text: "Scripted observation" },
  ].map(Object.freeze));
  const entries = conversationEntries(messages, true);
  assert.deepEqual(entries.map((entry) => entry.index), [3, 4, 5, 6]);
  entries.forEach(({ message, index }) => assert.equal(message, messages[index]));
  assert.equal(entries[0].message.text.length, 4000);
  assert.equal(entries[0].message.quote, "Complete quotation");
  assert.equal(entries[2].message.trial, messages[5].trial);
  assert.deepEqual(conversationEntries(messages).map((entry) => entry.index), [0, 1, 2, 3, 4, 5, 6]);
});

test("without a human question the latest room response is shown, and history remains available", () => {
  const messages = [{ role: "agent", text: "Welcome" }, { role: "agent", text: "Take this note" }];
  assert.deepEqual(conversationEntries(messages, true), [{ message: messages[1], index: 1 }]);
  assert.equal(conversationEntries(messages).length, 2);
});

test("presentation switching cannot mutate a real visit's draft, carried note, or transcript", () => {
  const studio = studiosForWorld({ mode: "fixture", agents: [] }).find((item) => item.id === "physics");
  let state = initialMeeting(studio);
  state = meetingEvent(state, studio, { type: "note", action: "take" });
  state = meetingEvent(state, studio, { type: "send", text: "What assumptions should we check?" });
  state = meetingEvent(state, studio, { type: "patch", value: { draft: "An unfinished follow-up" } });
  const before = structuredClone(state);
  assert.equal(conversationEntries(state.messages, true).at(-1).message, state.messages.at(-1));
  assert.equal(conversationEntries(state.messages).length, state.messages.length);
  assert.deepEqual(state, before);
  assert.equal(state.noteHeld, true);
  assert.equal(state.draft, "An unfinished follow-up");
});

test("compact dialogue and full history both keep first-person conversation ownership", () => {
  const meeting = { material: null, launcherActive: false };
  assert.equal(roomConversationMode("first-person", false, meeting), "dialogue");
  assert.equal(roomConversationMode("first-person", false, meeting, true), "history");
  for (const history of [false, true]) {
    assert.equal(roomConversationMode("first-person", true, meeting, history), null);
    assert.equal(roomConversationMode("overview", false, meeting, history), null);
    assert.equal(roomConversationMode("first-person", false, { ...meeting, launcherActive: true }, history), null);
    for (const material of ["paper", "result", "board", "experiment"])
      assert.equal(roomConversationMode("first-person", false, { ...meeting, material }, history), null);
  }
});

test("opening a browser trial keeps history in the same conversation until explicit return", () => {
  const studio = studiosForWorld({ mode: "fixture", agents: [] }).find((item) => item.id === "physics");
  const before = { ...initialMeeting(studio), draft: "Keep this thought", noteHeld: true };
  const event = { type: "trial-open" };
  const after = meetingEvent(before, studio, event);
  const focused = meetingRoomFocus(false, event, before.material, after.material);
  assert.equal(roomConversationMode("first-person", focused, after, true), "history");
  assert.equal(roomConversationMode("first-person", focused, after, false), "history");
  assert.ok(after.messages.at(-1).trial);
  assert.equal(after.draft, before.draft);
  assert.equal(after.noteHeld, true);
  assert.equal(roomConversationMode("first-person", true, after, false), null);
  const next = meetingEvent(after, studio, { type: "send", text: "Which assumption matters?" });
  assert.equal(roomConversationMode("first-person", false, next, false), "dialogue");
});

test("the dialogue material shortcut keeps every carried result and note reader reachable", () => {
  const meeting = { heldResultId: 7, resultCards: [{ id: 7 }], noteHeld: false, noteOffered: false };
  assert.deepEqual(conversationMaterialShortcut(meeting, true), {
    label: "Read carried result", shortLabel: "Read result", event: { type: "result-card", action: "read" },
  });
  assert.deepEqual(conversationMaterialShortcut(meeting, false).event, { type: "note", action: "offer" });
  assert.deepEqual(conversationMaterialShortcut({ ...meeting, heldResultId: 99 }, true).event, { type: "note", action: "offer" });
  assert.deepEqual(conversationMaterialShortcut({ ...meeting, heldResultId: null, noteHeld: true }, true).event, { type: "note", action: "read" });
  assert.deepEqual(conversationMaterialShortcut({ ...meeting, heldResultId: null, noteOffered: true }, true).event, { type: "note", action: "decline" });
});
