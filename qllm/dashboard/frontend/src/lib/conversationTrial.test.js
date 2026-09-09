import test from "node:test";
import assert from "node:assert/strict";
import { newConversationTrial, editConversationTrial, completeConversationTrial, calculationEvent, calculationForRoom } from "./conversationTrial.js";
import { DEMO_MODELS, simulateStudio } from "./studioSimulation.js";
import { initialMeeting, meetingEvent, meetingRoomFocus } from "./meetingInteraction.js";
import { createStudioCalculation } from "./studioCalculation.js";

const studio = { id: "physics", title: "Test", name: "Lyra", label: "Physics" };
test("all six conversation controls use the existing model bounds and reject invalid inputs", () => {
  for (const [id, model] of Object.entries(DEMO_MODELS)) {
    const trial = newConversationTrial(id);
    assert.equal(trial.parameter, model.initial);
    assert.equal(trial.result, null);
    for (const parameter of [NaN, Infinity, -Infinity, model.min - 1, model.max + 1])
      assert.equal(editConversationTrial(trial, parameter), trial);
    for (const parameter of [model.min, model.max])
      assert.equal(editConversationTrial(trial, parameter).parameter, parameter);
  }
  assert.equal(newConversationTrial("not-a-model"), null);
});

test("opening and editing a trial preserve material, proposal, draft, quotation and board", () => {
  const before = { ...initialMeeting(studio), material: "paper", draft: "Keep typing", quote: "A passage", experiment: "My actual proposal", noteHeld: true };
  const opened = meetingEvent(before, studio, { type: "trial-open" });
  assert.equal(opened.messages.length, before.messages.length + 1);
  assert.match(opened.messages.at(-1).text, /built-in browser experiment, separate from your research proposal/);
  const edited = meetingEvent(opened, studio, { type: "trial-input", index: 1, parameter: 60 });
  assert.equal(edited.messages[1].trial.parameter, 60);
  for (const field of ["material", "draft", "quote", "experiment", "board", "noteHeld", "resultCards", "illustration"])
    assert.equal(edited[field], before[field]);
  assert.equal(opened.messages[1].trial.parameter, 45);
  assert.equal(meetingRoomFocus(true, { type: "trial-open" }, "paper", "paper"), false);
  assert.equal(meetingRoomFocus(false, { type: "result" }, null, null), false);
});

test("completed trial results are copied, keep their fired input, and survive later room/card runs", () => {
  for (const [id, model] of Object.entries(DEMO_MODELS)) {
    const source = simulateStudio(id, model.min);
    const trial = completeConversationTrial(newConversationTrial(id, model.max), source);
    assert.equal(trial.parameter, model.max);
    assert.equal(trial.result.parameter, model.min);
    assert.notEqual(trial.result.points, source.points);
    const y = trial.result.points[60].y;
    source.points[60].y = 999;
    assert.equal(trial.result.points[60].y, y);
    assert.ok(Object.isFrozen(trial.result.points[60]));
    assert.match(trial.result.provenance, /No research agent/);
  }
  let state = meetingEvent(initialMeeting(studio), studio, { type: "trial-open" });
  const result = simulateStudio("physics", 60);
  state = meetingEvent(state, studio, calculationEvent(result, { trialIndex: 1 }));
  const saved = state.messages[1].trial.result;
  assert.equal(state.messages.length, 2);
  assert.equal(state.illustration.result, result);
  assert.equal(state.illustration.flightId, 1);
  state = meetingEvent(state, studio, { type: "trial-open" });
  state = meetingEvent(state, studio, calculationEvent(simulateStudio("physics", 30), { trialIndex: 2 }));
  state = meetingEvent(state, studio, calculationEvent(simulateStudio("physics", 45), "meadow"));
  assert.equal(state.messages[1].trial.result, saved);
  assert.equal(state.messages[2].trial.result.parameter, 30);
  state = meetingEvent(state, studio, { type: "result-card", action: "collect", illustration: { result: saved } });
  assert.equal(state.resultCards[0].parameter, 60);
  assert.equal(state.heldResultId, 1);
});

test("invalid or wrong-model completions do not replace a conversation result", () => {
  const trial = completeConversationTrial(newConversationTrial("physics"), simulateStudio("physics", 60));
  for (const result of [{}, simulateStudio("biology", .5), { ...simulateStudio("physics", 30), points: [] }])
    assert.equal(completeConversationTrial(trial, result), trial);
});

test("one worker owns conversation and room runs; error context survives for retry and does not leak to a later bench", () => {
  const workers = [];
  let presentation = { busy: false, error: "" }, state = meetingEvent(initialMeeting(studio), studio, { type: "trial-open" });
  const owner = createStudioCalculation({
    createWorker: () => { const worker = { postMessage() {}, terminate() {} }; workers.push(worker); return worker; },
    onState: (next) => {
      presentation = next;
      if (Number.isInteger(next.context?.trialIndex))
        state = meetingEvent(state, studio, { type: "trial-status", index: next.context.trialIndex, error: next.error });
    },
    onResult: (result, context) => { state = meetingEvent(state, studio, calculationEvent(result, context)); },
  });
  owner.start("physics", 60, { trialIndex: 1 });
  assert.equal(owner.start("physics", 30, "meadow"), false);
  workers[0].onerror();
  assert.equal(presentation.context.trialIndex, 1);
  assert.match(presentation.error, /Try again/);
  assert.match(state.messages[1].trial.error, /Try again/);
  assert.equal(state.messages[1].trial.result, null);
  // Both LocalIllustration and RoomLauncher receive this exact derived view
  // from ResearchSession, even when opened before any subsequent run starts.
  state = meetingEvent(state, studio, { type: "interact", id: "experiment" });
  assert.equal(calculationForRoom(presentation).error, "");
  state = meetingEvent(state, studio, { type: "launcher", action: "open" });
  assert.equal(calculationForRoom(presentation).error, "");
  assert.match(state.messages[1].trial.error, /Try again/);
  owner.start("physics", 30, null);
  workers[1].onerror();
  assert.match(calculationForRoom(presentation).error, /Try again/);
  assert.match(state.messages[1].trial.error, /Try again/);
  owner.start("physics", 60, { trialIndex: 1 });
  assert.equal(state.messages[1].trial.error, "");
  workers[2].onmessage({ data: { result: simulateStudio("physics", 60) } });
  assert.equal(state.messages[1].trial.result.parameter, 60);
  assert.equal(presentation.busy, false);
  owner.start("physics", 30, null);
  assert.equal(presentation.context, null);
  owner.dispose();
  workers[3].onmessage({ data: { result: simulateStudio("physics", 30) } });
  assert.equal(state.messages[1].trial.result.parameter, 60);
  assert.equal(owner.start("physics", 30, { trialIndex: 1 }), false);
});
