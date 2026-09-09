import test from "node:test";
import assert from "node:assert/strict";
import {
  captureResultCard,
  resultCardView,
  heldResultCard,
  canPinResult,
} from "./resultCards.js";
import { DEMO_MODELS, simulateStudio } from "./studioSimulation.js";
import {
  initialMeeting,
  meetingEvent,
  meetingRoomFocus,
} from "./meetingInteraction.js";

const studio = {
  id: "chemistry",
  name: "Mira Chen",
  title: "One change",
  label: "Chemistry",
};
const run = (parameter = 0.8, sampleIndex = 24) => ({
  result: simulateStudio("chemistry", parameter),
  sampleIndex,
});
const collect = (state, illustration = run()) =>
  meetingEvent(state, studio, {
    type: "result-card",
    action: "collect",
    illustration,
  });
const act = (state, action, id) =>
  meetingEvent(state, studio, { type: "result-card", action, id });

for (const [id, model] of Object.entries(DEMO_MODELS)) {
  test(`${id}: card reports the stored calculation with explicit toy provenance`, () => {
    const result = simulateStudio(id, model.max);
    const card = captureResultCard({
      result,
      parameter: model.min,
      sampleIndex: 24,
    });
    const index = id === "physics" ? 120 : 24;
    assert.equal(card.parameter, model.max);
    assert.equal(card.baselineParameter, model.initial);
    assert.equal(card.sampleIndex, index);
    assert.equal(card.own, result.points[index][id === "physics" ? "x" : "y"]);
    assert.equal(
      card.baseline,
      result.baseline[index][id === "physics" ? "x" : "y"],
    );
    const view = resultCardView(card);
    assert.ok(
      view.quote.includes(view.own) && view.quote.includes(view.baseline),
    );
    assert.match(
      view.quote,
      /Stored calculation, synthetic inputs, not a research result/,
    );
    if (id === "physics")
      assert.equal(view.location, "Computed flight endpoint");
  });
}

test("collection deep-copies selected worker samples; replay and later runs cannot rewrite a card", () => {
  const illustration = run();
  const card = captureResultCard(illustration);
  const snapshot = JSON.stringify(card);
  illustration.sampleIndex = 120;
  illustration.result.points[24].y = 0.999;
  illustration.result.baseline[24].y = 0.111;
  illustration.result.parameter = 1.5;
  assert.equal(JSON.stringify(card), snapshot);
  assert.equal(card.sampleIndex, 24);
  assert.ok(
    Object.isFrozen(card) &&
      Object.isFrozen(card.points) &&
      Object.isFrozen(card.points[24]),
  );
  assert.ok(Object.isFrozen(card.baselinePoints[24]));
  assert.throws(() => {
    card.points[24].y = 1;
  }, TypeError);
});

test("malformed or mismatched worker output cannot become a result card", () => {
  assert.equal(captureResultCard(null), null);
  for (const corrupt of [
    (r) => {
      r.id = "unknown";
    },
    (r) => {
      r.parameter = Infinity;
    },
    (r) => {
      r.baselineParameter = 10;
    },
    (r) => {
      r.points.pop();
    },
    (r) => {
      r.baseline[24].y = NaN;
    },
    (r) => {
      r.points[24].x = -1;
    },
    (r) => {
      r.baseline[24].x += 0.001;
    },
    (r) => {
      r.points[24].y = 1.2;
    },
  ]) {
    const illustration = run();
    corrupt(illustration.result);
    assert.equal(captureResultCard(illustration), null);
  }
  assert.deepEqual(
    collect(initialMeeting(studio), { result: simulateStudio("physics", 50) }),
    initialMeeting(studio),
  );
});

test("one hand holds a result or note; switching and subsequent runs preserve the full collection", () => {
  const original = {
    ...initialMeeting(studio),
    noteHeld: true,
    draft: "Unfinished",
    material: "experiment",
  };
  const first = collect(original);
  assert.equal(first.noteHeld, false);
  assert.equal(first.noteRevision, 1);
  assert.equal(first.material, null);
  assert.equal(heldResultCard(first).id, 1);
  const second = collect(first, run(1.2, 90));
  assert.equal(second.resultCards.length, 2);
  assert.equal(second.resultCards[0], first.resultCards[0]);
  assert.equal(second.heldResultId, 2);
  assert.equal(second.draft, original.draft);
  const note = meetingEvent(second, studio, { type: "note", action: "read" });
  assert.equal(note.heldResultId, null);
  assert.equal(note.noteHeld, true);
  assert.equal(note.resultCards, second.resultCards);
  const restored = act(note, "carry", 1);
  assert.equal(restored.noteHeld, false);
  assert.equal(restored.heldResultId, 1);
  assert.equal(restored.resultCards.length, 2);
});

test("pin is idempotent; carrying unpins only that card and never touches sketches", () => {
  const board = {
    question: "A",
    assumption: "B",
    test: "C",
    strokes: [[[1, 2]]],
  };
  const first = collect({ ...initialMeeting(studio), board });
  const pinned = act(first, "pin");
  assert.deepEqual(pinned.board.pinnedResultIds, [1]);
  assert.equal(pinned.heldResultId, null);
  assert.equal(pinned.material, "board");
  assert.deepEqual(act(pinned, "pin", 1), pinned);
  assert.deepEqual(act(pinned, "pin"), pinned);
  const two = act(collect(pinned, run(1.2)), "pin");
  assert.deepEqual(two.board.pinnedResultIds, [1, 2]);
  const carrying = act(two, "carry", 1);
  assert.deepEqual(carrying.board.pinnedResultIds, [2]);
  assert.equal(carrying.board.strokes, board.strokes);
  assert.equal(carrying.resultCards, two.resultCards);
});

test("reading/lowering/storing are reversible, invalid IDs cannot discard a different carried result", () => {
  let state = collect(collect(initialMeeting(studio)), run(1.2));
  assert.deepEqual(act(state, "store", 1), state);
  assert.deepEqual(act(state, "carry", 99), state);
  state = act(state, "read");
  assert.equal(state.material, "result");
  assert.equal(state.selectedResultId, 2);
  state = meetingEvent(state, studio, { type: "close-material" });
  assert.equal(state.heldResultId, 2);
  state = act(state, "store");
  assert.equal(state.heldResultId, null);
  assert.equal(state.resultCards.length, 2);
  // A stored session is merged with defaults without recreating its artifacts.
  const returned = meetingEvent(state, studio, {
    type: "interact",
    id: "board",
  });
  assert.equal(returned.resultCards, state.resultCards);
});

test("discussion quotes the exact saved card and preserves the user's unsent draft", () => {
  let state = collect({ ...initialMeeting(studio), draft: "My question" });
  const quote = resultCardView(state.resultCards[0]).quote;
  state = meetingEvent(state, studio, {
    type: "quote",
    kind: "observation",
    text: quote,
  });
  state = meetingEvent(state, studio, {
    type: "send",
    text: "Discuss this observation",
  });
  assert.equal(state.messages.at(-2).quote, quote);
  assert.ok(state.messages.at(-1).text.includes(quote));
  assert.match(state.messages.at(-1).text, /scripted/);
  assert.equal(state.draft, "My question");
});

test("room/read focus choices and near-board pin reach are explicit", () => {
  for (const action of ["collect", "carry", "store"])
    assert.equal(
      meetingRoomFocus(false, { type: "result-card", action }),
      true,
    );
  for (const action of ["read", "pin"])
    assert.equal(
      meetingRoomFocus(true, { type: "result-card", action }),
      false,
    );
  assert.equal(
    meetingRoomFocus(false, { type: "close-material" }, "result", null),
    true,
  );
  assert.equal(canPinResult({ x: -1.12, z: 0.32 }, true), true);
  assert.equal(canPinResult({ x: -1.12, z: 0.34 }, true), false);
  assert.equal(canPinResult({ x: NaN, z: 0 }, true), false);
  assert.equal(canPinResult({ x: 2, z: 2 }, false), true);
});
