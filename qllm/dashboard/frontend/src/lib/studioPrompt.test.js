import test from "node:test";
import assert from "node:assert/strict";
import { studioPrompt } from "./studioPrompt.js";

test("first-person prompt retains every ordinary and held-item action label", () => {
  for (const [target, label] of [
    ["greet", "Talk to researcher"], ["coffee", "Have a coffee"],
    ["paper", "Take working note"], ["board", "Use shared board"],
    ["experiment", "Try an experiment"], ["launcher", "Use launcher"],
    ["held-result", "Read result card"], ["seat", "Take a seat"],
  ]) assert.equal(studioPrompt("first-person", target, false, null, false), label);
  assert.equal(studioPrompt("first-person", "seat", false, null, true), "Stand up");
  assert.equal(studioPrompt("first-person", "paper", true, null, false), "Read held note");
  assert.equal(studioPrompt("first-person", "return-note", true, null, false), "Put note on table");
  assert.equal(studioPrompt("first-person", "board", false, { id: 1 }, false), "Pin result to board");
});

test("non-game views and absent or unsupported targets never invent an interaction", () => {
  for (const mode of ["overview", "inspection", undefined])
    assert.equal(studioPrompt(mode, "greet", true, { id: 1 }, true), null);
  for (const target of [null, undefined, "unknown", "return-note"])
    assert.equal(studioPrompt("first-person", target, false, null, false), null);
});
