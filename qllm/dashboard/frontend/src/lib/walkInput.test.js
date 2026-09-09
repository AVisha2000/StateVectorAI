import test from "node:test";
import assert from "node:assert/strict";
import { createWalkInput } from "./walkInput.js";
import { stepVisitor, walkClearance } from "./studioWalk.js";
import { VISITOR_SPAWN } from "./studioLayout.js";

test("independent pointer holds combine with keyboard without doubling an axis", () => {
  const input = createWalkInput(), keys = new Set(["w"]);
  input.press(1, 1, 0); input.press(2, 0, 1);
  assert.deepEqual(input.axes(keys), { forward: 1, right: 1 });
  input.release(1);
  assert.deepEqual(input.axes(keys), { forward: 1, right: 1 }, "W remains held");
  keys.clear();
  assert.deepEqual(input.axes(keys), { forward: 0, right: 1 });
  input.press(3, 0, -1);
  assert.deepEqual(input.axes(keys), { forward: 0, right: 0 });
  input.release(2);
  assert.deepEqual(input.axes(keys), { forward: 0, right: -1 });
  input.clear();
  assert.equal(input.size, 0);
  assert.deepEqual(input.axes(keys), { forward: 0, right: 0 });
});

test("repeat, stale releases and invalid holds cannot create stuck movement", () => {
  const input = createWalkInput();
  for (const args of [[NaN,1,0],[-1,1,0],[1,NaN,0],[1,2,0],[1,0,0]])
    assert.equal(input.press(...args), false);
  assert.equal(input.size, 0);
  input.press(1, 1, 0); input.press(1, 0, -1);
  assert.equal(input.size, 1);
  input.release(27); input.release(1); input.release(1);
  assert.deepEqual(input.axes(new Set()), { forward: 0, right: 0 });
});

test("held control uses keyboard collision path at three rates and stops exactly after release", () => {
  for (const hz of [30, 60, 120]) {
    const input = createWalkInput(), keys = new Set();
    input.press(7, 1, 0);
    let pointer = { ...VISITOR_SPAWN }, keyboard = { ...VISITOR_SPAWN };
    for (let i = 0; i < hz; i++) {
      const axes = input.axes(keys);
      pointer = stepVisitor(pointer, .75, axes.forward, axes.right, 1 / hz);
      keyboard = stepVisitor(keyboard, .75, 1, 0, 1 / hz);
      assert.deepEqual(pointer, keyboard);
      assert.ok(walkClearance(pointer) >= -1e-7);
    }
    assert.ok(Math.hypot(pointer.x - VISITOR_SPAWN.x, pointer.z - VISITOR_SPAWN.z) > .1);
    input.release(7);
    const stopped = { ...pointer };
    for (let i = 0; i < hz; i++) {
      const axes = input.axes(keys);
      pointer = stepVisitor(pointer, .75, axes.forward, axes.right, 1 / hz);
    }
    assert.deepEqual(pointer, stopped);
  }
});
