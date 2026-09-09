import test from "node:test";
import assert from "node:assert/strict";
import { projectile, aimAngle, flightPresentation } from "./projectileToy.js";
import { simulateStudio } from "./studioSimulation.js";
import {
  initialMeeting,
  meetingEvent,
  advanceGestureClock,
} from "./meetingInteraction.js";
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} ≠ ${b}`);
test("ideal projectile reaches analytic start, apex and landing at every allowed angle", () => {
  for (let angle = 20; angle <= 70; angle++) {
    const start = projectile(angle, 0),
      apex = projectile(angle, 0.5),
      end = projectile(angle, 1);
    assert.equal(start.x, 0);
    assert.equal(start.y, 0);
    assert.equal(end.y, 0);
    near(apex.vy, 0);
    near(apex.y, end.height);
    near(end.x, (400 * Math.sin((2 * angle * Math.PI) / 180)) / 9.81);
    const result = simulateStudio("physics", angle);
    near(result.points[120].x, end.range);
    assert.equal(result.points[120].y, 0);
  }
});
test("complementary angles share range, not apex or duration", () => {
  const low = projectile(30, 1),
    high = projectile(60, 1),
    base = projectile(45, 1);
  near(low.range, high.range);
  assert.ok(low.range < base.range);
  assert.ok(high.height > low.height);
  assert.ok(high.duration > low.duration);
});
test("aim input is bounded and has exact known direction mappings", () => {
  assert.equal(aimAngle(149, 230), 20);
  assert.equal(aimAngle(55, 50), 70);
  assert.equal(aimAngle(105, 180), 45);
  assert.equal(aimAngle(NaN, 0), 45);
  for (const angle of [NaN, Infinity, 0, 90])
    assert.throws(() => projectile(angle));
});
test("flight replay lands identically across presentation rates and preserves pause", () => {
  for (const hz of [30, 60, 120, 240]) {
    let clock = { elapsed: 0, id: 0, start: 0, instant: false, age: 0 };
    for (let i = 0; i < hz * 3; i++)
      clock = advanceGestureClock(clock, 1 / hz, 0, false, false);
    const pose = flightPresentation(60, clock.age);
    assert.ok(pose.landed);
    near(pose.x, projectile(60, 1).range);
    const frozen = advanceGestureClock(clock, 0.05, 0, true, false);
    assert.equal(frozen.age, clock.age);
  }
});
test("talking and aiming do not alter a recorded launch or its identity", () => {
  const studio = { id: "physics", name: "Lyra Vale", title: "A question" };
  let state = { ...initialMeeting(studio), draft: "Keep this thought" };
  const result = simulateStudio("physics", 30);
  state = meetingEvent(state, studio, { type: "result", result });
  const id = state.illustration.flightId;
  state = meetingEvent(state, studio, { type: "interact", id: "greet" });
  state = meetingEvent(state, studio, {
    type: "patch",
    value: { illustration: { ...state.illustration, parameter: 60 } },
  });
  assert.equal(state.illustration.result, result);
  assert.equal(state.illustration.flightId, id);
  assert.equal(result.parameter, 30);
  assert.equal(state.draft, "Keep this thought");
  state = meetingEvent(state, studio, {
    type: "result",
    result: simulateStudio("physics", 60),
  });
  assert.equal(state.illustration.flightId, id + 1);
  assert.equal(state.illustration.previousResult, result);
});
