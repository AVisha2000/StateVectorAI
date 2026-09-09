import test from "node:test";
import assert from "node:assert/strict";
import { createStudioCalculation } from "./studioCalculation.js";
import { simulateStudio } from "./studioSimulation.js";

function setup(factory) {
  const workers = [],
    states = [],
    results = [];
  const owner = createStudioCalculation({
    createWorker:
      factory ||
      (() => {
        const w = {
          sent: null,
          stops: 0,
          postMessage(data) {
            this.sent = data;
          },
          terminate() {
            this.stops++;
          },
        };
        workers.push(w);
        return w;
      }),
    onState: (state) => states.push(state),
    onResult: (result) => results.push(result),
  });
  return { owner, workers, states, results };
}
test("worker completion retains its captured target; failures publish no attempt and retry can select another target", () => {
  const workers = [], results = [], states = [];
  const owner = createStudioCalculation({
    createWorker: () => { const worker = { postMessage() {}, terminate() {} }; workers.push(worker); return worker; },
    onState: (state) => states.push(state),
    onResult: (result, target) => results.push({ result, target }),
  });
  let target = "meadow";
  owner.start("physics", 24, target);
  target = "clearing";
  workers[0].onerror();
  assert.equal(results.length, 0);
  assert.equal(states.at(-1).busy, false);
  owner.start("physics", 31, target);
  target = "horizon";
  workers[1].onmessage({ data: { result: simulateStudio("physics", 31) } });
  assert.equal(results[0].target, "clearing");
  assert.equal(results[0].result.parameter, 31);
  workers[0].onmessage({ data: { result: simulateStudio("physics", 24) } });
  assert.equal(results.length, 1);
  owner.dispose();
});
test("room and bench share one pending calculation; fired angle is fixed and duplicate replies cannot replay", () => {
  const { owner, workers, states, results } = setup();
  assert.equal(owner.start("physics", 30), true);
  assert.equal(owner.start("physics", 60), false);
  assert.deepEqual(workers[0].sent, { id: "physics", parameter: 30 });
  const result = simulateStudio("physics", 30);
  workers[0].onmessage({ data: { result } });
  workers[0].onmessage({ data: { result } });
  assert.deepEqual(results, [result]);
  assert.equal(workers[0].stops, 1);
  assert.deepEqual(states, [
    { busy: true, error: "", context: null },
    { busy: false, error: "", context: null },
  ]);
  assert.equal(owner.start("physics", 60), true);
  workers[0].onerror();
  assert.equal(states.at(-1).busy, true);
  owner.dispose();
});
test("leaving the session terminates its worker and ignores late completion or error", () => {
  const { owner, workers, states, results } = setup();
  owner.start("chemistry", 0.5);
  owner.dispose();
  workers[0].onmessage({ data: { result: simulateStudio("chemistry", 0.5) } });
  workers[0].onerror();
  assert.equal(workers[0].stops, 1);
  assert.equal(results.length, 0);
  assert.equal(states.length, 1);
  assert.equal(owner.start("chemistry", 0.5), false);
});
test("worker error, invalid result and mismatched input fail without publication; retry succeeds", () => {
  for (const data of [
    { error: "failure" },
    {},
    { result: simulateStudio("physics", 60) },
    { result: { ...simulateStudio("physics", 30), points: [] } },
  ]) {
    const { owner, workers, states, results } = setup();
    owner.start("physics", 30);
    workers[0].onmessage({ data });
    assert.equal(results.length, 0);
    assert.equal(states.at(-1).busy, false);
    assert.match(states.at(-1).error, /Try again/);
    assert.equal(owner.start("physics", 30), true);
    workers[1].onmessage({ data: { result: simulateStudio("physics", 30) } });
    assert.equal(results.length, 1);
    assert.equal(states.at(-1).error, "");
    owner.dispose();
  }
});
test("constructor/post failures clear busy state; invalid parameters never create workers", () => {
  for (const factory of [
    () => {
      throw new Error("no worker");
    },
    () => ({
      terminate() {},
      postMessage() {
        throw new Error("no post");
      },
    }),
  ]) {
    const { owner, states } = setup(factory);
    owner.start("physics", 45);
    assert.equal(states.at(-1).busy, false);
    assert.match(states.at(-1).error, /could not start/);
    owner.dispose();
  }
  const { owner, workers } = setup();
  for (const parameter of [NaN, Infinity, 19, 71])
    assert.equal(owner.start("physics", parameter), false);
  assert.equal(owner.start("unknown", 45), false);
  assert.equal(workers.length, 0);
});
