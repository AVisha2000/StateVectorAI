import test from "node:test";
import assert from "node:assert/strict";
import {
  clampSample,
  scrubSample,
  replaySample,
  kineticsFrame,
  kineticsObservation,
  startKineticReplay,
} from "./kineticsBench.js";
import { simulateStudio } from "./studioSimulation.js";
import { initialMeeting, meetingEvent } from "./meetingInteraction.js";
test("time input is bounded, rounded and independent of pixel width", () => {
  assert.equal(clampSample(NaN), 0);
  assert.equal(clampSample(-10), 0);
  assert.equal(clampSample(900), 120);
  assert.equal(scrubSample(150, 50, 200), 60);
  assert.equal(scrubSample(75, 25, 100), 60);
  assert.equal(scrubSample(1, 0, 0), 0);
  assert.equal(clampSample(59.7), 60);
});
test("both benches select exact paired worker samples, not a new calculation", () => {
  for (const id of ["chemistry", "biology"])
    for (const rate of [0.1, 0.5, 1.5]) {
      const result = simulateStudio(id, rate);
      for (let index = 0; index <= 120; index++) {
        const frame = kineticsFrame({ result, sampleIndex: index });
        assert.equal(frame.own, result.points[index].y);
        assert.equal(frame.baseline, result.baseline[index].y);
        assert.equal(frame.time, result.points[index].x);
        if (id === "chemistry")
          assert.ok(Math.abs(frame.product + frame.own - 1) < 1e-12);
      }
    }
});
test("reaction decreases and growth increases; matched rates agree at every sample", () => {
  for (const id of ["chemistry", "biology"]) {
    const result = simulateStudio(id, 0.5);
    for (let i = 1; i <= 120; i++) {
      const f = kineticsFrame({ result, sampleIndex: i });
      assert.equal(f.own, f.baseline);
      assert.ok(
        id === "chemistry"
          ? f.own <= result.points[i - 1].y
          : f.own >= result.points[i - 1].y,
      );
    }
  }
});
function replayHarness(start = 0) {
  const listeners = new Map(),
    callbacks = new Map(),
    samples = [];
  const document = {
    hidden: false,
    addEventListener: (name, fn) => listeners.set(name, fn),
    removeEventListener: (name) => listeners.delete(name),
  };
  let request = 0,
    stops = 0;
  const flags = { paused: false, reducedMotion: false };
  const dispose = startKineticReplay(
    {
      start,
      onSample: (i) => samples.push(i),
      onStop: () => stops++,
      state: () => flags,
    },
    {
      document,
      requestFrame: (fn) => {
        callbacks.set(++request, fn);
        return request;
      },
      cancelFrame: (id) => callbacks.delete(id),
    },
  );
  return {
    samples,
    flags,
    document,
    dispose,
    listeners,
    callbacks,
    get stops() {
      return stops;
    },
    frame: (now) => {
      const pending = [...callbacks.values()];
      callbacks.clear();
      pending.forEach((fn) => fn(now));
    },
    hide: () => {
      document.hidden = true;
      listeners.get("visibilitychange")?.();
    },
  };
}
test("default replay adapter keeps native animation callbacks bound to the window", (t) => {
  const nativeWindow = {
    requestAnimationFrame() {
      assert.equal(this, nativeWindow);
      return 1;
    },
    cancelAnimationFrame(id) {
      assert.equal(this, nativeWindow);
      assert.equal(id, 1);
    },
  };
  for (const [name, value] of Object.entries({
    window: nativeWindow,
    document: {
      hidden: false,
      addEventListener() {},
      removeEventListener() {},
    },
  })) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { value, configurable: true });
    t.after(() => {
      if (previous) Object.defineProperty(globalThis, name, previous);
      else delete globalThis[name];
    });
  }
  const dispose = startKineticReplay({
    start: 0,
    onSample() {},
    onStop() {},
    state: () => ({}),
  });
  dispose();
});
test("real replay clock matches direct seek at 30, 60, 120 and 240 Hz and terminates", () => {
  for (const hz of [30, 60, 120, 240]) {
    const h = replayHarness();
    for (let tick = 0; tick <= hz * 4; tick++) {
      h.frame((tick * 1000) / hz);
      if (tick % hz === 0)
        assert.equal(h.samples.at(-1), replaySample(0, tick / hz));
    }
    assert.equal(h.samples.at(-1), 120);
    assert.equal(h.stops, 1);
    assert.equal(h.callbacks.size, 0);
    assert.equal(h.listeners.size, 0);
  }
});
test("replay pause freezes samples; a long stall drops excess presentation time", () => {
  const h = replayHarness(30);
  h.frame(0);
  h.frame(50);
  assert.equal(h.samples.at(-1), 32);
  h.flags.paused = true;
  h.frame(100);
  h.frame(10000);
  assert.equal(h.samples.at(-1), 32);
  h.flags.paused = false;
  h.frame(20000);
  assert.equal(h.samples.at(-1), 33);
  h.dispose();
  assert.equal(h.callbacks.size, 0);
  assert.equal(h.listeners.size, 0);
  assert.equal(h.stops, 0);
});
test("hiding the page stops replay even when animation frames are suspended", () => {
  const h = replayHarness(120);
  h.frame(0);
  assert.equal(h.samples.at(-1), 0);
  h.hide();
  assert.equal(h.stops, 1);
  assert.equal(h.callbacks.size, 0);
  assert.equal(h.listeners.size, 0);
  h.document.hidden = false;
  h.frame(9000);
  assert.deepEqual(h.samples, [0]);
});
test("reduced motion stops replay without changing the selected sample", () => {
  const h = replayHarness(60);
  h.frame(0);
  h.flags.reducedMotion = true;
  h.frame(40);
  assert.deepEqual(h.samples, [60]);
  assert.equal(h.stops, 1);
  assert.equal(h.callbacks.size, 0);
});
test("scrubbing and sharing observation preserve drafts, source result and launch identity", () => {
  const studio = { id: "biology", name: "Fern", title: "Growth" };
  const result = simulateStudio("biology", 1.2);
  let state = { ...initialMeeting(studio), draft: "My unfinished thought" };
  state = meetingEvent(state, studio, { type: "result", result });
  const id = state.illustration.flightId;
  state = meetingEvent(state, studio, { type: "scrub", index: 30 });
  const frame = kineticsFrame(state.illustration);
  assert.equal(frame.time, 2.5);
  assert.equal(state.illustration.result, result);
  assert.equal(state.illustration.flightId, id);
  state = meetingEvent(state, studio, {
    type: "quote",
    kind: "observation",
    text: kineticsObservation(frame),
  });
  assert.equal(state.quoteKind, "observation");
  assert.equal(state.draft, "My unfinished thought");
  assert.match(state.quote, /t=2.50 model units/);
  assert.match(state.quote, /Synthetic inputs/);
  const discussed = meetingEvent(state, studio, {
    type: "send",
    text: "Explain this observation",
  });
  assert.equal(discussed.draft, "My unfinished thought");
  assert.match(
    discussed.messages.at(-1).text,
    /scripted prompt about the browser toy/,
  );
  assert.doesNotMatch(discussed.messages.at(-1).text, /this passage/);
  state = meetingEvent(state, studio, {
    type: "quote",
    text: "A paper passage",
  });
  assert.equal(state.quoteKind, "passage");
});
test("missing or incompatible kinetic result fails closed and does not scrub other models", () => {
  assert.equal(kineticsFrame({}), null);
  assert.equal(kineticsFrame({ result: simulateStudio("physics", 45) }), null);
  assert.equal(
    kineticsFrame({ result: { id: "biology", points: [], baseline: [] } }),
    null,
  );
  const studio = { name: "Lyra", title: "Physics" };
  const state = meetingEvent(initialMeeting(studio), studio, {
    type: "result",
    result: simulateStudio("physics", 45),
  });
  assert.deepEqual(
    meetingEvent(state, studio, { type: "scrub", index: 4 }),
    state,
  );
});
