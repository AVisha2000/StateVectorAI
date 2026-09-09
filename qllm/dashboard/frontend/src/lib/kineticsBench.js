export const KINETIC_STUDIOS = ["chemistry", "biology"];
export const SAMPLE_END = 120;
export function clampSample(value) {
  return Number.isFinite(value)
    ? Math.max(0, Math.min(SAMPLE_END, Math.round(value)))
    : 0;
}
export function scrubSample(x, left, width) {
  return Number.isFinite(width) && width > 0
    ? clampSample(((x - left) / width) * SAMPLE_END)
    : 0;
}
// Replay presents existing worker samples. It never re-runs or extrapolates a model.
export function replaySample(start, elapsedSeconds) {
  return clampSample(clampSample(start) + Math.max(0, elapsedSeconds) * 30);
}

// One bounded presentation clock for stored samples. A long frame deliberately
// drops time beyond 50 ms; neither the readout nor the 3D view advances alone.
export function startKineticReplay(
  { start, onSample, onStop, state },
  environment = {
    document,
    requestFrame: (callback) => window.requestAnimationFrame(callback),
    cancelFrame: (id) => window.cancelAnimationFrame(id),
  },
) {
  const doc = environment.document;
  let request,
    previous = null,
    elapsed = 0,
    last = -1,
    stopped = false;
  const initial = start >= SAMPLE_END ? 0 : clampSample(start);
  const dispose = () => {
    stopped = true;
    environment.cancelFrame(request);
    doc.removeEventListener("visibilitychange", visibility);
  };
  const finish = () => {
    dispose();
    onStop();
  };
  const visibility = () => {
    if (doc.hidden && !stopped) finish();
  };
  const tick = (now) => {
    if (stopped) return;
    const dt =
      previous === null
        ? 0
        : Math.max(0, Math.min(0.05, (now - previous) / 1000));
    previous = now;
    const current = state();
    if (doc.hidden || current.reducedMotion) {
      finish();
      return;
    }
    if (!current.paused) elapsed += dt;
    const index = replaySample(initial, elapsed);
    if (index !== last) {
      last = index;
      onSample(index);
    }
    if (index === SAMPLE_END) {
      finish();
      return;
    }
    request = environment.requestFrame(tick);
  };
  doc.addEventListener("visibilitychange", visibility);
  if (doc.hidden || state().reducedMotion) finish();
  else request = environment.requestFrame(tick);
  return dispose;
}
export function kineticsFrame(illustration) {
  const result = illustration?.result;
  if (!KINETIC_STUDIOS.includes(result?.id)) return null;
  const index = clampSample(illustration.sampleIndex ?? 60);
  const own = result.points?.[index],
    baseline = result.baseline?.[index];
  if (!own || !baseline || ![own.x, own.y, baseline.y].every(Number.isFinite))
    return null;
  if (
    own.x !== baseline.x ||
    own.x < 0 ||
    own.y < 0 ||
    own.y > 1 ||
    baseline.y < 0 ||
    baseline.y > 1
  )
    return null;
  const previous =
    illustration.previousResult?.id === result.id
      ? illustration.previousResult.points?.[index]?.y
      : null;
  return {
    id: result.id,
    index,
    time: own.x,
    own: own.y,
    baseline: baseline.y,
    previous: Number.isFinite(previous) ? previous : null,
    parameter: result.parameter,
    baselineParameter: result.baselineParameter,
    product: result.id === "chemistry" ? 1 - own.y : null,
  };
}
export function kineticsObservation(frame) {
  if (!frame) return "";
  const chemistry = frame.id === "chemistry";
  return `Browser toy — ${chemistry ? "first-order reaction" : "logistic growth"}. At t=${frame.time.toFixed(2)} ${chemistry ? "s" : "model units"}, ${chemistry ? "reactant remaining" : "population / capacity"} is ${(frame.own * 100).toFixed(1)}% with rate ${frame.parameter}; the rate-${frame.baselineParameter} baseline is ${(frame.baseline * 100).toFixed(1)}%. Difference: ${((frame.own - frame.baseline) * 100).toFixed(1)} percentage points. Synthetic inputs, not a research result.`;
}
