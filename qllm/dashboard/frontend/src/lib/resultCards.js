import { DEMO_MODELS } from "./studioSimulation.js";
import { clampSample, KINETIC_STUDIOS } from "./kineticsBench.js";

// Copy stored worker output, never the edited-but-unapplied slider or a replay
// prediction. Cards are immutable, visit-only browser-toy artifacts.
export function captureResultCard(illustration) {
  const result = illustration?.result,
    model = DEMO_MODELS[result?.id];
  if (
    !model ||
    ![result.parameter, result.baselineParameter].every(
      (v) => Number.isFinite(v) && v >= model.min && v <= model.max,
    )
  )
    return null;
  const valid = (points) =>
    Array.isArray(points) &&
    points.length === 121 &&
    points.every(
      (p, i) =>
        p &&
        Number.isFinite(p.x) &&
        Number.isFinite(p.y) &&
        (!i || p.x >= points[i - 1].x),
    );
  if (!valid(result.points) || !valid(result.baseline)) return null;
  const physics = result.id === "physics";
  const index = physics ? 120 : clampSample(illustration.sampleIndex ?? 60);
  const own = result.points[index],
    baseline = result.baseline[index];
  if (!physics && own.x !== baseline.x) return null;
  const kinetic = KINETIC_STUDIOS.includes(result.id);
  if (kinetic && ![own.y, baseline.y].every((v) => v >= 0 && v <= 1))
    return null;
  const copy = (points) =>
    Object.freeze(points.map((p) => Object.freeze({ x: p.x, y: p.y })));
  return Object.freeze({
    modelId: result.id,
    parameter: result.parameter,
    baselineParameter: result.baselineParameter,
    sampleIndex: index,
    x: own.x,
    own: physics ? own.x : own.y,
    baseline: physics ? baseline.x : baseline.y,
    points: copy(result.points),
    baselinePoints: copy(result.baseline),
  });
}

export function resultCardView(card) {
  const model = DEMO_MODELS[card.modelId],
    physics = card.modelId === "physics";
  const kinetic = KINETIC_STUDIOS.includes(card.modelId);
  const factor = kinetic ? 100 : 1,
    precision = kinetic ? 1 : physics ? 2 : 4;
  const unit = kinetic ? "%" : physics ? " m" : "";
  const own = `${(card.own * factor).toFixed(precision)}${unit}`;
  const baseline = `${(card.baseline * factor).toFixed(precision)}${unit}`;
  const delta = `${((card.own - card.baseline) * factor).toFixed(precision)}${kinetic ? " percentage points" : unit}`;
  const location = physics
    ? "Computed flight endpoint"
    : `${model.axis}: ${card.x.toFixed(2)}`;
  const label = physics
    ? "Range"
    : card.modelId === "chemistry"
      ? "Reactant remaining"
      : card.modelId === "biology"
        ? "Population / capacity"
        : "Sample output";
  const parameter = `${model.parameter}: ${card.parameter}${model.unit}`;
  const baselineParameter = `${card.baselineParameter}${model.unit}`;
  return {
    title: model.title,
    location,
    label,
    own,
    baseline,
    delta,
    parameter,
    baselineParameter,
    quote: `Browser toy — ${model.title}. ${location}. ${parameter}; baseline input ${baselineParameter}. ${label}: ${own}; baseline ${baseline}; difference ${delta}. Stored calculation, synthetic inputs, not a research result.`,
  };
}

export function heldResultCard(state) {
  return (
    state.resultCards?.find((card) => card.id === state.heldResultId) || null
  );
}
export function canPinResult(position, firstPerson) {
  return (
    !firstPerson || Math.hypot(position.x + 1.12, position.z + 0.85) <= 1.18
  );
}
