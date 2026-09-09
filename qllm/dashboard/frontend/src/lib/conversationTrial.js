import { DEMO_MODELS } from "./studioSimulation.js";
import { captureResultCard } from "./resultCards.js";

// Bench/launcher feedback must not borrow an error from a conversation card.
// Its durable retry message lives in the card, even when another surface opens.
export function calculationForRoom(state) {
  const trialIndex = state.context?.trialIndex;
  return {
    ...state,
    context: typeof state.context === "string" ? state.context : null,
    trialIndex,
    error: Number.isInteger(trialIndex) ? "" : state.error,
  };
}

export function calculationEvent(result, context) {
  return {
    type: "result", result,
    targetId: typeof context === "string" ? context : null,
    trialIndex: Number.isInteger(context?.trialIndex) && context.trialIndex >= 0 ? context.trialIndex : undefined,
  };
}

export function newConversationTrial(studioId, parameter) {
  const model = DEMO_MODELS[studioId];
  if (!model) return null;
  return editConversationTrial({ studioId, parameter: model.initial, result: null, error: "" }, parameter);
}

export function editConversationTrial(trial, parameter) {
  const model = DEMO_MODELS[trial.studioId];
  if (!model || !Number.isFinite(parameter) || parameter < model.min || parameter > model.max)
    return trial;
  return { ...trial, parameter };
}

// A conversation card keeps its own completed calculation, never the mutable
// room slider or a later bench run. Use the same validator/copy as carried cards.
export function completeConversationTrial(trial, result) {
  if (!trial || result?.id !== trial.studioId) return trial;
  const snapshot = captureResultCard({ result });
  if (!snapshot) return trial;
  return { ...trial, error: "", result: Object.freeze({
    id: snapshot.modelId,
    parameter: snapshot.parameter,
    baselineParameter: snapshot.baselineParameter,
    points: snapshot.points,
    baseline: snapshot.baselinePoints,
    provenance: "Built-in browser illustration with synthetic inputs. No research agent or external service was used.",
  }) };
}
