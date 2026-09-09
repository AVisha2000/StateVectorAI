import { captureResultCard } from "./resultCards.js";

export const LANDING_TARGETS = Object.freeze([
  Object.freeze({ id: "meadow", name: "Meadow pad", range: 30, tolerance: 0.8 }),
  Object.freeze({ id: "clearing", name: "Clearing pad", range: 36, tolerance: 0.8 }),
  Object.freeze({ id: "horizon", name: "Horizon pad", range: 40, tolerance: 0.5 }),
]);
export const landingTarget = (id) => LANDING_TARGETS.find((t) => t.id === id);
export const initialLandingChallenge = () => ({ targetId: null, lastTargetId: "meadow", shots: [] });
export const targetInterval = (target) => `${(target.range - target.tolerance).toFixed(1)}–${(target.range + target.tolerance).toFixed(1)} m`;

// SI range, inclusive interval, distance to the nearest allowed edge.
export function judgeLanding(target, range) {
  if (!target || !Number.isFinite(range)) return null;
  const low = target.range - target.tolerance, high = target.range + target.tolerance;
  const direction = range < low ? "short" : range > high ? "long" : "inside";
  return { hit: direction === "inside", direction, distance: Math.max(low - range, range - high, 0) };
}

export function recordLandingShot(challenge, result, targetId, flightId) {
  // A new committed flight supersedes presentation of any unfinished earlier one.
  const shots = challenge.shots.map((shot) => shot.status === "flying" ? { ...shot, status: "interrupted" } : shot);
  const target = landingTarget(targetId);
  if (target && result?.id === "physics" && captureResultCard({ result })) {
    const range = result.points.at(-1).x;
    shots.push(Object.freeze({ flightId, targetId, angle: result.parameter, range, ...judgeLanding(target, range), status: "flying" }));
  }
  return { ...challenge, shots };
}

export function landChallengeShot(challenge, flightId) {
  const shot = challenge.shots.find((s) => s.flightId === flightId && s.status === "flying");
  if (!shot) return challenge;
  return { ...challenge, shots: challenge.shots.map((s) => s === shot ? Object.freeze({ ...s, status: "landed" }) : s) };
}

export function presentedLandingTarget(challenge, pendingTargetId) {
  return landingTarget(challenge.shots.findLast((s) => s.status === "flying")?.targetId) || landingTarget(pendingTargetId) || landingTarget(challenge.targetId);
}

export function completedLandingTargets(challenge) {
  return [...new Set(challenge.shots.filter((s) => s.status === "landed" && s.hit).map((s) => s.targetId))];
}

export function landingFeedback(shot) {
  if (shot.status === "flying") return "Following your flight…";
  if (shot.status === "interrupted") return "Flight replaced before landing";
  return shot.hit ? "Inside the target!" : `${shot.distance.toFixed(2)} m ${shot.direction} of the target`;
}

export function landingObservation(shot) {
  const target = landingTarget(shot.targetId);
  return `${target.name}: ${shot.angle}° landed at ${shot.range.toFixed(2)} m; target ${targetInterval(target)} (inclusive). ${landingFeedback(shot).replace(/[.!]$/, "")}. Ideal browser toy: 20 m/s, no drag, flat ground; not research evidence.`;
}
