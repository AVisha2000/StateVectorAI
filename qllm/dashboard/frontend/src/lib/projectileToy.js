// Shared, ideal unpowered projectile. SI units; rocket shape is presentation only.
export const PROJECTILE = Object.freeze({
  speed: 20,
  gravity: 9.81,
  min: 20,
  max: 70,
  replaySeconds: 2.8,
});

export function projectile(angle, fraction = 0) {
  if (
    !Number.isFinite(angle) ||
    angle < PROJECTILE.min ||
    angle > PROJECTILE.max ||
    !Number.isFinite(fraction)
  )
    throw new Error("Choose a launch angle from 20° to 70°.");
  const a = (angle * Math.PI) / 180;
  const vx = PROJECTILE.speed * Math.cos(a),
    vy0 = PROJECTILE.speed * Math.sin(a);
  const duration = (2 * vy0) / PROJECTILE.gravity;
  const u = Math.max(0, Math.min(1, fraction)),
    t = u * duration;
  return {
    x: vx * t,
    y: u === 1 ? 0 : Math.max(0, vy0 * t - (PROJECTILE.gravity * t * t) / 2),
    vx,
    vy: vy0 - PROJECTILE.gravity * t,
    range: vx * duration,
    height: (vy0 * vy0) / (2 * PROJECTILE.gravity),
    duration,
    fraction: u,
  };
}

export function aimAngle(x, y, originX = 55, originY = 230) {
  if (![x, y, originX, originY].every(Number.isFinite)) return 45;
  return Math.round(
    Math.max(
      PROJECTILE.min,
      Math.min(
        PROJECTILE.max,
        (Math.atan2(originY - y, Math.max(0.001, x - originX)) * 180) / Math.PI,
      ),
    ),
  );
}

export function flightPresentation(angle, age) {
  const state = projectile(angle, Math.max(0, age) / PROJECTILE.replaySeconds);
  return {
    ...state,
    heading: Math.atan2(state.vy, state.vx),
    landed: state.fraction === 1,
  };
}
