import { PROJECTILE } from "./projectileToy.js";

export function boundedLaunchAngle(value) {
  return Math.max(
    PROJECTILE.min,
    Math.min(PROJECTILE.max, Math.round(Number.isFinite(value) ? value : 45)),
  );
}
// Relative vertical dragging is stable in every camera orientation. Screen
// coordinates are CSS pixels; 4 pixels change one degree. Cancel never fires.
export function draggedLaunchAngle(angle, startY, currentY) {
  return boundedLaunchAngle(
    angle +
      (Number.isFinite(startY) && Number.isFinite(currentY)
        ? (startY - currentY) / 4
        : 0),
  );
}
export function keyedLaunchAngle(angle, key, shift = false) {
  if (key === "Home") return PROJECTILE.min;
  if (key === "End") return PROJECTILE.max;
  if (key === "ArrowUp") return boundedLaunchAngle(angle + (shift ? 5 : 1));
  if (key === "ArrowDown") return boundedLaunchAngle(angle - (shift ? 5 : 1));
  return null;
}
