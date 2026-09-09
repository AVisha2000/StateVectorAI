import * as T from "three";
import { researcherArm } from "./robotPresence.js";

export const OFFER_SECONDS = .45;
export const OFFER_PITCH = -1.25;
const ease = (age) => {
  const t = T.MathUtils.clamp(age / OFFER_SECONDS, 0, 1);
  return t * t * t * (t * (t * 6 - 15) + 10);
};

// The single shoulder writer. Offer changes capture the delivered pitch; ordinary
// gestures retain their existing analytic poses after the finite transition.
export function createOfferingArm(robot) {
  const arm = robot.userData.arm;
  let previous = null, previousGesture = null, departure = 0, elapsed = OFFER_SECONDS;
  return ({ offered, gesture, age, dt, paused, reducedMotion }) => {
    const target = offered ? OFFER_PITCH : researcherArm(gesture, age);
    if (previous === null) {
      elapsed = OFFER_SECONDS;
      departure = target;
    } else if (previous !== offered) {
      departure = arm.rotation.x;
      elapsed = paused || reducedMotion ? OFFER_SECONDS : 0;
    } else if (reducedMotion || (paused && previousGesture !== gesture)) elapsed = OFFER_SECONDS;
    else if (!paused) elapsed = Math.min(OFFER_SECONDS, elapsed + Math.max(0, Math.min(dt, .05)));
    if (!paused || previous !== offered || previousGesture !== gesture || reducedMotion)
      arm.rotation.x = elapsed >= OFFER_SECONDS ? target : T.MathUtils.lerp(departure, target, ease(elapsed));
    previous = offered;
    previousGesture = gesture;
    return { pitch: arm.rotation.x, done: elapsed >= OFFER_SECONDS };
  };
}

// The GLB palm/finger coordinates come from the authored shoulder rig; fallback
// uses its own shorter hand. Scene-frame output, paper near edge at the fingers.
export function offeredNotePose(robot) {
  robot.updateWorldMatrix(true, true);
  const palm = new T.Vector3(...(robot.userData.robotAsset ? [.016, -.146, .008] : [0, -.13, 0]));
  const position = robot.userData.arm.localToWorld(palm);
  const quaternion = robot.getWorldQuaternion(new T.Quaternion());
  position.add(new T.Vector3(0, .009, .105).applyQuaternion(quaternion));
  return { position, quaternion, scale: 1 };
}
