import * as T from "three";

export const COFFEE_SECONDS = 1.8;
const HANDOFF_SECONDS = .6;
const RETARGET_SECONDS = .3;
const ease = t => t * t * t * (t * (t * 6 - 15) + 10);
const copyPose = object => ({ position: object.position.clone(), quaternion: object.quaternion.clone() });

function blend(from, to, fraction, clearance = 0) {
  const t = T.MathUtils.clamp(fraction, 0, 1);
  const pose = {
    position: from.position.clone().lerp(to.position, ease(t)),
    quaternion: from.quaternion.clone().slerp(to.quaternion, ease(t)).normalize(),
  };
  pose.position.y += clearance * 16 * t * t * (1 - t) * (1 - t);
  return pose;
}

// Authored finger coordinates shared with the existing GLB/fallback offer rig.
// Return the actual world-space point; the cup stays in its existing parent.
export function coffeeGrip(robot) {
  robot.updateWorldMatrix(true, true);
  return robot.userData.arm.localToWorld(new T.Vector3(
    ...(robot.userData.robotAsset ? [.016, -.146, .008] : [0, -.13, 0]),
  ));
}

export function coffeePose(cup, robot, rest, seconds) {
  const age = Number.isFinite(seconds) ? Math.max(0, seconds) : COFFEE_SECONDS;
  if (age === 0 || age >= COFFEE_SECONDS) return { ...copyPose(rest), phase: "rest" };
  const position = cup.parent.worldToLocal(coffeeGrip(robot));
  const quaternion = cup.parent.getWorldQuaternion(new T.Quaternion()).invert()
    .multiply(robot.getWorldQuaternion(new T.Quaternion()))
    .multiply(new T.Quaternion().setFromAxisAngle(new T.Vector3(0, 0, 1), -.12 * Math.sin(Math.PI * age / COFFEE_SECONDS)))
    .normalize();
  // The existing torus handle is centered at cup-local +X .05. Keep that
  // point on the fingers while the cup remains upright for a small toast.
  position.sub(new T.Vector3(.05, 0, 0).multiply(cup.scale).applyQuaternion(quaternion));
  const held = { position, quaternion };
  if (age < HANDOFF_SECONDS)
    return { ...blend(rest, held, age / HANDOFF_SECONDS, .12), phase: "handoff" };
  if (age <= COFFEE_SECONDS - HANDOFF_SECONDS) return { ...held, phase: "held" };
  return { ...blend(held, rest, (age - (COFFEE_SECONDS - HANDOFF_SECONDS)) / HANDOFF_SECONDS, .12), phase: "return" };
}

// Sole cup-transform writer, after the shoulder writer on the existing clock.
// This is a kinematic toast, not reach IK, reparenting or a rigid-body simulation.
export function createCoffeeMotion(cup, robot) {
  const rest = copyPose(cup);
  let previous = null, departure = null, startedAt = 0, phase = "rest";
  return ({ gesture, age, offered, paused, reducedMotion }) => {
    const changed = previous && (previous.id !== gesture.id || previous.offered !== offered);
    if (changed && (cup.position.distanceToSquared(rest.position) > 1e-16 || cup.quaternion.angleTo(rest.quaternion) > 1e-8)) {
      departure = copyPose(cup);
      startedAt = age;
    }
    if (!paused || changed || reducedMotion || !previous) {
      const active = gesture.type === "coffee" && !offered && !paused && !reducedMotion;
      let pose = active ? coffeePose(cup, robot, rest, age) : { ...copyPose(rest), phase: "rest" };
      if (paused || reducedMotion) departure = null;
      if (departure) {
        const progress = (age - startedAt) / RETARGET_SECONDS;
        if (progress >= 1) departure = null;
        else pose = { ...blend(departure, pose, progress, .08), phase: "retarget" };
      }
      cup.position.copy(pose.position);
      cup.quaternion.copy(pose.quaternion);
      phase = pose.phase;
    }
    previous = { id: gesture.id, offered };
    cup.updateWorldMatrix(true, false);
    const gripError = cup.localToWorld(new T.Vector3(.05, 0, 0)).distanceTo(coffeeGrip(robot));
    return { phase, gripError, position: cup.position.clone() };
  };
}
