import * as T from "three";

const clamp = (value, limit) => Math.max(-limit, Math.min(limit, value));
export function gazeTarget(localTarget) {
  if (!localTarget || ![localTarget.x, localTarget.y, localTarget.z].every(Number.isFinite) || localTarget.z <= .02)
    return { yaw: 0, pitch: 0 };
  return {
    yaw: clamp(Math.atan2(localTarget.x, localTarget.z), .7),
    pitch: clamp(-Math.atan2(localTarget.y - .416, Math.hypot(localTarget.x, localTarget.z)), .22),
  };
}

// Non-physical, exponential response; current target is sampled at interval end.
export function followGaze(previous, target, dt) {
  const alpha = 1 - Math.exp(-8 * Math.max(0, Number.isFinite(dt) ? dt : 0));
  return { yaw: previous.yaw + (target.yaw - previous.yaw) * alpha, pitch: previous.pitch + (target.pitch - previous.pitch) * alpha };
}

export function faceExpression(elapsed, gesture, age, reducedMotion = false) {
  if (reducedMotion) return { eyeScale: 1, nod: 0 };
  const phase = ((Math.max(0, elapsed) % 4.8) + 4.8) % 4.8;
  const blink = phase >= 4.4 && phase <= 4.58 ? Math.sin(Math.PI * (phase - 4.4) / .18) ** 2 : 0;
  const nod = ["greet", "talk"].includes(gesture) && age > 0 && age < 1.15
    ? .12 * Math.sin(Math.PI * age / 1.15) ** 2 : 0;
  return { eyeScale: 1 - .86 * blink, nod };
}

export function researcherArm(gesture, age) {
  const safeAge = Math.max(0, age);
  const envelope = Math.max(0, 1 - safeAge / 2);
  if (gesture === "coffee") return safeAge >= 1.8 ? 0 : -1.65 * Math.sin(safeAge / 1.8 * Math.PI);
  if (["greet", "talk"].includes(gesture)) return -.15 + envelope * (-.55 - Math.sin(safeAge * 7) * .4);
  if (["paper", "board", "experiment"].includes(gesture)) return -1.1 * envelope;
  return -.15;
}

// Sole neck/eyelid writer. The room owns root/arm motion; the camera owns itself.
export function createRobotPresence(robot) {
  const head = robot.userData.head, eyes = robot.userData.eyes || [];
  let gaze = { yaw: 0, pitch: 0 };
  const local = new T.Vector3();
  return ({ camera, firstPerson, elapsed, gesture, age, dt, paused, reducedMotion }) => {
    if (!head) return null;
    if (reducedMotion) {
      gaze = { yaw: 0, pitch: 0 };
      head.rotation.set(0, 0, 0);
      eyes.forEach((eye) => { eye.scale.y = 1; });
    } else if (!paused) {
      robot.updateWorldMatrix(true, false);
      const target = firstPerson ? gazeTarget(robot.worldToLocal(local.copy(camera.position))) : { yaw: 0, pitch: 0 };
      gaze = followGaze(gaze, target, dt);
      const expression = faceExpression(elapsed, gesture.type, age);
      // Yaw then local pitch; the head pivot has an identity rest orientation.
      head.rotation.set(gaze.pitch + expression.nod, gaze.yaw, 0, "YXZ");
      eyes.forEach((eye) => { eye.scale.y = expression.eyeScale; });
    }
    return { yaw: head.rotation.y, pitch: head.rotation.x, eyeScale: eyes[0]?.scale.y ?? 1 };
  };
}
