import * as T from "three";

// Capture the authored, resting robot once. Gaze/gestures must not cause camera
// breathing. The upper-body support includes a small silhouette/motion margin.
export function conversationSubject(robot) {
  robot.updateWorldMatrix(true, true);
  const bounds = new T.Box3().setFromObject(robot);
  const target = robot.userData.head
    ? robot.userData.head.getWorldPosition(new T.Vector3())
    : new T.Vector3(robot.position.x, bounds.max.y - .13, robot.position.z);
  // The neck pivot is below the face: use the visible eyes for eye level,
  // without pulling the shot forward to their protruding visor geometry.
  const eyes = robot.userData.eyes || [];
  if (eyes.length) target.y = eyes.reduce((sum, eye) =>
    sum + eye.getWorldPosition(new T.Vector3()).y, 0) / eyes.length;
  target.y -= .04;
  bounds.min.y = Math.max(bounds.min.y, target.y - .23);
  bounds.expandByScalar(.04);
  const forward = new T.Vector3(0, 0, 1).transformDirection(robot.matrixWorld);
  forward.y = 0;
  forward.normalize();
  return { bounds, target, forward };
}

// Explicit portrait cut, never a travel path or a change to visitor position.
// Camera stays 1.1 studio units in front of the subject, above the coffee table;
// only the lens refits on resize. All eight support corners stay inside .84 NDC.
export function conversationPose(subject, aspect) {
  const { bounds, target, forward } = subject;
  if (bounds.isEmpty() || !Number.isFinite(aspect) || aspect <= 0 ||
      ![...bounds.min.toArray(), ...bounds.max.toArray(), ...target.toArray(), ...forward.toArray()].every(Number.isFinite) ||
      Math.abs(forward.length() - 1) > 1e-6 || Math.abs(forward.y) > 1e-6) return null;
  const distance = 1.1, right = new T.Vector3(0, 1, 0).cross(forward);
  let tangent = Math.tan(T.MathUtils.degToRad(40 / 2));
  for (const x of [bounds.min.x, bounds.max.x])
    for (const y of [bounds.min.y, bounds.max.y])
      for (const z of [bounds.min.z, bounds.max.z]) {
        const offset = new T.Vector3(x, y, z).sub(target);
        const depth = distance - offset.dot(forward);
        if (depth <= .05 || depth >= 35) return null;
        tangent = Math.max(tangent, Math.abs(offset.y) / (depth * .84),
          Math.abs(offset.dot(right)) / (depth * aspect * .84));
      }
  const fov = T.MathUtils.radToDeg(2 * Math.atan(tangent));
  if (fov > 110) return null;
  const position = target.clone().addScaledVector(forward, distance);
  return { position, target: target.clone(), fov,
    quaternion: new T.Quaternion().setFromRotationMatrix(
      new T.Matrix4().lookAt(position, target, new T.Vector3(0, 1, 0))),
  };
}

// Unparented Y-up perspective camera, studio units. Fit all eight corners of a
// conservative volume inside 84% of each NDC axis, at the authored lens.
export function inspectionPose(bounds, aspect, fov = 44) {
  if (
    bounds.isEmpty() ||
    !Number.isFinite(aspect) ||
    aspect <= 0 ||
    !Number.isFinite(fov) ||
    fov <= 0 ||
    fov >= 179
  )
    return null;
  const sphere = bounds.getBoundingSphere(new T.Sphere());
  if (![...sphere.center.toArray(), sphere.radius].every(Number.isFinite))
    return null;
  const back = new T.Vector3(0.85, 0.6, 1).normalize();
  const right = new T.Vector3(0, 1, 0).cross(back).normalize();
  const up = back.clone().cross(right).normalize();
  const tangent = Math.tan(T.MathUtils.degToRad(fov / 2)) * 0.84;
  let distance = 0.3,
    nearestOffset = Infinity;
  for (const x of [bounds.min.x, bounds.max.x])
    for (const y of [bounds.min.y, bounds.max.y])
      for (const z of [bounds.min.z, bounds.max.z]) {
        const offset = new T.Vector3(x, y, z).sub(sphere.center),
          depth = offset.dot(back);
        distance = Math.max(
          distance,
          depth + 0.1,
          depth + Math.abs(offset.dot(right)) / (tangent * aspect),
          depth + Math.abs(offset.dot(up)) / tangent,
        );
        nearestOffset = Math.min(nearestOffset, depth);
      }
  if (distance - nearestOffset >= 35) return null;
  const position = sphere.center.clone().addScaledVector(back, distance);
  const quaternion = new T.Quaternion().setFromRotationMatrix(
    new T.Matrix4().lookAt(position, sphere.center, new T.Vector3(0, 1, 0)),
  );
  return {
    position,
    quaternion,
    target: sphere.center,
    distance,
    radius: sphere.radius,
    fov,
  };
}

// Finite 0.65-second control handoff. Never repeatedly lerp a moving start.
export function inspectionBlend(start, end, seconds) {
  const t = Math.max(0, Math.min(1, seconds / 0.65));
  const ease = t * t * (3 - 2 * t);
  return {
    position: start.position.clone().lerp(end.position, ease),
    fov: T.MathUtils.lerp(start.fov ?? 44, end.fov ?? 44, ease),
    quaternion: start.quaternion
      .clone()
      .slerp(end.quaternion, ease)
      .normalize(),
    done: t === 1,
  };
}
