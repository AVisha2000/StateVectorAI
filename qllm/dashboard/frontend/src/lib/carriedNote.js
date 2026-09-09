import * as T from "three";

export const NOTE_TABLE = new T.Vector3(0, 0.473, 0.37);
export const NOTE_TRAVEL_SECONDS = 0.45;

// Authored studio units, not metres or a rigid-body simulation.
export function canReturnNote(position, firstPerson) {
  return !firstPerson || Math.hypot(position.x, position.z - 0.35) <= 1.12;
}

export function noteTarget(camera, firstPerson, held, overviewHand, restPose, conversation = false) {
  const position = NOTE_TABLE.clone(),
    quaternion = new T.Quaternion();
  let scale = 1;
  if (!held && restPose)
    return {
      position: restPose.position.clone(),
      quaternion: restPose.quaternion.clone(),
      scale: restPose.scale ?? 1,
    };
  if (held && firstPerson) {
    // The portrait's wide narrow-screen lens otherwise lowers the hand into
    // the coffee table. A near-camera viewmodel keeps it in front of furniture.
    const depth = conversation ? .12 : .5;
    const halfHeight = Math.tan(T.MathUtils.degToRad(camera.fov / 2)) * depth;
    position.set(halfHeight * camera.aspect * (conversation ? .55 : .47),
      -halfHeight * (conversation ? .66 : .28), -depth);
    position.applyMatrix4(camera.matrixWorld);
    quaternion
      .copy(camera.quaternion)
      .multiply(
        new T.Quaternion().setFromEuler(
          new T.Euler(Math.PI / 2 - 0.2, 0, -0.12),
        ),
      )
      .normalize();
    // Keep a real-sized note on desktop, smaller in a narrow view. The centre
    // stays clear for aiming; scale never changes semantic possession.
    scale = conversation
      // Lower the complete sheet + grip into the portrait's bottom band.
      // Height also bounds size: a wide conversation view can have a short lens.
      ? Math.min(.9, halfHeight * 1.45, halfHeight * camera.aspect / .36)
      : Math.min(0.9, (halfHeight * camera.aspect) / 0.27);
  } else if (held) {
    position.copy(overviewHand?.position || new T.Vector3(0.6, 0.4, 0.76));
    if (overviewHand) quaternion.copy(overviewHand.quaternion);
    else quaternion.setFromEuler(new T.Euler(-0.2, -2.3, 0.1));
  }
  return { position, quaternion, scale };
}

export function blendNotePose(from, to, seconds) {
  const t = T.MathUtils.clamp(seconds / NOTE_TRAVEL_SECONDS, 0, 1);
  const ease = t * t * t * (t * (t * 6 - 15) + 10);
  return {
    position:
      t === 1
        ? to.position.clone()
        : from.position.clone().lerp(to.position, ease),
    quaternion:
      t === 1
        ? to.quaternion.clone()
        : from.quaternion.clone().slerp(to.quaternion, ease).normalize(),
    scale: t === 1 ? to.scale : T.MathUtils.lerp(from.scale, to.scale, ease),
    done: t === 1,
  };
}

// The sole paper-transform writer. Called after the camera, on its existing
// render loop. No reparenting: paper and camera use the studio's world frame.
// Carry samples the current camera pose; a finite pickup blends from a captured
// departure to that current hand frame. This is kinematic presentation, not an
// inertial body. Camera mode cuts settle; possession never changes on a cut.
export function createNoteMotion(paper) {
  let previous = null,
    departure = null,
    age = NOTE_TRAVEL_SECONDS;
  return ({
    camera,
    firstPerson,
    held,
    revision,
    dt,
    paused,
    reducedMotion,
    overviewHand,
    restPose,
    conversation = false,
  }) => {
    const target = noteTarget(
      camera,
      firstPerson,
      held,
      overviewHand,
      restPose,
      conversation,
    );
    if (!previous || previous.firstPerson !== firstPerson || previous.conversation !== conversation) {
      departure = target;
      age = NOTE_TRAVEL_SECONDS;
    } else if (previous.revision !== revision) {
      departure = {
        position: paper.position.clone(),
        quaternion: paper.quaternion.clone(),
        scale: paper.scale.x,
      };
      age = paused || reducedMotion ? NOTE_TRAVEL_SECONDS : 0;
    } else if (reducedMotion) age = NOTE_TRAVEL_SECONDS;
    else if (!paused) age += Math.max(0, Math.min(dt, 0.05));
    previous = { firstPerson, revision, conversation };
    const pose = blendNotePose(departure, target, age);
    paper.position.copy(pose.position);
    paper.quaternion.copy(pose.quaternion);
    paper.scale.setScalar(pose.scale);
    return pose;
  };
}
