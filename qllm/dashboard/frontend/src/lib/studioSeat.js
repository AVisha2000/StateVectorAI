import { STUDIO_LAYOUT } from "./studioLayout.js";

// Presentation-only posture, in the room's Y-up project frame. The standing
// walk position never enters the solid chair footprint. Sit/stand are explicit
// camera cuts, not paths through furniture or additional animation owners.
export function createCoffeeSeat() {
  let standingLook = null;
  const chair = STUDIO_LAYOUT.visitorChair;
  const researcher = STUDIO_LAYOUT.researcher;
  return {
    get seated() { return standingLook !== null; },
    sit(yaw, pitch) {
      if (standingLook) return null;
      standingLook = { yaw, pitch };
      const dx = researcher.x - chair.x, dz = researcher.z - chair.z;
      return {
        yaw: Math.atan2(-dx, -dz),
        pitch: Math.atan2(.78 - .72, Math.hypot(dx, dz)),
      };
    },
    stand() {
      const look = standingLook;
      standingLook = null;
      return look;
    },
    position(walkingPosition) {
      return standingLook ? chair : walkingPosition;
    },
    get height() { return standingLook ? .72 : .8; },
    get fov() { return standingLook ? 48 : 64; },
  };
}
