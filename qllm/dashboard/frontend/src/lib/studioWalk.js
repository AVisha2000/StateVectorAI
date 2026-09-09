import { WALK_OBSTACLES, WALK_EXTENT, VISITOR_SPAWN } from "./studioLayout.js";
export { WALK_OBSTACLES } from "./studioLayout.js";

// Signed clearance and outward normal for a circle or rounded oriented rectangle.
export function walkContact(point, obstacle) {
  const dx = point.x - obstacle.x, dz = point.z - obstacle.z;
  if (obstacle.radius !== undefined) {
    const distance = Math.hypot(dx,dz);
    return { gap: distance - obstacle.radius - obstacle.padding,
      x: distance > 1e-12 ? dx/distance : 1, z: distance > 1e-12 ? dz/distance : 0 };
  }
  const c = Math.cos(obstacle.turn), s = Math.sin(obstacle.turn);
  const x = c*dx-s*dz, z = s*dx+c*dz;
  const qx = Math.abs(x)-obstacle.width/2, qz = Math.abs(z)-obstacle.depth/2;
  const ox = Math.max(qx,0), oz = Math.max(qz,0), distance = Math.hypot(ox,oz);
  let nx, nz;
  if (distance > 1e-12) { nx = Math.sign(x)*ox/distance; nz = Math.sign(z)*oz/distance; }
  else if (qx > qz) { nx = x < 0 ? -1 : 1; nz = 0; }
  else { nx = 0; nz = z < 0 ? -1 : 1; }
  return { gap: distance + Math.min(Math.max(qx,qz),0) - obstacle.padding,
    x: c*nx+s*nz, z: -s*nx+c*nz };
}

export function walkClearance(point) {
  return Math.min(WALK_EXTENT-Math.hypot(point.x,point.z), ...WALK_OBSTACLES.map(o => walkContact(point,o).gap));
}

// Spatially bounded kinematic projection, not a dynamics integrator. Contacts do
// not change furniture, camera orientation or the research/animation clocks.
export function walkDisplacement(position, dx, dz, maxStep = .008) {
  let point = { ...position };
  const distance = Math.hypot(dx,dz);
  if (!Number.isFinite(distance) || !distance || !Number.isFinite(maxStep) || maxStep <= 0 || walkClearance(point) < -1e-7) return point;
  const steps = Math.ceil(distance / maxStep), sx = dx/steps, sz = dz/steps;
  for (let i=0; i<steps; i++) {
    const next = { x: point.x+sx, z: point.z+sz };
    for (let pass=0; pass<16; pass++) {
      for (const obstacle of WALK_OBSTACLES) {
        const contact = walkContact(next,obstacle);
        if (contact.gap < 0) {
          next.x += contact.x * (-contact.gap + 1e-9);
          next.z += contact.z * (-contact.gap + 1e-9);
        }
      }
      const extent = Math.hypot(next.x,next.z);
      if (extent > WALK_EXTENT) { next.x *= WALK_EXTENT/extent; next.z *= WALK_EXTENT/extent; }
      if (walkClearance(next) >= -1e-7) break;
    }
    // Intersecting contacts may not converge. Keep the last safe point, never
    // push through a neighbour or make a larger move than the user requested.
    if (walkClearance(next) < -1e-7 || Math.hypot(next.x-point.x,next.z-point.z) > distance/steps+1e-6) break;
    point = next;
  }
  return point;
}

export function stepVisitor(position, yaw, forward, right, seconds, fast = false) {
  if (!position || ![position.x,position.z].every(Number.isFinite)) return { ...VISITOR_SPAWN };
  if (![yaw,forward,right,seconds].every(Number.isFinite)) return { ...position };
  const norm = Math.max(1,Math.hypot(forward,right));
  const speed = (fast ? 1.8 : 1.05) * Math.max(0,Math.min(seconds,.05)) / norm;
  return walkDisplacement(position,
    (-Math.sin(yaw)*forward+Math.cos(yaw)*right)*speed,
    (-Math.cos(yaw)*forward-Math.sin(yaw)*right)*speed);
}
