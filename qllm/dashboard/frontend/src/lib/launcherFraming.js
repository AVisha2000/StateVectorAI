import * as T from "three";
import { projectile } from "./projectileToy.js";

// World-space support for the complete rocket, not an isotropic padding sphere.
// The allowed aim is integer degrees. A 0.03-unit guard covers between-sample
// support; tests independently sample half-steps over the complete flight.
export function launcherFlightBounds(body, labWorld) {
  const bounds = new T.Box3(),
    corners = [];
  for (const x of [body.min.x, body.max.x])
    for (const y of [body.min.y, body.max.y])
      for (const z of [body.min.z, body.max.z])
        corners.push(new T.Vector3(x, y, z));
  const point = new T.Vector3(),
    turn = new T.Matrix4();
  for (let angle = 20; angle <= 70; angle++)
    for (let i = 0; i <= 120; i++) {
      const p = projectile(angle, i / 120),
        heading = i === 120 ? 0 : Math.atan2(p.vy, p.vx);
      turn.makeRotationZ(Math.PI / 2 - heading);
      for (const corner of corners) {
        point.copy(corner).applyMatrix4(turn);
        point.x += 0.13 - p.x * 0.04;
        point.y += 0.08 + p.y * 0.04;
        point.z -= 0.06;
        bounds.expandByPoint(point.applyMatrix4(labWorld));
      }
    }
  return bounds.expandByScalar(0.03);
}
