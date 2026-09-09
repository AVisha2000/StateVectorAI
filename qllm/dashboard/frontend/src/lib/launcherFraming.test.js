import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { launcherFlightBounds } from "./launcherFraming.js";
import { inspectionPose } from "./studioCamera.js";
import { projectile } from "./projectileToy.js";

test("rotated launcher framing contains full body at independent flight half-steps and narrow/wide views", () => {
  const body = new T.Box3(
    new T.Vector3(-0.16, -0.18, -0.16),
    new T.Vector3(0.16, 0.56, 0.16),
  );
  const lab = new T.Matrix4().compose(
    new T.Vector3(0.94, 0.02, -0.85),
    new T.Quaternion().setFromAxisAngle(new T.Vector3(0, 1, 0), Math.PI / 2),
    new T.Vector3(1.02, 1.02, 1.02),
  );
  const bounds = launcherFlightBounds(body, lab);
  const support = [];
  for (let angle = 20; angle <= 70; angle++)
    for (let i = 0; i < 120; i++) {
      const p = projectile(angle, (i + 0.5) / 120),
        rotation = Math.PI / 2 - Math.atan2(p.vy, p.vx);
      for (const x of [body.min.x, body.max.x])
        for (const y of [body.min.y, body.max.y])
          for (const z of [body.min.z, body.max.z]) {
            const point = new T.Vector3(
              x * Math.cos(rotation) -
                y * Math.sin(rotation) +
                0.13 -
                p.x * 0.04,
              x * Math.sin(rotation) +
                y * Math.cos(rotation) +
                0.08 +
                p.y * 0.04,
              z - 0.06,
            ).applyMatrix4(lab);
            assert.ok(bounds.containsPoint(point));
            support.push(point);
          }
    }
  for (const aspect of [0.6, 1, 2.2, 3.5]) {
    const pose = inspectionPose(bounds, aspect),
      camera = new T.PerspectiveCamera(pose.fov, aspect, 0.05, 35);
    camera.position.copy(pose.position);
    camera.quaternion.copy(pose.quaternion);
    camera.updateMatrixWorld();
    for (const point of support) {
      const ndc = point.clone().project(camera);
      assert.ok(ndc.toArray().every(Number.isFinite));
      assert.ok(
        Math.abs(ndc.x) <= 0.841 &&
          Math.abs(ndc.y) <= 0.841 &&
          Math.abs(ndc.z) < 1,
      );
    }
  }
});
