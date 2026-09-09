import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { createStudioInterior, studioInteriorPlan } from "../components/studioInterior.js";
import { createStudioLighting } from "./studioLighting.js";
import { disposeScene } from "../components/researchPlanetScene.js";

test("pavilion plan replays with unique identities, positive dimensions and clear walking space", () => {
  const plan = studioInteriorPlan();
  assert.deepEqual(plan, studioInteriorPlan());
  assert.equal(new Set(plan.map(p => p.id)).size, plan.length);
  for (const p of plan) {
    assert.ok(p.size.every(v => Number.isFinite(v) && v > 0), p.id);
    assert.ok(p.at.every(Number.isFinite) && Number.isFinite(p.turn), p.id);
    if (p.id.startsWith("wall-")) {
      // Lower envelope of any wall/exterior volume is outside the radius-2.1 walk disk.
      assert.ok(Math.abs(p.at[2]) - p.size[2] / 2 > 2.1, p.id);
    }
    if (p.id.startsWith("hill-"))
      assert.ok(Math.hypot(p.at[0], p.at[2]) - Math.max(p.size[0], p.size[2]) / 2 > 2.7, p.id);
  }
  const boards = plan.filter(p => p.id.startsWith("board-"));
  for (let i = 0; i < boards.length; i++) {
    const a = boards[i];
    assert.ok(Math.abs(a.at[0]) + a.size[0] / 2 <= 2.7);
    assert.ok(Math.abs(a.at[2]) + a.size[2] / 2 <= 2.7);
    for (const b of boards.slice(i + 1)) {
      const xOverlap = Math.abs(a.at[0] - b.at[0]) < (a.size[0] + b.size[0]) / 2;
      const zOverlap = Math.abs(a.at[2] - b.at[2]) < (a.size[2] + b.size[2]) / 2;
      assert.ok(!xOverlap || !zOverlap, `${a.id}/${b.id}`);
    }
  }
});

test("compiled pavilion has bounded finite indexed resources, no texture downloads, and no pick targets", () => {
  const room = createStudioInterior();
  assert.equal(room.visible, false);
  assert.ok(room.children.length <= 14);
  let vertices = 0;
  for (const mesh of room.children) {
    assert.ok(mesh.geometry.index);
    for (const attribute of Object.values(mesh.geometry.attributes)) assert.ok(attribute.array.every(Number.isFinite));
    assert.ok(Number.isFinite(mesh.geometry.boundingSphere.radius));
    assert.ok(mesh.material.isMaterial);
    if(mesh.material.map) assert.ok(mesh.material.map.isDataTexture, "generated locally, no texture request");
    vertices += mesh.geometry.attributes.position.count;
    const intersections = [];
    mesh.raycast(new T.Raycaster(), intersections);
    assert.deepEqual(intersections, []);
    assert.equal(mesh.castShadow, mesh.name === "Studio frames");
  }
  assert.ok(vertices <= 25000);
  assert.ok(room.getObjectByName("Studio frames").receiveShadow);
  assert.equal(room.getObjectByName("Studio sky").material.side, T.BackSide);
  disposeScene(room);
});

test("actual geometry has open window rays on all walls, solid fascia and exterior depth", () => {
  const room = createStudioInterior();
  room.updateMatrixWorld(true);
  const hits = (origin, direction) => {
    const ray = new T.Raycaster(origin, direction, .01, 35), results = [];
    // Deliberately use the real geometry implementation, bypassing only UI non-pickability.
    room.children.forEach(mesh => T.Mesh.prototype.raycast.call(mesh, ray, results));
    return results.sort((a,b) => a.distance - b.distance);
  };
  for (let i = 0; i < 4; i++) {
    const turn = new T.Matrix4().makeRotationY(i * Math.PI / 2);
    for (const eye of [.8, 1.4]) {
      const result = hits(new T.Vector3(.5, eye, 0).applyMatrix4(turn), new T.Vector3(0,0,-1).transformDirection(turn));
      assert.ok(result.length > 0 && result[0].distance > 3, `wall ${i}: window ${eye}`);
      assert.ok(result[0].distance < 27, "exterior stays within far plane from walk area");
    }
    const closed = hits(new T.Vector3(.5, 1.86, 0).applyMatrix4(turn), new T.Vector3(0,0,-1).transformDirection(turn));
    // A transverse ceiling beam may be nearer; require the actual fascia too.
    const fascia = closed.find(hit => hit.object.name === "Studio plaster");
    assert.ok(fascia && fascia.distance > 2.5 && fascia.distance < 2.7);
  }
  const a = hits(new T.Vector3(.4,.8,0), new T.Vector3(0,0,-1))[0].point;
  const b = hits(new T.Vector3(.9,.8,0), new T.Vector3(0,0,-1))[0].point;
  assert.ok(a.distanceTo(b) > .4, "walking samples real exterior surfaces");
  disposeScene(room);
});

test("lighting uses one shadow map and restores overview without accumulating resources", () => {
  const scene = new T.Scene(), lighting = createStudioLighting(scene);
  assert.equal(scene.children.length, 2);
  assert.equal(scene.children.filter(o => o.castShadow).length, 1);
  assert.deepEqual(lighting.key.shadow.mapSize.toArray(), [1024,1024]);
  lighting.present(false);
  const before = [lighting.fill.intensity, lighting.key.intensity, scene.environmentIntensity];
  for (let i = 0; i < 20; i++) {
    lighting.present(true);
    assert.ok(lighting.fill.intensity < before[0] && scene.environmentIntensity < before[2]);
    lighting.present(false);
    assert.deepEqual([lighting.fill.intensity, lighting.key.intensity, scene.environmentIntensity], before);
  }
  assert.equal(scene.children.length, 2);
  scene.updateMatrixWorld(true);
  lighting.key.shadow.updateMatrices(lighting.key);
  const camera = lighting.key.shadow.camera;
  const room = createStudioInterior();
  const frames = room.getObjectByName("Studio frames").geometry.attributes;
  for (let i = 0; i < frames.position.count; i++) {
    const p = new T.Vector3().fromBufferAttribute(frames.position,i);
    p.addScaledVector(new T.Vector3().fromBufferAttribute(frames.normal,i), lighting.key.shadow.normalBias);
    p.project(camera);
    assert.ok(Math.abs(p.x)<.98 && Math.abs(p.y)<.98 && Math.abs(p.z)<.98, `frame ${i}: ${p.toArray()}`);
  }
  for (let i = 0; i < 32; i++) {
    const a = i * Math.PI / 16;
    for (const y of [0, .9]) {
      const point = new T.Vector3(2.1*Math.cos(a),y,2.1*Math.sin(a)).project(camera);
      assert.ok(Math.abs(point.x)<.98 && Math.abs(point.y)<.98 && Math.abs(point.z)<.98, point.toArray().join());
    }
  }
  // CPU target stand-in checks explicit shadow ownership without a GPU context.
  const target = new T.WebGLRenderTarget(1,1);
  let retired = 0;
  target.addEventListener("dispose", () => retired++);
  lighting.key.shadow.map = target;
  lighting.dispose();
  assert.equal(retired, 1);
  assert.equal(scene.children.length, 0);
  disposeScene(room); disposeScene(scene);
});

test("pavilion instances independently own and retire each generated mesh and material", () => {
  const a = createStudioInterior(), b = createStudioInterior();
  let ownDisposals = 0, otherDisposals = 0;
  a.children.forEach((mesh,i) => {
    assert.notEqual(mesh.geometry,b.children[i].geometry);
    assert.notEqual(mesh.material,b.children[i].material);
    mesh.geometry.addEventListener("dispose",()=>ownDisposals++);
    mesh.material.addEventListener("dispose",()=>ownDisposals++);
    b.children[i].geometry.addEventListener("dispose",()=>otherDisposals++);
    b.children[i].material.addEventListener("dispose",()=>otherDisposals++);
  });
  disposeScene(a);
  assert.equal(ownDisposals, a.children.length * 2);
  assert.equal(otherDisposals, 0);
  disposeScene(b);
});
