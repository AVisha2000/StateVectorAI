import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { createPaperGrip } from "./paperGrip.js";
import { PAPER_TOP, PAPER_BOTTOM } from "./paperSheet.js";
import { disposeScene } from "../components/researchPlanetScene.js";

test("authored hand is a bounded finite indexed mesh with smooth unit normals", () => {
  const grip = createPaperGrip();
  assert.equal(grip.children.length, 4);
  let triangles = 0;
  for (const part of grip.children) {
    const { position, normal } = part.geometry.attributes;
    const { index } = part.geometry;
    assert.equal(position.count, normal.count);
    assert.equal(index.count % 3, 0);
    assert.ok([...index.array].every(i => i >= 0 && i < position.count));
    assert.ok([...position.array, ...normal.array].every(Number.isFinite));
    for (let i = 0; i < normal.count; i++) {
      assert.ok(Math.abs(new T.Vector3().fromBufferAttribute(normal, i).length() - 1) < 2e-6);
    }
    assert.equal(part.material.isMeshStandardMaterial, true);
    assert.equal(part.material.map, null);
    triangles += index.count / 3;
  }
  assert.ok(triangles > 500 && triangles <= 8000);
  const bounds = new T.Box3().setFromObject(grip);
  assert.ok(bounds.min.y < -.025 && bounds.max.y > PAPER_TOP + .008);
  assert.ok(bounds.max.x < .14 && bounds.max.z < .185);
  disposeScene(grip);
});

test("opposing thumb contacts the paper face without covering its written area", () => {
  const grip = createPaperGrip();
  const geometry = grip.getObjectByName("Sculpted palm and five digits").geometry;
  const skin = geometry.attributes.position;
  let above = 0, behind = 0;
  for (let i = 0; i < skin.count; i++) {
    const x = skin.getX(i), y = skin.getY(i), z = skin.getZ(i);
    if (x < .095 && z < .125) {
      if (y > PAPER_TOP) { above++; assert.ok(x > .066, "thumb must stay outside writing"); }
      if (y <= PAPER_BOTTOM) behind++;
    }
  }
  const intersections = patch => {
    let count = 0;
    for (let i = 0; i < geometry.index.count; i += 3) {
      const triangle = new T.Triangle(...[0,1,2].map(j => new T.Vector3().fromBufferAttribute(skin,geometry.index.getX(i+j))));
      if (patch.intersectsTriangle(triangle)) count++;
    }
    return count;
  };
  const thumb = new T.Box3(new T.Vector3(.068,PAPER_BOTTOM,.05),new T.Vector3(.095,PAPER_TOP,.1));
  const fingertips = new T.Box3(new T.Vector3(-.095,PAPER_BOTTOM,.05),new T.Vector3(.068,PAPER_TOP,.11));
  assert.ok(intersections(thumb) > 0, "thumb must meet the actual thin sheet");
  assert.ok(intersections(fingertips) > 0, "curled fingertips must support the actual sheet back");
  for (const patch of [thumb,fingertips]) assert.equal(intersections(patch.clone().translate(new T.Vector3(.5,0,0))),0);
  assert.ok(above > 20 && behind > 20, "grip must oppose both sides of the paper");
  disposeScene(grip);
});

test("result grip shares resources within a room; new rooms own independent resources", () => {
  const first = createPaperGrip(), second = createPaperGrip(), result = first.clone(true);
  const room = new T.Group();
  room.add(first, result);
  let geometryDisposals = 0, materialDisposals = 0;
  for (let i = 0; i < first.children.length; i++) {
    const part = first.children[i];
    assert.equal(part.geometry, result.children[i].geometry);
    assert.equal(part.material, result.children[i].material);
    assert.notEqual(part.geometry, second.children[i].geometry);
    assert.notEqual(part.material, second.children[i].material);
    part.geometry.addEventListener("dispose", () => geometryDisposals++);
    part.material.addEventListener("dispose", () => materialDisposals++);
  }
  disposeScene(room);
  assert.equal(geometryDisposals, 4);
  assert.equal(materialDisposals, 4);
  disposeScene(second);
});
