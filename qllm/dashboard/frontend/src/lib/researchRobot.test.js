import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import * as T from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { ROBOT_FIELDS, cloneResearchRobot, loadResearchRobot } from "./researchRobot.js";
import { disposeScene } from "../components/researchPlanetScene.js";

const asset = async (field) => {
  const bytes = await readFile(new URL(`../../public/models/research-robots/${field}-robot.glb`, import.meta.url));
  return { bytes, buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
};
test("six distinct Blender exports meet the geometry, material and rig budget without external assets", async () => {
  const hashes = new Set();
  for (const field of ROBOT_FIELDS) {
    const { bytes, buffer } = await asset(field);
    assert.ok(bytes.length < 800000, field);
    hashes.add(createHash("sha256").update(bytes).digest("hex"));
    const json = JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString());
    assert.equal(json.images?.length || 0, 0);
    assert.equal(json.cameras?.length || 0, 0);
    assert.equal(json.animations?.length || 0, 0);
    assert.ok(json.buffers.every((b) => !b.uri));
    const { scene } = await new GLTFLoader().parseAsync(buffer, "");
    const arm = scene.getObjectByName("gesture_arm");
    assert.ok(arm?.children.length > 0, field);
    assert.ok(Math.abs(arm.position.y - .323) < .0001);
    const size = new T.Box3().setFromObject(scene).getSize(new T.Vector3());
    assert.ok(size.y > .45 && size.y < .6, `${field}: ${size.toArray()}`);
    assert.ok(size.x < .32 && size.z < .25);
    let primitives = 0, metallic = false, luminous = false;
    scene.traverse((o) => {
      if (!o.isMesh) return;
      primitives++;
      assert.ok(o.geometry.attributes.position.array.every(Number.isFinite));
      metallic ||= o.material.metalness > .6;
      luminous ||= o.material.emissiveIntensity > 0 && o.material.emissive.getHex() !== 0;
    });
    assert.ok(primitives <= 20 && metallic && luminous, `${field}: ${primitives}`);
    disposeScene(scene);
  }
  assert.equal(hashes.size, 6);
});
test("robot clones own render resources and shoulder transforms independently", async () => {
  const { buffer } = await asset("physics");
  const { scene } = await new GLTFLoader().parseAsync(buffer, "");
  const a = cloneResearchRobot(scene), b = cloneResearchRobot(scene);
  const resources = (root) => { const list = []; root.traverse((o) => { if (o.isMesh) list.push(o); }); return list; };
  const original = resources(scene), am = resources(a), bm = resources(b);
  for (let i = 0; i < am.length; i++) {
    assert.notEqual(am[i].geometry, original[i].geometry);
    assert.notEqual(am[i].geometry, bm[i].geometry);
    assert.notEqual(am[i].material, bm[i].material);
    assert.ok(am[i].castShadow && am[i].receiveShadow);
  }
  a.userData.arm.rotation.x = 1;
  assert.ok(Math.abs(b.userData.arm.rotation.x) < 1e-12);
  assert.ok(Math.abs(scene.getObjectByName("gesture_arm").rotation.x) < 1e-12);
  let otherDisposals = 0;
  bm.forEach((o) => o.geometry.addEventListener("dispose", () => otherDisposals++));
  disposeScene(a);
  assert.equal(otherDisposals, 0);
  disposeScene(b); disposeScene(scene);
  assert.equal(cloneResearchRobot(null), null);
});
test("sculpted physics fibres are closed head-owned sweeps with valid normals and triangles", async () => {
  const { buffer } = await asset("physics");
  const { scene } = await new GLTFLoader().parseAsync(buffer, "");
  const head = scene.getObjectByName("attention_head"), fibres = [];
  scene.traverse((object) => {
    if (object.isMesh && object.material.name === "Ivory fibre") fibres.push(object);
  });
  assert.equal(fibres.length, 1, "all locks share one existing material primitive");
  const hair = fibres[0];
  assert.equal(hair.parent, head);
  const { position, normal } = hair.geometry.attributes, indices = hair.geometry.index;
  assert.ok(position.count < 5000, "bounded authored hair geometry");
  const a = new T.Vector3(), b = new T.Vector3(), c = new T.Vector3();
  const ab = new T.Vector3(), ac = new T.Vector3(), surface = new T.Vector3();
  const edges = new Map(), neighbours = Array.from({ length: position.count }, () => []);
  for (let i = 0; i < position.count; i++) {
    const length = new T.Vector3().fromBufferAttribute(normal, i).length();
    assert.ok(Number.isFinite(length) && Math.abs(length - 1) < 1e-5);
  }
  for (let i = 0; i < indices.count; i += 3) {
    const ids = [indices.getX(i), indices.getX(i+1), indices.getX(i+2)];
    a.fromBufferAttribute(position, ids[0]);
    b.fromBufferAttribute(position, ids[1]);
    c.fromBufferAttribute(position, ids[2]);
    surface.crossVectors(ab.subVectors(b, a), ac.subVectors(c, a));
    assert.ok(surface.lengthSq() > 1e-18, `nondegenerate triangle ${i/3}`);
    for (let j = 0; j < 3; j++) {
      const from = ids[j], to = ids[(j+1)%3];
      neighbours[from].push(to);
      const key = `${Math.min(from,to)}:${Math.max(from,to)}`;
      edges.set(key, (edges.get(key) || 0) + 1);
    }
  }
  assert.ok([...edges.values()].every((count) => count === 2), "no open roots or tips");
  const visited = new Set();
  let locks = 0;
  for (let i = 0; i < position.count; i++) {
    if (visited.has(i)) continue;
    locks++;
    const pending = [i];
    while (pending.length) {
      const vertex = pending.pop();
      if (visited.has(vertex)) continue;
      visited.add(vertex);
      pending.push(...neighbours[vertex].filter((next) => !visited.has(next)));
    }
  }
  assert.equal(locks, 22, "14 side locks, four crown locks, two brows and two whiskers");
  scene.updateMatrixWorld(true);
  const before = hair.localToWorld(new T.Vector3().fromBufferAttribute(position, 0));
  head.rotation.y = .4;
  scene.updateMatrixWorld(true);
  const after = hair.localToWorld(new T.Vector3().fromBufferAttribute(position, 0));
  assert.ok(after.distanceTo(before) > .001, "an actual fibre vertex follows the head");
  assert.equal(scene.getObjectByName("eye_left").parent, head);
  assert.equal(scene.getObjectByName("eye_right").parent, head);
  disposeScene(scene);
});
test("failed model downloads reject for UI fallback and remain retryable; successful templates are cached", async (t) => {
  const { buffer } = await asset("biology");
  let calls = 0;
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls++;
    assert.equal(url, "/models/research-robots/biology-robot.glb");
    assert.ok(options.signal instanceof AbortSignal);
    return calls === 1 ? { ok: false } : { ok: true, arrayBuffer: async () => buffer };
  });
  assert.equal(await loadResearchRobot("unknown"), null);
  await assert.rejects(loadResearchRobot("biology"), /unavailable/);
  const pending = loadResearchRobot("biology");
  assert.equal(loadResearchRobot("biology"), pending);
  const scene = await pending;
  assert.ok(scene.getObjectByName("gesture_arm"));
  assert.equal(calls, 2);
});
test("stale or partial facial rigs reject for basic-character fallback and a complete retry succeeds", async (t) => {
  const { bytes } = await asset("chemistry");
  let nextBytes = bytes;
  t.mock.method(globalThis, "fetch", async () => ({ ok: true, arrayBuffer: async () => nextBytes.buffer.slice(nextBytes.byteOffset, nextBytes.byteOffset + nextBytes.byteLength) }));
  for(const [name, replacement] of [["attention_head","attention_dead"],["eye_left","eye_lost"],["eye_right","eye_wrong"]]) {
    nextBytes = Buffer.from(bytes);
    const offset = nextBytes.indexOf(Buffer.from(`"${name}"`)) + 1;
    assert.ok(offset > 20);
    assert.equal(name.length,replacement.length);
    nextBytes.write(replacement,offset);
    await assert.rejects(loadResearchRobot("chemistry"), /face rig missing/);
  }
  nextBytes = bytes;
  assert.ok((await loadResearchRobot("chemistry")).getObjectByName("attention_head"));
});
