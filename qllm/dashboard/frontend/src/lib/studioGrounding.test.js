import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import * as T from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { ROBOT_FIELDS } from "./researchRobot.js";
import { STUDIO_TYPES } from "./researchStudios.js";
import { STUDIO_LAYOUT } from "./studioLayout.js";
import { conversationPose } from "./studioCamera.js";
import { offeredNotePose } from "./paperOffer.js";
import { createMeetingScene } from "../components/meetingScene.js";
import { disposeScene } from "../components/researchPlanetScene.js";

function canvasDouble(t) {
  const previous = globalThis.document;
  globalThis.document = { createElement: () => ({ getContext: () => ({
    clearRect() {}, fillRect() {}, strokeRect() {}, fillText() {}, beginPath() {}, arc() {}, fill() {},
    measureText: text => ({ width: text.length * 10 }),
  }) }) };
  t.after(() => { if (previous === undefined) delete globalThis.document; else globalThis.document = previous; });
}

test("every actual robot and fallback stands on the real rug in overview, first-person and gestures", async t => {
  canvasDouble(t);
  for (const field of [...ROBOT_FIELDS, "fallback"]) {
    let template = null;
    if (field !== "fallback") {
      const bytes = await readFile(new URL(`../../public/models/research-robots/${field}-robot.glb`, import.meta.url));
      template = (await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "")).scene;
    }
    const studio = STUDIO_TYPES.find(s => s.id === field) || STUDIO_TYPES.find(s => s.id === "physics");
    const room = createMeetingScene(studio, template);
    const robot = room.group.getObjectByName("Furniture researcher"), rug = room.group.getObjectByName("Studio rug");
    room.group.updateMatrixWorld(true);
    const top = new T.Box3().setFromObject(rug).max.y;
    const root = robot.position.clone();
    assert.equal(root.x, STUDIO_LAYOUT.researcher.x); assert.equal(root.z, STUDIO_LAYOUT.researcher.z);
    assert.equal(robot.scale.x, template ? 1.75 : 1.6);
    for (const firstPerson of [false, true]) {
      for (const type of ["idle", "talk", "board", "paper"]) {
        for (const age of [0, .5, 1, 3]) {
          room.animate(age, { type }, age, 45, firstPerson, null, null);
          room.presentArm({ offered: type === "paper", gesture: type, age, dt: .05, paused: false, reducedMotion: false });
          room.group.updateMatrixWorld(true);
          const bounds = new T.Box3().setFromObject(robot);
          assert.ok(Math.abs(bounds.min.y - top) <= 1e-6, `${field}/${type}/${age}: ${bounds.min.y - top}`);
          assert.deepEqual(robot.position, root, "animation never lifts or resets the grounded root");
          let contacts = 0;
          robot.traverse(mesh => {
            const vertices = mesh.geometry?.attributes.position;
            if (!vertices) return;
            for (let i = 0; i < vertices.count; i++) {
              const point = new T.Vector3().fromBufferAttribute(vertices, i).applyMatrix4(mesh.matrixWorld);
              assert.ok(point.y >= top - 1e-6, "no character mesh penetrates support");
              if (point.y <= top + 1e-6) {
                const local = rug.worldToLocal(point);
                assert.ok(Math.hypot(local.x, local.z) <= .99 + 1e-6, "sole rests within actual elliptical rug");
                contacts++;
              }
            }
          });
          assert.ok(contacts > 0);
        }
      }
    }
    for (const aspect of [.45, 1, 1.7, 3.5]) assert.ok(conversationPose(room.conversation, aspect), `${field} portrait`);
    const paper = offeredNotePose(robot);
    assert.ok(paper.position.toArray().every(Number.isFinite));
    assert.ok(paper.position.y > top + .2, "offered paper follows grounded hand above floor");
    disposeScene(room.group); if (template) disposeScene(template);
  }
});

test("an empty detailed asset cannot turn grounding into an infinite root translation", t => {
  canvasDouble(t);
  const room = createMeetingScene(STUDIO_TYPES.find(s => s.id === "physics"), new T.Group());
  assert.ok(room.group.getObjectByName("Furniture researcher").position.toArray().every(Number.isFinite));
  assert.equal(conversationPose(room.conversation, 1), null);
  disposeScene(room.group);
});
