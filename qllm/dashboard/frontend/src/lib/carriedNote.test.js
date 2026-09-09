import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { createPaperGrip } from "./paperGrip.js";
import { createPaperGeometry } from "./paperSheet.js";
import { disposeScene } from "../components/researchPlanetScene.js";
import {
  NOTE_TABLE,
  noteTarget,
  blendNotePose,
  createNoteMotion,
  canReturnNote,
} from "./carriedNote.js";

function camera(aspect = 1.5, fov = 64) {
  const c = new T.PerspectiveCamera(fov, aspect, 0.05, 35);
  c.position.set(0.83, 0.8, 1.65);
  c.rotation.set(-0.07, 0.75, 0, "YXZ");
  c.updateMatrixWorld();
  return c;
}
test("one real note fits a narrow/wide camera hand frame with finite normalized orientation", () => {
  const grip = createPaperGrip();
  grip.add(new T.Mesh(createPaperGeometry()));
  const vertices = [];
  grip.traverse(object => {
    const position = object.geometry?.attributes.position;
    if (position) for (let i = 0; i < position.count; i++)
      vertices.push(new T.Vector3().fromBufferAttribute(position, i));
  });
  // Both actual seated and standing lenses, including their scale-cap knees.
  for (const fov of [48, 64]) for (const aspect of [0.3, 0.45, 0.7, 0.77, 1, 1.1, 1.6, 2.4]) {
    const c = camera(aspect, fov),
      target = noteTarget(c, true, true);
    const matrix = new T.Matrix4().compose(
      target.position,
      target.quaternion,
      new T.Vector3().setScalar(target.scale),
    );
    // Inspect the actual sheet and grip, not obsolete thick-board bounds.
    for (const vertex of vertices) {
      const p = vertex.clone().applyMatrix4(matrix).project(c);
      assert.ok(p.toArray().every(Number.isFinite));
      assert.ok(
        Math.abs(p.x) < 0.98 && Math.abs(p.y) < 0.98 && Math.abs(p.z) < 1,
      );
    }
    assert.ok(Math.abs(target.quaternion.length() - 1) < 1e-12);
  }
  disposeScene(grip);
});
test("note pickup has identical exact endpoints and shared checkpoints at four presentation rates", () => {
  const from = noteTarget(camera(), false, false),
    to = noteTarget(camera(), true, true);
  for (const hz of [30, 60, 120, 240]) {
    let sample;
    for (let i = 0; i <= hz; i++) sample = blendNotePose(from, to, i / hz);
    assert.equal(sample.done, true);
    assert.ok(sample.position.distanceTo(to.position) < 1e-12);
    assert.ok(1 - Math.abs(sample.quaternion.dot(to.quaternion)) < 1e-12);
    assert.equal(sample.scale, to.scale);
    assert.deepEqual(
      blendNotePose(from, to, Math.round(0.2 * hz) / hz),
      blendNotePose(from, to, 0.2),
    );
  }
  assert.deepEqual(blendNotePose(from, to, 0).position, from.position);
});

test("conversation keeps the actual paper and grip in a lower safe band at portrait lenses", () => {
  const model = createPaperGrip();
  model.add(new T.Mesh(createPaperGeometry()));
  for (const fov of [40, 48, 64, 85, 110]) for (const aspect of [.45, .7, 1, 1.6, 2.4, 3.5]) {
    const c = camera(aspect, fov), before = c.matrixWorld.clone();
    const pose = noteTarget(c, true, true, undefined, undefined, true);
    model.position.copy(pose.position); model.quaternion.copy(pose.quaternion); model.scale.setScalar(pose.scale);
    model.updateMatrixWorld(true);
    model.traverse(object => {
      const vertices = object.geometry?.attributes.position;
      if (!vertices) return;
      for (let i = 0; i < vertices.count; i++) {
        const p = new T.Vector3().fromBufferAttribute(vertices, i).applyMatrix4(object.matrixWorld).project(c);
        assert.ok(p.toArray().every(Number.isFinite));
        assert.ok(Math.abs(p.x) < .98 && p.y > -.98 && p.y < -.30 && Math.abs(p.z) < 1,
          `${fov}/${aspect}: ${p.toArray()}`);
      }
    });
    assert.deepEqual(c.matrixWorld, before, "paper never changes camera");
  }
  disposeScene(model);
});

test("conversation cuts settle an interrupted pickup and restore carrying without dropping it", () => {
  for (const paused of [false, true]) for (const reducedMotion of [false, true]) {
    const paper = new T.Object3D(), update = createNoteMotion(paper);
    const input = {camera:camera(), firstPerson:true, held:false, revision:0, dt:.05};
    update(input);
    update({...input, held:true, revision:1});
    update({...input, held:true, revision:1});
    const held = {...input, held:true, revision:1, paused, reducedMotion};
    const talking = {...held, conversation:true, camera:camera(.6,85)};
    assert.equal(update(talking).done, true, "settle on the same frame as portrait cut");
    assert.deepEqual(paper.position, noteTarget(talking.camera,true,true,undefined,undefined,true).position);
    assert.equal(update(held).done, true, "return cut restores the existing carry frame");
    assert.deepEqual(paper.position, noteTarget(held.camera,true,true).position);
    assert.equal(held.revision, 1); assert.equal(held.held, true);
    assert.deepEqual(noteTarget(input.camera,false,true), noteTarget(input.camera,false,true,undefined,undefined,true));
    assert.deepEqual(noteTarget(input.camera,true,false), noteTarget(input.camera,true,false,undefined,undefined,true));
  }
});
test("return interrupted mid-pickup starts at the delivered pose, then docks exactly", () => {
  const paper = new T.Object3D(),
    update = createNoteMotion(paper);
  const input = {
    camera: camera(),
    firstPerson: true,
    held: false,
    revision: 0,
    dt: 0.05,
    paused: false,
    reducedMotion: false,
  };
  update(input);
  update({ ...input, held: true, revision: 1 });
  for (let i = 0; i < 4; i++) update({ ...input, held: true, revision: 1 });
  const interrupted = paper.position.clone();
  update({ ...input, held: false, revision: 2 });
  assert.ok(paper.position.distanceTo(interrupted) < 1e-12);
  for (let i = 0; i < 10; i++) update({ ...input, held: false, revision: 2 });
  assert.ok(paper.position.distanceTo(NOTE_TABLE) < 1e-12);
  assert.equal(paper.scale.x, 1);
});
test("pause holds pickup, a new paused action settles, and reduced motion or a mode cut cannot restart it", () => {
  const paper = new T.Object3D(),
    update = createNoteMotion(paper);
  let input = {
    camera: camera(),
    firstPerson: true,
    held: false,
    revision: 0,
    dt: 0.05,
    paused: false,
    reducedMotion: false,
  };
  update(input);
  input = { ...input, held: true, revision: 1 };
  update(input);
  update(input);
  const midpoint = paper.position.clone();
  for (let i = 0; i < 10; i++) update({ ...input, paused: true });
  assert.deepEqual(paper.position, midpoint);
  update({ ...input, reducedMotion: true });
  assert.deepEqual(
    paper.position,
    noteTarget(input.camera, true, true).position,
  );
  update({ ...input, paused: true, revision: 2, held: false });
  assert.deepEqual(paper.position, NOTE_TABLE);
  update({ ...input, firstPerson: false, revision: 3, held: true });
  assert.deepEqual(
    paper.position,
    noteTarget(input.camera, false, true).position,
  );
});
test("carried pose follows a changed camera after settlement; camera and paper have separate writers", () => {
  const paper = new T.Object3D(),
    update = createNoteMotion(paper),
    c = camera();
  const input = {
    camera: c,
    firstPerson: true,
    held: true,
    revision: 1,
    dt: 0.05,
  };
  update(input);
  c.position.x += 0.7;
  c.rotation.y += 0.4;
  c.updateMatrixWorld();
  const before = c.matrixWorld.clone();
  update(input);
  assert.deepEqual(paper.position, noteTarget(c, true, true).position);
  assert.deepEqual(c.matrixWorld, before);
});
test("put-down proximity is bounded and overview is an accessible alternative", () => {
  assert.equal(canReturnNote({ x: 0, z: 1.46 }, true), true);
  assert.equal(canReturnNote({ x: 0, z: 1.49 }, true), false);
  assert.equal(canReturnNote({ x: NaN, z: 1 }, true), false);
  assert.equal(canReturnNote({ x: 2, z: 1 }, false), true);
});

test("a result card uses its own bench rest pose without sharing note placement", () => {
  const object = new T.Object3D(),
    update = createNoteMotion(object);
  const restPose = {
    position: new T.Vector3(0.94, 0.68, -0.85),
    quaternion: new T.Quaternion(),
    scale: 1,
  };
  const input = {
    camera: camera(),
    firstPerson: true,
    held: false,
    revision: 0,
    dt: 0.05,
    restPose,
  };
  update(input);
  assert.deepEqual(object.position, restPose.position);
  for (let i = 0; i < 12; i++) update({ ...input, held: true, revision: 1 });
  assert.deepEqual(
    object.position,
    noteTarget(input.camera, true, true).position,
  );
  for (let i = 0; i < 12; i++) update({ ...input, revision: 2 });
  assert.deepEqual(object.position, restPose.position);
  assert.deepEqual(restPose.position.toArray(), [0.94, 0.68, -0.85]);
});
