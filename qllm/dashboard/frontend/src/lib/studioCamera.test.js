import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { inspectionPose, inspectionBlend } from "./studioCamera.js";
import { meetingRoomFocus } from "./meetingInteraction.js";
import { readFile } from "node:fs/promises";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { ROBOT_FIELDS, cloneResearchRobot } from "./researchRobot.js";
import { createResearcher, disposeScene } from "../components/researchPlanetScene.js";
import { conversationSubject, conversationPose } from "./studioCamera.js";
import { STUDIO_LAYOUT } from "./studioLayout.js";
import { createPaperGrip } from "./paperGrip.js";
import { createPaperGeometry } from "./paperSheet.js";
import { noteTarget } from "./carriedNote.js";

test("six real robot portraits and fallback fit their resting upper-body support at narrow/wide aspects", async () => {
  for (const field of [...ROBOT_FIELDS, "fallback"]) {
    let robot, template;
    if (field === "fallback") robot = createResearcher();
    else {
      const bytes = await readFile(new URL(`../../public/models/research-robots/${field}-robot.glb`, import.meta.url));
      template = (await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "")).scene;
      robot = cloneResearchRobot(template);
    }
    robot.scale.setScalar(template ? 1.75 : 1.6);
    robot.position.set(STUDIO_LAYOUT.researcher.x, .05, STUDIO_LAYOUT.researcher.z);
    robot.rotation.y = STUDIO_LAYOUT.researcher.turn;
    // Match the actual studio's rug grounding before deriving the portrait.
    robot.position.y += .01 - new T.Box3().setFromObject(robot).min.y;
    const subject = conversationSubject(robot), reference = conversationPose(subject, 1);
    const originalBounds = subject.bounds.clone();
    const paper = createPaperGrip(); paper.add(new T.Mesh(createPaperGeometry()));
    const table = new T.Mesh(new T.CylinderGeometry(STUDIO_LAYOUT.table.radius, STUDIO_LAYOUT.table.radius, .065, 32));
    table.position.set(STUDIO_LAYOUT.table.x, .43, STUDIO_LAYOUT.table.z);
    table.updateMatrixWorld(true);
    const ray = new T.Raycaster();
    for (const aspect of [.4, .45, .7, 1, 1.6, 2.4, 3.5]) {
      const pose = conversationPose(subject, aspect);
      assert.ok(pose, `${field}/${aspect}`);
      assert.ok(pose.position.distanceTo(reference.position) < 1e-12, "resize only refits lens");
      assert.ok(pose.position.y > .49 && Math.abs(pose.position.x) < 2 && Math.abs(pose.position.z) < 2, "clear table-top portrait location");
      assert.ok(pose.fov >= 40 && pose.fov <= 110);
      assert.ok(Math.abs(pose.quaternion.length() - 1) < 1e-12);
      const camera = new T.PerspectiveCamera(pose.fov, aspect, .05, 35);
      camera.position.copy(pose.position); camera.quaternion.copy(pose.quaternion); camera.updateMatrixWorld();
      const carry = noteTarget(camera, true, true, undefined, undefined, true);
      paper.position.copy(carry.position); paper.quaternion.copy(carry.quaternion); paper.scale.setScalar(carry.scale);
      paper.updateMatrixWorld(true);
      // GLB head hierarchy, or the fallback's actual .063-radius face sphere.
      const head = robot.userData.head || robot.children.find(c => c.geometry?.parameters.radius === .063);
      assert.ok(head, `${field}: actual face geometry exists`);
      let paperTop = -Infinity, faceBottom = Infinity;
      for (const [object, isPaper] of [[paper, true], [head, false]]) object.traverse(mesh => {
        const vertices = mesh.geometry?.attributes.position;
        if (vertices) for (let i = 0; i < vertices.count; i++) {
          const point = new T.Vector3().fromBufferAttribute(vertices, i).applyMatrix4(mesh.matrixWorld);
          const p = point.clone().project(camera);
          if (isPaper) {
            paperTop = Math.max(paperTop, p.y);
            ray.set(camera.position, point.clone().sub(camera.position).normalize());
            ray.far = point.distanceTo(camera.position);
            assert.equal(ray.intersectObject(table).length, 0,
              `${field}/${aspect}: tabletop cannot obscure the held sheet/grip`);
          }
          else faceBottom = Math.min(faceBottom, p.y);
        }
      });
      assert.ok(paperTop < faceBottom - .03, `${field}/${aspect}: face remains clear of carried paper/grip`);
      for (const x of [subject.bounds.min.x, subject.bounds.max.x])
        for (const y of [subject.bounds.min.y, subject.bounds.max.y])
          for (const z of [subject.bounds.min.z, subject.bounds.max.z]) {
            const point = new T.Vector3(x, y, z), ndc = point.clone().project(camera);
            assert.ok(Math.abs(ndc.x) <= .84 + 1e-12 && Math.abs(ndc.y) <= .84 + 1e-12, `${field}/${aspect}: ${ndc.toArray()}`);
            assert.ok(ndc.z > -1 && ndc.z < 1);
          }
    }
    // Later gaze/gesture poses do not drag the camera or change its snapshot.
    robot.rotation.y += .6; robot.userData.arm.rotation.x = -1;
    assert.deepEqual(conversationPose(subject, 1), reference);
    assert.deepEqual(subject.bounds, originalBounds);
    for (const aspect of [0, NaN, Infinity, .001]) assert.equal(conversationPose(subject, aspect), null);
    disposeScene(table); disposeScene(paper); disposeScene(robot); if (template) disposeScene(template);
  }
});

test("portrait refuses invalid support or facing without leaking nonfinite camera values", () => {
  const subject = { bounds: new T.Box3(), target: new T.Vector3(), forward: new T.Vector3(0, 0, 1) };
  assert.equal(conversationPose(subject, 1), null);
  subject.bounds = new T.Box3(new T.Vector3(-1,-1,-1), new T.Vector3(1,1,2));
  assert.equal(conversationPose(subject, 1), null, "support intersects near plane");
  subject.bounds.max.z = .1; subject.forward.y = .5;
  assert.equal(conversationPose(subject, 1), null);
  subject.forward.y = 0; subject.target.x = NaN;
  assert.equal(conversationPose(subject, 1), null);
});

const bounds = new T.Box3(
  new T.Vector3(-1.7, -0.2, -1.6),
  new T.Vector3(1.8, 2, 0.5),
);
test("inspection fit encloses deep volume within safe NDC and camera depth at narrow/wide aspects", () => {
  for (const aspect of [0.45, 0.7, 1, 1.6, 2.4]) {
    const pose = inspectionPose(bounds, aspect);
    const camera = new T.PerspectiveCamera(44, aspect, 0.05, 35);
    camera.position.copy(pose.position);
    camera.quaternion.copy(pose.quaternion);
    camera.updateMatrixWorld();
    for (const x of [bounds.min.x, bounds.max.x])
      for (const y of [bounds.min.y, bounds.max.y])
        for (const z of [bounds.min.z, bounds.max.z]) {
          const world = new T.Vector3(x, y, z),
            view = world.clone().applyMatrix4(camera.matrixWorldInverse),
            ndc = world.clone().project(camera);
          assert.ok(view.z < -0.05 && view.z > -35);
          assert.ok(
            Math.abs(ndc.x) <= 0.84 + 1e-12 && Math.abs(ndc.y) <= 0.84 + 1e-12,
          );
          assert.ok(ndc.toArray().every(Number.isFinite));
        }
    assert.ok(Math.abs(pose.quaternion.length() - 1) < 1e-12);
  }
});
test("invalid and empty bounds do not produce a camera pose", () => {
  assert.equal(inspectionPose(new T.Box3(), 1), null);
  assert.equal(inspectionPose(bounds, 0), null);
  assert.equal(inspectionPose(bounds, NaN), null);
  assert.equal(inspectionPose(bounds, 1, NaN), null);
  assert.equal(inspectionPose(bounds, 0.001), null);
});
test("inspection handoff reaches identical exact endpoints at four presentation rates", () => {
  const start = {
    position: new T.Vector3(3.35, 3, 4.2),
    quaternion: new T.Quaternion(),
    fov: 60,
  };
  const end = inspectionPose(bounds, 1.6);
  for (const hz of [30, 60, 120, 240]) {
    let pose;
    for (let frame = 0; frame <= Math.ceil(0.65 * hz); frame++)
      pose = inspectionBlend(start, end, frame / hz);
    assert.equal(pose.done, true);
    assert.ok(pose.position.distanceTo(end.position) < 1e-12);
    assert.ok(1 - Math.abs(pose.quaternion.dot(end.quaternion)) < 1e-12);
    assert.equal(pose.fov, end.fov);
  }
  assert.deepEqual(inspectionBlend(start, end, 0).position, start.position);
  assert.equal(inspectionBlend(start, end, 0).fov, start.fov);
  assert.equal(inspectionBlend(start, end, 0.325).fov, 52);
});
test("first OrbitControls update preserves the delivered inspection pose", () => {
  for (const aspect of [0.45, 1.3, 2.4]) {
    const pose = inspectionPose(bounds, aspect),
      camera = new T.PerspectiveCamera(44, aspect, 0.05, 35);
    camera.position.copy(pose.position);
    camera.quaternion.copy(pose.quaternion);
    const controls = new OrbitControls(camera, null);
    controls.target.copy(pose.target);
    controls.minDistance = 0.8;
    controls.maxDistance = Math.max(
      12,
      camera.position.distanceTo(pose.target) * 1.8,
    );
    controls.minPolarAngle = 0.15;
    controls.maxPolarAngle = Math.PI / 2.05;
    controls.update();
    assert.ok(camera.position.distanceTo(pose.position) < 1e-12);
    assert.ok(1 - Math.abs(camera.quaternion.dot(pose.quaternion)) < 1e-12);
  }
});
test("equipment opens room focus; reading a result does not toggle it; discussion reveals chat", () => {
  assert.equal(
    meetingRoomFocus(false, { type: "interact", id: "paper" }, null, null),
    true,
  );
  assert.equal(
    meetingRoomFocus(true, { type: "note", action: "read" }, null, "paper"),
    false,
  );
  assert.equal(
    meetingRoomFocus(false, { type: "close-material" }, "paper", null),
    true,
  );
  assert.equal(
    meetingRoomFocus(
      false,
      { type: "interact", id: "experiment" },
      null,
      "experiment",
    ),
    true,
  );
  assert.equal(
    meetingRoomFocus(false, { type: "send" }, null, "experiment"),
    true,
  );
  assert.equal(
    meetingRoomFocus(true, { type: "result" }, "experiment", "experiment"),
    true,
  );
  assert.equal(
    meetingRoomFocus(false, { type: "scrub" }, "experiment", "experiment"),
    false,
  );
  assert.equal(
    meetingRoomFocus(true, { type: "quote" }, "experiment", "experiment"),
    false,
  );
  assert.equal(
    meetingRoomFocus(true, { type: "close-material" }, "experiment", null),
    false,
  );
});
