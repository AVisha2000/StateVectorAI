import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { readFile } from "node:fs/promises";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { cloneResearchRobot, ROBOT_FIELDS } from "./researchRobot.js";
import { gazeTarget, followGaze, faceExpression, researcherArm, createRobotPresence } from "./robotPresence.js";
import { initialGestureClock, advanceGestureClock } from "./meetingInteraction.js";

test("gaze is finite, anatomically bounded and does not turn backward through the neck", () => {
  for (const x of [-20, -.2, 0, .2, 20]) for (const y of [-20, .416, 20]) for (const z of [-20, 0, .1, 20]) {
    const target = gazeTarget({x,y,z});
    assert.ok(Number.isFinite(target.yaw) && Number.isFinite(target.pitch));
    assert.ok(Math.abs(target.yaw) <= .7 && Math.abs(target.pitch) <= .22);
    if (z <= 0) assert.deepEqual(target, {yaw:0,pitch:0});
  }
  assert.deepEqual(gazeTarget({x:NaN,y:0,z:1}), {yaw:0,pitch:0});
  assert.deepEqual(gazeTarget({x:0,y:.416,z:1}), {yaw:0,pitch:-0});
  assert.ok(gazeTarget({x:1,y:.7,z:1}).yaw > 0);
  assert.ok(gazeTarget({x:1,y:.7,z:1}).pitch < 0);
});
test("held gaze agrees analytically at 30/60/120/240 Hz and smooth end-sampled input converges within .015 rad", () => {
  const traces = [];
  for (const hz of [30,60,120,240]) {
    let fixed = {yaw:0,pitch:0}, varying = {yaw:0,pitch:0};
    const trace = [];
    for(let i=1;i<=hz*4;i++) {
      const t = i/hz;
      fixed = followGaze(fixed, {yaw:.6,pitch:.2}, 1/hz);
      varying = followGaze(varying, {yaw:.5*Math.sin(t*1.3),pitch:.18*Math.cos(t)}, 1/hz);
      if(i%(hz/2)===0) {
        assert.ok(Math.abs(fixed.yaw - .6*(1-Math.exp(-8*t))) < 1e-10);
        assert.ok(Math.abs(fixed.pitch - .2*(1-Math.exp(-8*t))) < 1e-10);
        trace.push(varying);
      }
    }
    traces.push(trace);
  }
  for(const trace of traces) trace.forEach((pose,i)=> {
    assert.ok(Math.abs(pose.yaw-traces.at(-1)[i].yaw)<.015);
    assert.ok(Math.abs(pose.pitch-traces.at(-1)[i].pitch)<.015);
  });
});
test("blink and nod are seekable with bounded exact terminal poses, and greeting arm returns to rest", () => {
  assert.equal(faceExpression(4.4,"idle",10).eyeScale,1);
  assert.ok(Math.abs(faceExpression(4.49,"idle",10).eyeScale-.14)<1e-12);
  assert.ok(Math.abs(faceExpression(4.58,"idle",10).eyeScale-1)<1e-12);
  for(const hz of [30,60,120,240]) for(let i=0;i<=hz*6;i++) {
    const t=i/hz, pose=faceExpression(t,"greet",t);
    assert.ok(pose.eyeScale>=.14-1e-12 && pose.eyeScale<=1);
    assert.ok(pose.nod>=0 && pose.nod<=.12);
    if(t>=1.15) assert.equal(pose.nod,0);
  }
  assert.deepEqual(faceExpression(4.49,"talk",.575,true),{eyeScale:1,nod:0});
  assert.equal(researcherArm("greet",2),-.15);
  assert.equal(researcherArm("talk",10),-.15);
  assert.equal(researcherArm("coffee",1.8),0);
});

test("a remounted room restores the old greeting as settled but animates the next new acknowledgement", () => {
  let clock = initialGestureClock(8);
  for(let i=0;i<20;i++) clock = advanceGestureClock(clock,.05,8,false,false);
  assert.equal(clock.age,10);
  assert.equal(faceExpression(clock.elapsed,"greet",clock.age).nod,0);
  assert.equal(researcherArm("greet",clock.age),-.15);
  clock = advanceGestureClock(clock,.05,9,false,false);
  for(let i=0;i<10;i++) clock = advanceGestureClock(clock,.05,9,false,false);
  assert.ok(faceExpression(clock.elapsed,"talk",clock.age).nod>.1);
});

test("all actual rigs move head and eyes without moving garment or shoulder; pause/reduced motion and fallback remain safe", async () => {
  for (const field of ROBOT_FIELDS) {
    const bytes = await readFile(new URL(`../../public/models/research-robots/${field}-robot.glb`, import.meta.url));
    const {scene} = await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),"");
    const robot = cloneResearchRobot(scene), untouched = cloneResearchRobot(scene);
    const head = robot.userData.head;
    assert.ok(head && robot.userData.eyes.length===2);
    assert.ok(Math.abs(head.position.y-.362)<1e-6);
    assert.ok(robot.userData.eyes.every(eye=>eye.parent===head));
    assert.ok(!head.getObjectByName("gesture_arm"));
    const body = robot.getObjectByName("Body_Discipline_garment");
    robot.updateMatrixWorld(true);
    const bodyBefore = body.matrixWorld.clone(), shoulderBefore = robot.userData.arm.matrixWorld.clone();
    const faceBefore = robot.userData.eyes[0].getWorldPosition(new T.Vector3());
    const camera = new T.PerspectiveCamera(); camera.position.set(.6,.6,1);
    const present = createRobotPresence(robot);
    const args={camera,firstPerson:true,elapsed:4.49,gesture:{type:"greet"},age:.575,dt:.5,paused:false,reducedMotion:false};
    const result=present(args);
    assert.ok(result.yaw>.4 && result.eyeScale<.15);
    robot.updateMatrixWorld(true);
    assert.ok(body.matrixWorld.equals(bodyBefore));
    assert.ok(robot.userData.arm.matrixWorld.equals(shoulderBefore));
    assert.ok(robot.userData.eyes[0].getWorldPosition(new T.Vector3()).distanceTo(faceBefore)>.005);
    assert.ok(Math.abs(head.quaternion.length()-1)<1e-12);
    camera.position.x=-.6;
    assert.deepEqual(present({...args,paused:true,elapsed:5,age:2}),result);
    assert.deepEqual(present({...args,paused:true,reducedMotion:true}),{yaw:0,pitch:0,eyeScale:1});
    assert.equal(untouched.userData.eyes[0].scale.y,1);
    assert.ok(Math.abs(untouched.userData.head.rotation.y)<1e-12);
    assert.deepEqual(present({...args,firstPerson:false,age:10,elapsed:6}),{yaw:0,pitch:0,eyeScale:1});
  }
  const fallback=new T.Group();
  assert.equal(createRobotPresence(fallback)({}),null);
});
