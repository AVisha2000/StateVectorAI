import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { readFile } from "node:fs/promises";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { cloneResearchRobot, ROBOT_FIELDS } from "./researchRobot.js";
import { createResearcher, disposeScene } from "../components/researchPlanetScene.js";
import { createOfferingArm, OFFER_PITCH, offeredNotePose } from "./paperOffer.js";
import { initialMeeting, meetingEvent, meetingRoomFocus } from "./meetingInteraction.js";
import { createNoteMotion, NOTE_TABLE } from "./carriedNote.js";
import { studioPrompt } from "./studioPrompt.js";
import { createPaperGeometry, PAPER_BOTTOM, PAPER_TOP } from "./paperSheet.js";

const studio = { id: "physics", name: "Lyra Vale", title: "Compare boundaries" };
const event = (state, action) => meetingEvent(state, studio, { type: "note", action });
const robot = () => {
  const model = new T.Group(), arm = new T.Group();
  arm.position.set(.097, .323, 0); model.add(arm);
  model.userData = { arm, robotAsset: true };
  model.position.set(-.78, .05, .15); model.rotation.y = .72; model.scale.setScalar(1.75);
  return model;
};

test("request offers without taking, reading, storing a held result or erasing context; repeat is idempotent", () => {
  const before = { ...initialMeeting(studio), draft: "My question", quote: "My passage", heldResultId: 3, resultCards: [{id:3}] };
  const offered = event(before, "offer");
  assert.equal(offered.noteOffered, true); assert.equal(offered.noteHeld, false);
  assert.equal(offered.material, null); assert.equal(offered.heldResultId, 3);
  for (const key of ["draft", "quote", "board", "resultCards", "resultRevision"]) assert.equal(offered[key], before[key]);
  assert.deepEqual(event(offered, "offer"), offered);
  const taken = event(offered, "take");
  assert.equal(taken.noteOffered, false); assert.equal(taken.noteHeld, true);
  assert.equal(taken.heldResultId, null); assert.equal(taken.resultCards, before.resultCards);
  assert.equal(taken.noteRevision, 2); assert.equal(taken.material, null);
  assert.equal(event(taken, "read").material, "paper");
  assert.equal(event(taken, "return").noteHeld, false);
  assert.equal(meetingRoomFocus(false, {type:"note",action:"offer"}, "board", null), true);
  assert.equal(studioPrompt("first-person", "offered-note"), "Take offered note");
  assert.equal(studioPrompt("overview", "offered-note"), null);
});

test("decline, direct paper pickup, launcher and result pickup resolve offers without duplicate possession", () => {
  const offered = event(initialMeeting(studio), "offer");
  const declined = event(offered, "decline");
  assert.equal(declined.noteOffered, false); assert.equal(declined.noteHeld, false);
  assert.deepEqual(event(declined, "decline"), declined);
  const picked = meetingEvent(offered, studio, {type:"interact",id:"paper"});
  assert.equal(picked.noteHeld, true); assert.equal(picked.noteOffered, false);
  const launcher = meetingEvent(offered, studio, {type:"launcher",action:"open"});
  assert.equal(launcher.noteOffered, false); assert.equal(launcher.noteRevision, 2);
  const carrying = meetingEvent({...offered,resultCards:[{id:1}]}, studio, {type:"result-card",action:"carry",id:1});
  assert.equal(carrying.noteOffered, false); assert.equal(carrying.noteRevision, 2);
});

test("typed paper requests offer too; discussion of a quoted passage and held-note reading do not re-offer", () => {
  const before = {...initialMeeting(studio), heldResultId:1, resultCards:[{id:1}], draft:"Show me a paper"};
  const offered = meetingEvent(before,studio,{type:"send"});
  assert.equal(offered.noteOffered,true); assert.equal(offered.noteHeld,false);
  assert.equal(offered.material,null); assert.equal(offered.heldResultId,1);
  assert.equal(offered.draft,""); assert.match(offered.messages.at(-1).text,/Take it from my hand/);
  const repeated = meetingEvent(offered,studio,{type:"send",text:"Show me a paper"});
  assert.equal(repeated.noteRevision,offered.noteRevision);
  const held = event(offered,"take");
  const read = meetingEvent(held,studio,{type:"send",text:"Read the paper"});
  assert.equal(read.noteOffered,false); assert.equal(read.noteHeld,true); assert.equal(read.material,"paper");
  const quoted = meetingEvent({...held,quote:"An assumption"},studio,{type:"send",text:"What does this paper assume?"});
  assert.equal(quoted.noteOffered,false); assert.equal(quoted.noteRevision,held.noteRevision);
  assert.equal(meetingRoomFocus(false,{type:"send"},null,null,true),true);
});

test("offer arm and same paper agree at shared checkpoints across 30/60/120/240 Hz", () => {
  const traces = [];
  for (const hz of [30,60,120,240]) {
    const model = robot(), arm = createOfferingArm(model), paper = new T.Object3D(), motion = createNoteMotion(paper);
    const camera = new T.PerspectiveCamera(64,1.7,.05,35);
    const input = {gesture:"idle",age:10,dt:1/hz,paused:false,reducedMotion:false};
    arm({...input,offered:false});
    motion({camera,firstPerson:true,held:false,revision:0,dt:0});
    const samples = [];
    for(let i=0;i<=hz;i++) {
      const pose = arm({...input,offered:true});
      const hand = offeredNotePose(model);
      motion({camera,firstPerson:true,held:false,revision:1,restPose:hand,dt:1/hz});
      if(i%(hz/10)===0) samples.push([pose.pitch,...paper.position.toArray()]);
      assert.ok(Math.abs(hand.quaternion.length()-1)<1e-9);
      if(i/hz>=.5) { assert.equal(pose.pitch,OFFER_PITCH); assert.ok(paper.position.distanceTo(hand.position)<1e-9); }
    }
    traces.push(samples);
  }
  for(const trace of traces) trace.forEach((row,i)=>row.forEach((v,j)=>assert.ok(Math.abs(v-traces[0][i][j])<1e-9)));
});

test("offer interruption captures delivered arm pitch; pause, reduced motion and restore cannot replay it", () => {
  const model = robot(), arm = createOfferingArm(model);
  const input = {gesture:"idle",age:10,dt:.05,paused:false,reducedMotion:false};
  arm({...input,offered:false}); arm({...input,offered:true});
  for(let i=0;i<4;i++) arm({...input,offered:true});
  const midpoint = model.userData.arm.rotation.x;
  for(let i=0;i<10;i++) arm({...input,offered:true,paused:true});
  assert.equal(model.userData.arm.rotation.x, midpoint);
  arm({...input,offered:false}); assert.equal(model.userData.arm.rotation.x,midpoint);
  arm({...input,offered:false,reducedMotion:true}); assert.equal(model.userData.arm.rotation.x,-.15);
  arm({...input,offered:false,gesture:"coffee",paused:true}); assert.equal(model.userData.arm.rotation.x,0);
  arm({...input,offered:true,paused:true}); assert.equal(model.userData.arm.rotation.x,OFFER_PITCH);
  const restore = createOfferingArm(robot());
  assert.deepEqual(restore({...input,offered:true}),{pitch:OFFER_PITCH,done:true});
  assert.deepEqual(NOTE_TABLE.toArray(),[0,.473,.37]);
});

test("offered paper contacts actual hand geometry below every GLB face and the fallback head", async () => {
  for(const field of [...ROBOT_FIELDS, "fallback"]) {
    let template;
    if(field !== "fallback") {
      const bytes = await readFile(new URL(`../../public/models/research-robots/${field}-robot.glb`,import.meta.url));
      template = (await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),"")).scene;
    }
    const model = template ? cloneResearchRobot(template) : createResearcher();
    model.position.set(-.78,.05,.15); model.rotation.y=.72;
    model.scale.setScalar(template ? 1.75 : 1.6);
    createOfferingArm(model)({offered:true,gesture:"idle",age:10,dt:0,paused:false,reducedMotion:false});
    const pose = offeredNotePose(model);
    const matrix = new T.Matrix4().compose(pose.position,pose.quaternion,new T.Vector3(1,1,1));
    const inverse = matrix.clone().invert(), handInverse = model.userData.arm.matrixWorld.clone().invert();
    const paperGeometry = createPaperGeometry(), sheet = paperGeometry.boundingBox;
    // This entire right-half patch is flat in the actual sheet. A thin slab
    // can intersect a triangle between its vertices; test the real surfaces.
    const contactPatch = new T.Box3(new T.Vector3(0,PAPER_BOTTOM,-.125),new T.Vector3(.095,PAPER_TOP,.125));
    const detachedPatch = contactPatch.clone().translate(new T.Vector3(.5,0,0));
    let contacts = 0, displacedContacts = 0;
    model.userData.arm.traverse(mesh=>{
      const attribute = mesh.geometry?.attributes.position;
      if(!attribute) return;
      const index = mesh.geometry.index;
      for(let i=0;i<(index?.count ?? attribute.count);i+=3) {
        const points = [0,1,2].map(j => new T.Vector3().fromBufferAttribute(attribute,index ? index.getX(i+j) : i+j).applyMatrix4(mesh.matrixWorld));
        if(points.some(point => point.clone().applyMatrix4(handInverse).y > -.1)) continue;
        const triangle = new T.Triangle(...points.map(point => point.applyMatrix4(inverse)));
        if(contactPatch.intersectsTriangle(triangle)) contacts++;
        if(detachedPatch.intersectsTriangle(triangle)) displacedContacts++;
      }
    });
    assert.ok(contacts > 0,`${field}: paper must touch actual palm/fingers`);
    assert.equal(displacedContacts,0,`${field}: detached paper is rejected`);
    const head = new T.Box3();
    if(model.userData.head) head.setFromObject(model.userData.head);
    else for(const part of model.children) if(part.isMesh && part.position.y >= .3) head.union(new T.Box3().setFromObject(part));
    assert.ok(sheet.clone().applyMatrix4(matrix).max.y < head.min.y,`${field}: paper stays below face`);
    assert.ok(pose.position.toArray().every(Number.isFinite));
    assert.ok(Math.abs(pose.quaternion.length()-1)<1e-9);
    const initialPosition = pose.position.clone();
    model.userData.arm.rotation.x += .2;
    assert.ok(offeredNotePose(model).position.distanceTo(initialPosition)>.02,`${field}: pose consumes current arm frame`);
    paperGeometry.dispose();
    disposeScene(model); if(template) disposeScene(template);
  }
});
