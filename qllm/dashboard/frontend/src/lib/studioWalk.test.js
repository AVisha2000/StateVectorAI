import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { STUDIO_LAYOUT, VISITOR_SPAWN, WALK_OBSTACLES, WALK_EXTENT } from "./studioLayout.js";
import { walkContact, walkClearance, walkDisplacement, stepVisitor } from "./studioWalk.js";
import { createMeetingScene } from "../components/meetingScene.js";
import { disposeScene } from "../components/researchPlanetScene.js";
import { STUDIO_TYPES } from "./researchStudios.js";
import { captureResultCard } from "./resultCards.js";
import { simulateStudio } from "./studioSimulation.js";

test("actual furniture vertices fit their footprints in every studio, including pinned results", (t) => {
  // Canvas drawing is irrelevant to this CPU geometry check; rendered appearance
  // is checked separately in a real browser. All scene meshes are genuine Three.
  const previousDocument = globalThis.document;
  globalThis.document = { createElement: () => ({ getContext: () => ({
    clearRect() {}, fillRect() {}, strokeRect() {}, fillText() {}, beginPath() {}, arc() {}, fill() {},
    measureText: text => ({width:text.length*10}),
  }) }) };
  t.after(() => { if(previousDocument===undefined) delete globalThis.document; else globalThis.document=previousDocument; });
  const card=captureResultCard({result:simulateStudio("physics",45),sampleIndex:24});
  for(const studio of STUDIO_TYPES) {
    const scene=createMeetingScene(studio);
    scene.writeBoard({});
    assert.equal(scene.group.getObjectByName("Results side panel").visible,true);
    scene.writeBoard({pinnedResultIds:[card.id]},[card]);
    assert.equal(scene.group.getObjectByName("Results side panel").visible,true);
    scene.group.updateMatrixWorld(true);
    for(const id of ["table","researcher-chair","visitor-chair","board","bookcase"]) {
      const object=scene.group.getObjectByName(`Furniture ${id}`);
      const footprint={...WALK_OBSTACLES.find(o=>o.id===id),padding:0};
      let vertices=0;
      object.traverse(mesh=>{
        const positions=mesh.geometry?.attributes.position;
        if(!positions)return;
        for(let i=0;i<positions.count;i++) {
          const point=new T.Vector3().fromBufferAttribute(positions,i).applyMatrix4(mesh.matrixWorld);
          assert.ok(walkContact(point,footprint).gap<=1e-6,`${studio.id}/${id} vertex ${i}`);
          vertices++;
        }
      });
      assert.ok(vertices>0,id);
    }
    const board=scene.group.getObjectByName("Furniture board");
    const panelPoint=board.localToWorld(new T.Vector3(STUDIO_LAYOUT.board.pinX,0,0));
    assert.ok(walkClearance(panelPoint)<0,"pinned panel must not be walkable");
    const textures=new Set();
    scene.group.traverse(object=>{
      for(const material of [].concat(object.material||[])) {
        for(const value of Object.values(material))if(value?.isTexture)textures.add(value);
      }
    });
    // Exercise the complete meeting/table integration, not just the pavilion.
    const timberTextures=[...textures].filter(texture=>texture.name.startsWith("Timber "));
    assert.equal(timberTextures.length,2,`${studio.id}: shared room timber`);
    let retiredTimber=0;
    timberTextures.forEach(texture=>texture.addEventListener("dispose",()=>retiredTimber++));
    disposeScene(scene.group);
    assert.equal(retiredTimber,2);
  }
});

test("rounded furniture footprints follow the scene's positive Y rotation and retain clearance", () => {
  const box = {x:.3,z:-.4,turn:.7,width:.8,depth:.2,padding:.11};
  const world = (x,z) => ({x:box.x+Math.cos(box.turn)*x+Math.sin(box.turn)*z,z:box.z-Math.sin(box.turn)*x+Math.cos(box.turn)*z});
  assert.ok(Math.abs(walkContact(world(0,0),box).gap+.21)<1e-12);
  assert.ok(Math.abs(walkContact(world(.51,0),box).gap)<1e-12);
  const corner = walkContact(world(.4+.11/Math.sqrt(2),.1+.11/Math.sqrt(2)),box);
  assert.ok(Math.abs(corner.gap)<1e-12);
  assert.ok(Math.abs(Math.hypot(corner.x,corner.z)-1)<1e-12);
  assert.equal(WALK_OBSTACLES.length,7);
  assert.ok(walkClearance(VISITOR_SPAWN)>.1);
  for (const obstacle of WALK_OBSTACLES) assert.ok(walkContact(obstacle,obstacle).gap<0);
});

test("walking cannot enter chairs, bookshelf or the thin rotated board, and idle input never creeps", () => {
  for (const id of ["visitor-chair","bookcase","board"]) {
    const obstacle = WALK_OBSTACLES.find(o=>o.id===id);
    // Approach an exposed face, not through a neighbouring piece of furniture.
    const forward = new T.Vector3(...(id==="visitor-chair"?[0,0,-1]:id==="bookcase"?[-1,0,0]:[0,0,1]))
      .applyAxisAngle(new T.Vector3(0,1,0),obstacle.turn);
    let point = {x:obstacle.x+forward.x*.6,z:obstacle.z+forward.z*.6};
    assert.ok(walkClearance(point)>=0,`${id} initial approach`);
    for(let i=0;i<60;i++) {
      const before=point;
      point=walkDisplacement(point,-forward.x*.09,-forward.z*.09);
      assert.ok(walkClearance(point)>=-1e-7,id);
      assert.ok(Math.hypot(point.x-before.x,point.z-before.z)<=.090001);
    }
    assert.ok(walkContact(point,obstacle).gap<.02,id);
    const stopped={...point};
    for(let i=0;i<500;i++) point=stepVisitor(point,.4,0,0,.05);
    assert.deepEqual(point,stopped);
  }
});

test("seeded movement retains all simultaneous contacts, bounds, finite positions and displacement budget", () => {
  let seed=74921, point={...VISITOR_SPAWN};
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  for(let i=0;i<6000;i++) {
    const before={...point}, yaw=random()*Math.PI*2, seconds=random()*.09, fast=random()>.5;
    const forward=Math.round(random()*2)-1, right=Math.round(random()*2)-1;
    point=stepVisitor(point,yaw,forward,right,seconds,fast);
    assert.ok(Number.isFinite(point.x)&&Number.isFinite(point.z));
    assert.ok(walkClearance(point)>=-1e-7,`${i}: ${JSON.stringify(point)}`);
    const distance=(fast?1.8:1.05)*Math.min(seconds,.05)*(forward||right?1:0);
    assert.ok(Math.hypot(point.x-before.x,point.z-before.z)<=distance+1e-6);
  }
  const original={...point};
  stepVisitor(point,0,1,0,.05);
  assert.deepEqual(point,original);
  assert.deepEqual(stepVisitor(point,NaN,1,0,.05),point);
  assert.deepEqual(stepVisitor({x:NaN,z:0},0,1,0,.05),VISITOR_SPAWN);
});

test("oblique input slides along the board and backing away releases contact", () => {
  const board=WALK_OBSTACLES.find(o=>o.id==="board");
  const c=Math.cos(board.turn),s=Math.sin(board.turn);
  const local=(x,z)=>({x:board.x+c*x+s*z,z:board.z-s*x+c*z});
  let point=local(.35,.16), initial={...point};
  assert.ok(walkClearance(point)>=0);
  for(let i=0;i<30;i++) point=walkDisplacement(point,-c*.008-s*.006,s*.008-c*.006);
  assert.ok(Math.hypot(point.x-initial.x,point.z-initial.z)>.15);
  assert.ok(walkClearance(point)>=-1e-7);
  const before=walkContact(point,board).gap;
  point=walkDisplacement(point,s*.09,c*.09);
  assert.ok(walkContact(point,board).gap>before+.08);
});

test("matched held-input paths converge across presentation cadences and geometric step halving", () => {
  const traces=[];
  for(const hz of [30,60,120,240]) {
    let point={...VISITOR_SPAWN}; const samples=[];
    for(let i=1;i<=hz*8;i++) {
      const segment=Math.floor((i-1)/hz/2);
      const inputs=[[.75,1,0],[.75,0,-1],[0,1,0],[0,0,1]][segment];
      point=stepVisitor(point,...inputs,1/hz);
      if(i%hz===0) samples.push(point);
    }
    traces.push(samples);
  }
  for(const trace of traces) trace.forEach((p,i)=>assert.ok(Math.hypot(p.x-traces.at(-1)[i].x,p.z-traces.at(-1)[i].z)<.02));
  let coarse={...VISITOR_SPAWN},fine={...VISITOR_SPAWN};
  for(let i=0;i<600;i++) {
    const yaw=i<200?.75:i<400?2:4;
    const dx=-Math.sin(yaw)*.00875,dz=-Math.cos(yaw)*.00875;
    coarse=walkDisplacement(coarse,dx,dz,.008);
    fine=walkDisplacement(fine,dx,dz,.004);
    assert.ok(Math.hypot(coarse.x-fine.x,coarse.z-fine.z)<.01);
  }
});

test("every main interaction remains reachable from spawn around the furniture", () => {
  const spacing=.045, queue=[{...VISITOR_SPAWN}], seen=new Set(["0,0"]), reached=new Set();
  const targets=[STUDIO_LAYOUT.table,STUDIO_LAYOUT.researcher,STUDIO_LAYOUT.board,STUDIO_LAYOUT.lab];
  const offsets=[[1,0],[-1,0],[0,1],[0,-1]];
  const cells=[[0,0]];
  for(let cursor=0;cursor<queue.length;cursor++) {
    const p=queue[cursor], [cx,cz]=cells[cursor];
    targets.forEach(t=>{if(Math.hypot(p.x-t.x,p.z-t.z)<(t.id==="lab"?.88:.75)) reached.add(t.id);});
    for(const [dx,dz] of offsets) {
      const nx=cx+dx,nz=cz+dz,key=`${nx},${nz}`;
      if(seen.has(key))continue;
      seen.add(key);
      const next={x:VISITOR_SPAWN.x+nx*spacing,z:VISITOR_SPAWN.z+nz*spacing};
      if(Math.hypot(next.x,next.z)>WALK_EXTENT || walkClearance(next)<.005) continue;
      const moved=walkDisplacement(p,next.x-p.x,next.z-p.z);
      if(Math.hypot(moved.x-next.x,moved.z-next.z)>.00001)continue;
      queue.push(next);cells.push([nx,nz]);
    }
  }
  assert.deepEqual([...reached].sort(),targets.map(t=>t.id).sort());
});
