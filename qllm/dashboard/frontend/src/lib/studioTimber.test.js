import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { timberField, createTimberTextures, timberMaterial, timberUV, TIMBER_SCALE } from "./studioTimber.js";
import { createStudioInterior } from "../components/studioInterior.js";
import { disposeScene } from "../components/researchPlanetScene.js";

test("timber field is periodic, bounded and exactly repeatable", () => {
  let low=1,high=0;
  for(let i=0;i<1000;i++) {
    const u=i*.037,v=i*.061,a=timberField(u,v);
    assert.equal(a,timberField(u,v));
    assert.ok(Number.isFinite(a)&&a>=0&&a<=1);
    assert.ok(Math.abs(a-timberField(u+1,v))<1e-11);
    assert.ok(Math.abs(a-timberField(u,v+1))<1e-11);
    low=Math.min(low,a); high=Math.max(high,a);
  }
  assert.ok(high-low>.7,"resolvable structural variation");
});

test("two shared generated textures preserve channel semantics, filtering and allocation bounds", () => {
  const a=createTimberTextures(),b=createTimberTextures();
  assert.equal(a.color.colorSpace,T.LinearSRGBColorSpace);
  assert.equal(a.surface.colorSpace,T.NoColorSpace);
  assert.equal(a.color.image.data.byteLength+a.surface.image.data.byteLength,524288);
  for(const key of ["color","surface"]) {
    const map=a[key];
    assert.deepEqual(map.image.data,b[key].image.data);
    assert.notEqual(map,b[key]);
    assert.equal(map.generateMipmaps,true);
    assert.equal(map.minFilter,T.LinearMipmapLinearFilter);
    assert.equal(map.magFilter,T.LinearFilter);
    assert.equal(map.wrapS,T.RepeatWrapping); assert.equal(map.wrapT,T.RepeatWrapping);
    assert.equal(map.anisotropy,4);
  }
  for(let i=0;i<a.color.image.data.length;i+=4) {
    const color=a.color.image.data,surface=a.surface.image.data,height=surface[i]/255;
    assert.ok(Math.abs(color[i]/255-(.66+.28*height))<.004);
    assert.ok(Math.abs(surface[i+1]/255-(1-.17*height))<.004);
    assert.equal(color[i],color[i+1]); assert.equal(color[i],color[i+2]);
    assert.equal(color[i+3],255); assert.equal(surface[i+3],255);
  }
  const material=timberMaterial("#c39e71",a);
  assert.equal(material.metalness,0);
  assert.equal(material.map,a.color);
  assert.equal(material.bumpMap,material.roughnessMap);
  assert.ok(material.bumpScale>0&&material.bumpScale<=.0005);
  assert.ok(material.roughness>=.7);
  material.dispose(); Object.values(a).forEach(t=>t.dispose()); Object.values(b).forEach(t=>t.dispose());
});

test("timber UVs preserve physical scale, non-degenerate faces and exact mesh geometry", () => {
  for(const axis of ["x","y","z","table"]) {
    const geometry=axis==="table"?new T.CylinderGeometry(.47,.47,.065,32):new T.BoxGeometry(1.35,.11,.217);
    const original=geometry.clone(),positions=geometry.attributes.position.array.slice();
    timberUV(geometry,axis,"member-1");
    assert.deepEqual(geometry.attributes.position.array,positions);
    assert.deepEqual(geometry.index.array,original.index.array);
    const uv=geometry.attributes.uv,indices=geometry.index.array;
    assert.ok(uv.array.every(Number.isFinite));
    for(let i=0;i<indices.length;i+=3) {
      const [a,b,c]=indices.slice(i,i+3);
      const area=(uv.getX(b)-uv.getX(a))*(uv.getY(c)-uv.getY(a))-(uv.getY(b)-uv.getY(a))*(uv.getX(c)-uv.getX(a));
      assert.ok(Math.abs(area)>1e-8,`${axis}: triangle ${i/3}`);
    }
    timberUV(original,axis,"member-2");
    assert.notDeepEqual(uv.array,original.attributes.uv.array,"stable member variation");
    geometry.dispose(); original.dispose();
  }
  const plane=new T.PlaneGeometry(1.35,.217).rotateX(-Math.PI/2);
  timberUV(plane,"x");
  const values=plane.attributes.uv;
  assert.ok(Math.abs(Math.abs(values.getX(1)-values.getX(0))-1.35/TIMBER_SCALE[0])<1e-6);
  assert.ok(Math.abs(Math.abs(values.getY(2)-values.getY(0))-.217/TIMBER_SCALE[1])<1e-6);
  plane.dispose();
});

test("each room shares exactly two timber textures and disposes them once, independently", () => {
  for(let visit=0;visit<5;visit++) {
    const a=createStudioInterior(),b=createStudioInterior();
    const inventory=room=>{
      const set=new Set();
      room.traverse(mesh=>{ for(const value of Object.values(mesh.material||{}))if(value?.isTexture)set.add(value); });
      return [...set];
    };
    const own=inventory(a),other=inventory(b);
    assert.equal(own.length,2); assert.equal(other.length,2);
    let retired=0,untouched=0;
    own.forEach(texture=>texture.addEventListener("dispose",()=>retired++));
    other.forEach(texture=>texture.addEventListener("dispose",()=>untouched++));
    disposeScene(a);
    assert.equal(retired,2); assert.equal(untouched,0);
    disposeScene(b); assert.equal(untouched,2);
  }
});

test("the circular table veneer closes with the same grain at coincident seam vertices", () => {
  const geometry=timberUV(new T.CylinderGeometry(.47,.47,.065,32),"table");
  const {position,normal,uv}=geometry.attributes, seen=new Map();
  let seamPairs=0;
  for(let i=0;i<position.count;i++) {
    if(Math.abs(normal.getY(i))>.9)continue;
    const key=[position.getX(i),position.getY(i),position.getZ(i),normal.getX(i),normal.getZ(i)]
      .map(v=>Math.round(v*1e5)).join(",");
    if(seen.has(key)) {
      const j=seen.get(key);
      assert.ok(Math.abs(timberField(uv.getX(i),uv.getY(i))-timberField(uv.getX(j),uv.getY(j)))<1e-4);
      seamPairs++;
    } else seen.set(key,i);
  }
  assert.equal(seamPairs,2);
  const circumference=Math.PI*2*.47,sidePeriod=circumference/2;
  assert.ok(Math.abs(sidePeriod/TIMBER_SCALE[0]-1)<.1,"authored side scale adjustment below 10%");
  geometry.dispose();
});
