import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { createPaperGeometry, createPaperSheet, createWorkingNoteTexture, paperBend, PAPER_BOTTOM, PAPER_TOP, PAPER_THICKNESS } from "./paperSheet.js";
import { disposeScene } from "../components/researchPlanetScene.js";

test("thin sheet is closed with exact groups, outward normals and readable top UVs", () => {
  const geometry = createPaperGeometry(), {position,normal,uv} = geometry.attributes, index = geometry.index;
  assert.equal(position.count,562); assert.equal(index.count,2640);
  assert.deepEqual(geometry.groups,[{start:0,count:1152,materialIndex:0},{start:1152,count:1488,materialIndex:1}]);
  const edges = new Map(), columns = new Map();
  const identity = point => point.toArray().map(v=>v.toFixed(8)).join(",");
  for(let i=0;i<position.count;i++) {
    const p=new T.Vector3().fromBufferAttribute(position,i),n=new T.Vector3().fromBufferAttribute(normal,i);
    assert.ok(p.toArray().every(Number.isFinite)); assert.ok(Math.abs(n.length()-1)<1e-6);
    assert.ok(geometry.boundingBox.containsPoint(p));
    const key=[p.x,p.z].map(v=>v.toFixed(8)).join(","), ys=columns.get(key)||[];
    ys.push(p.y); columns.set(key,ys);
  }
  for(const ys of columns.values()) assert.ok(Math.abs(Math.max(...ys)-Math.min(...ys)-PAPER_THICKNESS)<1e-8);
  for(let i=0;i<index.count;i+=3) {
    const ids=[index.getX(i),index.getX(i+1),index.getX(i+2)],points=ids.map(j=>new T.Vector3().fromBufferAttribute(position,j));
    const triangle=new T.Triangle(...points);
    assert.ok(triangle.getArea()>1e-10);
    const n=ids.reduce((sum,j)=>sum.add(new T.Vector3().fromBufferAttribute(normal,j)),new T.Vector3());
    assert.ok(triangle.getNormal(new T.Vector3()).dot(n)>0);
    for(let j=0;j<3;j++) {
      const a=identity(points[j]),b=identity(points[(j+1)%3]),key=[a,b].sort().join("|");
      const edge=edges.get(key)||{count:0,direction:0}; edge.count++; edge.direction+=a<b?1:-1; edges.set(key,edge);
    }
    if(i<1152) for(const j of ids) {
      const x=position.getX(j),z=position.getZ(j);
      assert.ok(Math.abs(position.getY(j)-PAPER_TOP-paperBend(x,z))<1e-8);
      assert.ok(Math.abs(uv.getX(j)-(x/.19+.5))<1e-7);
      assert.ok(Math.abs(uv.getY(j)-(.5-z/.25))<1e-7);
    }
  }
  assert.ok([...edges.values()].every(edge=>edge.count===2&&edge.direction===0));
  assert.ok(Math.abs(geometry.boundingBox.min.y-PAPER_BOTTOM)<1e-8);
  assert.ok(Math.abs(geometry.boundingBox.max.y-(PAPER_TOP+.004))<1e-8);
  assert.equal(paperBend(.08,.07),0,"grip margin is flat");
  // Thickness must hold between grid vertices too, not just at authored nodes.
  const material = new T.MeshBasicMaterial({side:T.DoubleSide});
  const mesh = new T.Mesh(geometry,[material,material]);
  mesh.updateMatrixWorld();
  for (const [x,z] of [[-.084,-.117],[-.031,-.079],[-.061,-.017],[.033,.071]]) {
    const hits = new T.Raycaster(new T.Vector3(x,.1,z),new T.Vector3(0,-1,0)).intersectObject(mesh);
    assert.equal(hits.length,2);
    assert.ok(Math.abs(hits[0].point.y-hits[1].point.y-PAPER_THICKNESS)<1e-8);
  }
  material.dispose();
  geometry.dispose();
});

test("the printed note uses the real document and its texture is disposed with the sheet", () => {
  const previous=globalThis.document, printed=[];
  const context={font:"",fillStyle:"",fillRect(){},measureText(text){return {width:text.length*12};},fillText(text){printed.push(text);}};
  globalThis.document={createElement:()=>({width:0,height:0,getContext:()=>context})};
  try {
    const texture=createWorkingNoteTexture({title:"A bounded comparison",description:"Preserve the negative result."});
    assert.ok(printed.includes("A bounded comparison"));
    assert.ok(printed.includes("Preserve the negative result."));
    assert.ok(printed.join(" ").includes("not a published paper"));
    assert.equal(texture.colorSpace,T.SRGBColorSpace);
    assert.equal(texture.image.width,768); assert.equal(texture.image.height,1008);
    const sheet=createPaperSheet(texture);
    let disposed=0; texture.addEventListener("dispose",()=>disposed++);
    disposeScene(sheet); assert.equal(disposed,1);
  } finally { if(previous===undefined) delete globalThis.document; else globalThis.document=previous; }
});
