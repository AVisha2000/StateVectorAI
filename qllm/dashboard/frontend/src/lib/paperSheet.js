import * as T from "three";
import { NOTE_CANVAS, workingNote, workingNoteLayout } from "./workingNote.js";

// Y-up paper-local coordinates. Keep the previous bottom support plane, not
// the obsolete thick-board top. The grip margin stays flat through every pose.
export const PAPER_WIDTH = .19, PAPER_DEPTH = .25;
export const PAPER_BOTTOM = -.006, PAPER_THICKNESS = .0014;
export const PAPER_TOP = PAPER_BOTTOM + PAPER_THICKNESS;
export function paperBend(x, z) {
  const u = Math.max(0, -x / (PAPER_WIDTH / 2));
  const v = Math.max(0, -z / (PAPER_DEPTH / 2));
  return .004 * u * u * v * v;
}

export function createPaperGeometry() {
  // BoxGeometry owns exact capacity, seams and six closed sides. Bend once,
  // then combine groups into the printed face and the unprinted back/edges.
  const geometry = new T.BoxGeometry(PAPER_WIDTH, PAPER_THICKNESS, PAPER_DEPTH, 12, 1, 16);
  const position = geometry.attributes.position;
  for (let i = 0; i < position.count; i++) {
    position.setY(i, position.getY(i) + PAPER_BOTTOM + PAPER_THICKNESS / 2
      + paperBend(position.getX(i), position.getZ(i)));
  }
  const top = geometry.groups[2], bottom = geometry.groups[3];
  const original = geometry.index.array.slice();
  // Match front/back diagonals after bending: otherwise opposite BoxGeometry
  // diagonals produce slightly different interior heights in a curved cell.
  const key = i => `${position.getX(i)},${position.getZ(i)}`;
  const back = new Map();
  for (let i = bottom.start; i < bottom.start + bottom.count; i++)
    back.set(key(original[i]), original[i]);
  for (let i = 0; i < top.count; i += 3)
    for (const [j, offset] of [0,2,1].entries())
      original[bottom.start + i + j] = back.get(key(original[top.start + i + offset]));
  const indices = new Uint16Array(original.length);
  indices.set(original.subarray(top.start, top.start + top.count));
  indices.set(original.subarray(0, top.start), top.count);
  indices.set(original.subarray(top.start + top.count), top.count + top.start);
  geometry.setIndex(new T.BufferAttribute(indices, 1));
  geometry.clearGroups();
  geometry.addGroup(0, top.count, 0);
  geometry.addGroup(top.count, indices.length - top.count, 1);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

export function createPaperSheet(texture) {
  const sheet = new T.Mesh(createPaperGeometry(), [
    new T.MeshStandardMaterial({ map: texture, roughness: .94 }),
    new T.MeshStandardMaterial({ color: "#eee6d2", roughness: .96 }),
  ]);
  sheet.name = "Thin working sheet";
  sheet.castShadow = true;
  sheet.receiveShadow = true;
  return sheet;
}

export function createWorkingNoteTexture(studio) {
  const canvas = document.createElement("canvas");
  canvas.width = NOTE_CANVAS.width; canvas.height = NOTE_CANVAS.height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#f6f0de";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#adb7a2";
  ctx.fillRect(64, 76, 640, 2);
  ctx.fillRect(64, 876, 640, 2);
  const rows = workingNoteLayout(workingNote(studio), (text, font) => {
    ctx.font = font;
    return ctx.measureText(text).width;
  });
  for (const row of rows) {
    ctx.font = row.font; ctx.fillStyle = row.color;
    ctx.fillText(row.text, row.x, row.y);
  }
  const texture = new T.CanvasTexture(canvas);
  texture.colorSpace = T.SRGBColorSpace;
  return texture;
}
