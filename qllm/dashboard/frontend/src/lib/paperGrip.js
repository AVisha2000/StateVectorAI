import * as T from "three";
import mesh from "./paperGripMesh.js";

// Static Blender-authored viewmodel in the paper's frame. The room owns and
// disposes these resources; the result clone shares them only within that room.
export function createPaperGrip() {
  const grip = new T.Group();
  grip.name = "First-person paper grip";
  for (const part of mesh.parts) {
    const geometry = new T.BufferGeometry();
    geometry.setAttribute("position", new T.Float32BufferAttribute(part.positions, 3));
    geometry.setAttribute("normal", new T.Float32BufferAttribute(part.normals, 3));
    geometry.setIndex(part.indices);
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const object = new T.Mesh(geometry, new T.MeshStandardMaterial({
      color: part.color, roughness: part.roughness,
    }));
    object.name = part.name;
    object.castShadow = true;
    object.receiveShadow = true;
    grip.add(object);
  }
  return grip;
}
