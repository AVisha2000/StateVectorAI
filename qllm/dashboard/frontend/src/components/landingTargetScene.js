import * as T from "three";
import { LANDING_TARGETS } from "../lib/landingChallenge.js";

// Lab-local coordinates, identical to the rocket's 0.04 studio units / metre.
// The coloured strip's X extent is the allowed range, not the decorative flags.
export function createLandingTargetScene() {
  const group = new T.Group();
  const material = new T.MeshStandardMaterial({ color: "#f0c07e", roughness: 0.8 });
  const strip = new T.Mesh(new T.BoxGeometry(1, 0.012, 0.28), material);
  group.add(strip);
  const posts = [-1, 1].map((side) => {
    const post = new T.Group();
    const pole = new T.Mesh(new T.CylinderGeometry(0.008, 0.008, 0.25, 8), new T.MeshStandardMaterial({ color: "#f0e4c5" }));
    pole.position.set(0, 0.125, 0.14);
    const flag = new T.Mesh(new T.BoxGeometry(0.035, 0.045, 0.004), material);
    flag.position.set(side * 0.02, 0.225, 0.14);
    post.add(pole, flag);
    group.add(post);
    return post;
  });
  const canvas = document.createElement("canvas");
  canvas.width = 256; canvas.height = 128;
  const context = canvas.getContext("2d");
  const texture = new T.CanvasTexture(canvas);
  texture.colorSpace = T.SRGBColorSpace;
  const label = new T.Sprite(new T.SpriteMaterial({ map: texture, transparent: true }));
  label.position.set(0, 0.38, 0.14);
  label.scale.set(0.42, 0.21, 1);
  group.add(label);
  group.traverse((object) => { object.raycast = () => {}; });
  let identity = "";
  const present = (target, hit = false) => {
    group.visible = Boolean(target);
    if (!target) return;
    group.position.set(0.13 - target.range * 0.04, 0.066, -0.06);
    strip.scale.x = target.tolerance * 2 * 0.04;
    posts[0].position.x = -target.tolerance * 0.04;
    posts[1].position.x = target.tolerance * 0.04;
    const next = `${target.id}:${hit}`;
    if (next === identity) return;
    identity = next;
    material.color.set(hit ? "#96d0a3" : "#f0c07e");
    context.clearRect(0, 0, 256, 128);
    context.fillStyle = hit ? "#234a36" : "#283c32";
    context.fillRect(8, 8, 240, 112);
    context.strokeStyle = hit ? "#a3d0a7" : "#e7bf83";
    context.lineWidth = 3; context.strokeRect(8, 8, 240, 112);
    context.textAlign = "center"; context.fillStyle = "#f5e8cb";
    context.font = "bold 40px Georgia"; context.fillText(`${target.range} m`, 128, 62);
    context.font = "20px sans-serif"; context.fillText(hit ? "TARGET FOUND" : "LAND HERE", 128, 98);
    texture.needsUpdate = true;
  };
  // Conservative lab-local support includes every allowed target and label.
  const bounds = new T.Box3();
  for (const target of LANDING_TARGETS) {
    present(target); group.updateMatrixWorld(true);
    bounds.union(new T.Box3().setFromObject(group));
  }
  present(null);
  return { group, bounds, present };
}
