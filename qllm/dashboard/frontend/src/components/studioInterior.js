import * as T from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { createTimberTextures, timberMaterial, timberUV } from "../lib/studioTimber.js";

// Fixed authored pavilion, not a collision or terrain system. Y-up project units.
// Local facade +X runs along a wall, +Y is up; rotate around the world origin.
export function studioInteriorPlan() {
  const plan = [];
  const place = (id, slot, size, at, turn = 0, shape = "box") =>
    plan.push({ id, slot, size, at, turn, shape });
  place("subfloor", "floor", [5.4, .06, 5.4], [0, -.045, 0]);
  place("ceiling", "plaster", [5.4, .05, 5.4], [0, 1.96, 0]);
  // Short, staggered boards; gaps expose the dark subfloor.
  for (let row = 0; row < 24; row++) {
    const z = -2.5875 + row * .225;
    const cuts = row % 2 ? [-2.7, -2.025, -.675, .675, 2.025, 2.7] : [-2.7, -1.35, 0, 1.35, 2.7];
    for (let board = 1; board < cuts.length; board++)
      place(`board-${row}-${board}`, (row + board) % 3 ? "oak" : "oak-light",
        [cuts[board] - cuts[board - 1] - .008, .014, .217],
        [(cuts[board] + cuts[board - 1]) / 2, -.009, z]);
  }
  for (let side = 0; side < 4; side++) {
    const turn = side * Math.PI / 2;
    const wall = (id, slot, size, at) => place(`wall-${side}-${id}`, slot, size, at, turn);
    wall("base", "plaster", [5.35, .43, .1], [0, .2, -2.65]);
    wall("skirting", "metal", [5.35, .055, .12], [0, .02, -2.59]);
    wall("sill", "timber", [5.35, .06, .18], [0, .445, -2.58]);
    // No opaque window infill: the exterior is real, static geometry with parallax.
    wall("lintel", "timber", [5.35, .15, .13], [0, 1.71, -2.62]);
    wall("fascia", "plaster", [5.35, .15, .1], [0, 1.86, -2.65]);
    wall("light", "light", [4.9, .025, .035], [0, 1.615, -2.565]);
    for (let bay = -2; bay <= 2; bay++) {
      wall(`mullion-${bay}`, "frames", [.035, 1.22, .075], [bay * 1.065, 1.055, -2.59]);
      wall(`panel-${bay}`, "timber", [.032, .29, .018], [bay * 1.065, .2, -2.589]);
    }
    wall("transom", "frames", [5.3, .025, .055], [0, 1.21, -2.61]);
    wall("patio", "stone", [6.1, .04, .85], [0, -.04, -3.125]);
    wall("path", "stone", [1, .025, 6], [0, -.0525, -6.55]);
    for (const x of [-2, 2]) {
      wall(`garden-wall-${x}`, "stone", [.18, .22, 2.4], [x, .05, -4.75]);
      wall(`garden-cap-${x}`, "timber", [.22, .04, 2.45], [x, .18, -4.75]);
    }
  }
  for (let x = -2; x <= 2; x++) {
    place(`beam-${x}`, "timber", [.085, .1, 5.3], [x * 1.04, 1.87, 0]);
    if (x % 2 === 0) place(`ceiling-light-${x}`, "light", [.04, .01, 2.9], [x * 1.04, 1.813, 0]);
  }
  place("garden-ground", "lawn", [46, .04, 46], [0, -.08, 0]);
  // Authored rounded landforms, not a generated ecological/planetary model.
  for (let i = 0; i < 12; i++) {
    const a = i * Math.PI / 6, r = 13 + (i % 3);
    place(`hill-${i}`, i % 2 ? "hill" : "hill-light", [9 + (i % 3), 4 + (i % 4), 8],
      [Math.cos(a) * r, -.5, Math.sin(a) * r], a, "ellipsoid");
  }
  place("sky-enclosure", "sky", [48, 48, 48], [0, 0, 0], 0, "ellipsoid");
  return plan;
}

export function createStudioInterior(timber = createTimberTextures()) {
  const group = new T.Group(), slots = new Map();
  const surface = (color, roughness = .8, metalness = 0) => new T.MeshStandardMaterial({ color, roughness, metalness });
  const palette = {
    floor: surface("#544d40"),
    oak: timberMaterial("#b58b60", timber),
    "oak-light": timberMaterial("#c39e71", timber),
    plaster: surface("#b8c3b3", .9),
    timber: timberMaterial("#927451", timber),
    metal: surface("#29443f", .4, .3),
    frames: surface("#29443f", .4, .3),
    light: new T.MeshBasicMaterial({ color: "#ffdfac" }),
    stone: surface("#b1b39b", .94),
    lawn: surface("#739270", 1),
    hill: surface("#718f7d", 1),
    "hill-light": surface("#93a48a", 1),
    sky: new T.MeshBasicMaterial({ color: "#b8d3d5", side: T.BackSide }),
  };
  for (const { id, slot, size, at, turn, shape } of studioInteriorPlan()) {
    const geometry = shape === "ellipsoid"
      ? new T.SphereGeometry(.5, 24, 12).scale(...size)
      : new T.BoxGeometry(...size);
    if (["oak","oak-light","timber"].includes(slot)) {
      const axis = ["x","y","z"][size.indexOf(Math.max(...size))];
      timberUV(geometry,axis,id);
    }
    // Landform orientation is local; facade translations rotate around the room.
    if (shape === "ellipsoid") geometry.rotateY(turn).translate(...at);
    else geometry.translate(...at).rotateY(turn);
    if (!slots.has(slot)) slots.set(slot, []);
    slots.get(slot).push(geometry);
  }
  for (const [slot, parts] of slots) {
    const geometry = mergeGeometries(parts, false);
    geometry.computeBoundingBox(); geometry.computeBoundingSphere();
    const mesh = new T.Mesh(geometry, palette[slot]);
    mesh.name = `Studio ${slot}`;
    mesh.receiveShadow = !["light", "sky", "hill", "hill-light"].includes(slot);
    // Authored interior key: frame silhouettes cast; roof/walls do not eclipse it.
    mesh.castShadow = slot === "frames";
    mesh.raycast = () => {};
    group.add(mesh);
    parts.forEach((part) => part.dispose());
  }
  group.visible = false;
  return group;
}
