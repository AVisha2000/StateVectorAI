import * as T from "three";
import { launchHeight } from "../lib/researchStudios.js";

const Y = new T.Vector3(0, 1, 0);
export const PLANET_RADIUS = 3.2;
function material(color, extra = {}) {
  return new T.MeshStandardMaterial({
    color,
    roughness: 0.8,
    metalness: 0,
    ...extra,
  });
}
function mesh(parent, geometry, color, position = [0, 0, 0], extra = {}) {
  const item = new T.Mesh(
    geometry,
    typeof color === "string" ? material(color, extra) : color,
  );
  item.position.set(...position);
  item.castShadow = true;
  item.receiveShadow = true;
  parent.add(item);
  return item;
}
function box(p, size, color, at) {
  return mesh(p, new T.BoxGeometry(...size), color, at);
}
function ball(p, radius, color, at, extra) {
  return mesh(p, new T.SphereGeometry(radius, 12, 8), color, at, extra);
}
function cylinder(p, rt, rb, height, color, at, segments = 12) {
  return mesh(p, new T.CylinderGeometry(rt, rb, height, segments), color, at);
}
function bar(parent, a, b, radius, color) {
  const start = new T.Vector3(...a),
    end = new T.Vector3(...b),
    direction = end.clone().sub(start);
  const item = cylinder(
    parent,
    radius,
    radius,
    direction.length(),
    color,
    start.clone().add(end).multiplyScalar(0.5).toArray(),
    6,
  );
  item.quaternion.setFromUnitVectors(Y, direction.normalize());
  return item;
}
function textBoard(
  parent,
  lines,
  { width = 0.65, height = 0.48, at = [0, 0.55, -0.22] } = {},
) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 384;
  const ctx = canvas.getContext("2d");
  const texture = new T.CanvasTexture(canvas);
  texture.colorSpace = T.SRGBColorSpace;
  const write = (nextLines) => {
    ctx.fillStyle = "#183d37";
    ctx.fillRect(0, 0, 512, 384);
    ctx.strokeStyle = "#9ca99a";
    ctx.lineWidth = 2;
    ctx.strokeRect(22, 22, 468, 340);
    ctx.fillStyle = "#e9efdb";
    ctx.textAlign = "center";
    nextLines.forEach((line, i) => {
      ctx.font = `${i === 0 ? 28 : 44}px Georgia`;
      ctx.fillText(line, 256, 88 + i * 80);
    });
    texture.needsUpdate = true;
  };
  write(lines);
  box(parent, [width + 0.06, height + 0.06, 0.045], "#bd9b76", at);
  const board = mesh(
    parent,
    new T.PlaneGeometry(width, height),
    new T.MeshBasicMaterial({ map: texture }),
    [at[0], at[1], at[2] + 0.026],
  );
  board.userData.write = write;
  return board;
}
export function createResearcher(color = "#d79c85") {
  const person = new T.Group();
  cylinder(person, 0.065, 0.085, 0.17, color, [0, 0.22, 0]);
  ball(person, 0.063, "#ebc7a6", [0, 0.36, 0]);
  for (const x of [-0.022, 0.022])
    ball(person, 0.007, "#354244", [x, 0.361, 0.058]);
  const hair = ball(person, 0.067, "#31373d", [0, 0.39, -0.012]);
  hair.scale.y = 0.62;
  for (const x of [-0.04, 0.04])
    box(person, [0.053, 0.12, 0.06], "#364752", [x, 0.075, 0]);
  const arm = new T.Group();
  arm.position.set(0.08, 0.29, 0);
  person.add(arm);
  cylinder(arm, 0.024, 0.025, 0.13, color, [0, -0.05, 0]);
  ball(arm, 0.027, "#ebc7a6", [0, -0.13, 0]);
  const left = cylinder(person, 0.024, 0.025, 0.16, color, [-0.085, 0.24, 0]);
  left.rotation.z = -0.25;
  person.userData.arm = arm;
  return person;
}
function tree(parent, x, z, scale = 1, color = "#477d61") {
  const tree = new T.Group();
  tree.position.set(x, 0, z);
  tree.scale.setScalar(scale);
  parent.add(tree);
  cylinder(tree, 0.022, 0.03, 0.24, "#8a7254", [0, 0.1, 0], 6);
  const crown = mesh(
    tree,
    new T.IcosahedronGeometry(0.15, 0),
    color,
    [0, 0.3, 0],
  );
  crown.scale.y = 1.3;
  return tree;
}
function table(parent, x = 0, z = 0, w = 0.5) {
  box(parent, [w, 0.055, 0.24], "#d7b88f", [x, 0.27, z]);
  for (const dx of [-w * 0.38, w * 0.38])
    for (const dz of [-0.07, 0.07])
      box(parent, [0.025, 0.25, 0.025], "#829798", [x + dx, 0.12, z + dz]);
}
function robot(parent, x, z) {
  const robot = new T.Group();
  robot.position.set(x, 0, z);
  parent.add(robot);
  box(robot, [0.18, 0.22, 0.12], "#e0e8dd", [0, 0.27, 0]);
  const head = box(robot, [0.21, 0.15, 0.14], "#ececdd", [0, 0.48, 0]);
  box(robot, [0.15, 0.065, 0.012], "#243944", [0, 0.49, 0.077]);
  for (const x of [-0.042, 0.042])
    ball(robot, 0.016, "#8fddd2", [x, 0.49, 0.09], {
      emissive: "#6bbcb3",
      emissiveIntensity: 0.5,
    });
  for (const x of [-0.06, 0.06]) {
    box(robot, [0.056, 0.13, 0.065], "#8aa3ad", [x, 0.095, 0]);
    box(robot, [0.07, 0.04, 0.095], "#d9e2dc", [x, 0.022, 0.015]);
  }
  const arm = box(robot, [0.045, 0.17, 0.05], "#8aa3ad", [0.14, 0.28, 0]);
  arm.rotation.z = 0.5;
  box(robot, [0.045, 0.17, 0.05], "#8aa3ad", [-0.14, 0.28, 0]);
  return { robot, head, arm };
}
export function createStudio(studio, { meeting = false } = {}) {
  const group = new T.Group(),
    animated = [];
  cylinder(group, 0.57, 0.62, 0.08, "#c8ccb1", [0, -0.03, 0], 32);
  const researcher = createResearcher(studio.color);
  researcher.position.set(-0.27, 0, 0.27);
  researcher.rotation.y = -0.35;
  group.add(researcher);
  animated.push((t) => {
    researcher.userData.arm.rotation.x = -0.6 + Math.sin(t * 2.1) * 0.22;
  });
  if (studio.kind === "quantum-game") {
    researcher.visible = false;
    cylinder(group, 0.3, 0.35, 0.12, "#718e8f", [0.08, 0.12, -0.02], 24);
    const globe = new T.Group();
    globe.position.set(0.08, 0.56, -0.02);
    group.add(globe);
    for (let i = 0; i < 3; i++) {
      const ring = mesh(globe, new T.TorusGeometry(0.3, 0.011, 6, 48), "#b9a5e4");
      if (i === 1) ring.rotation.x = Math.PI / 2;
      if (i === 2) ring.rotation.y = Math.PI / 2;
    }
    bar(globe, [0, 0, 0], [0.16, 0.22, 0.12], 0.014, "#b0efd3");
    ball(globe, 0.044, "#b0efd3", [0.16, 0.22, 0.12], { emissive: "#73af9b", emissiveIntensity: 0.3 });
    const gate = mesh(group, new T.TorusGeometry(0.15, 0.022, 8, 32, Math.PI), "#e4c297", [-0.29, 0.18, 0.25]);
    gate.rotation.z = 0;
    for (const x of [-0.44, -0.14]) cylinder(group, 0.022, 0.022, 0.16, "#e4c297", [x, 0.1, 0.25]);
  } else if (studio.id === "mathematics") {
    const board = textBoard(group, ["SUM OF SQUARES", "n = 2", "∑ k² = 5"]);
    let written = 2;
    animated.push((time) => {
      const n = 2 + (Math.floor(time / 3) % 5);
      if (n !== written) {
        written = n;
        board.userData.write([
          "SUM OF SQUARES",
          `n = ${n}`,
          `∑ k² = ${(n * (n + 1) * (2 * n + 1)) / 6}`,
        ]);
      }
    });
    for (const x of [-0.27, 0.27])
      bar(group, [x, 0, -0.24], [x, 0.79, -0.24], 0.018, "#bda17c");
    table(group, 0.25, 0.18, 0.26);
    for (let i = 0; i < 3; i++)
      box(group, [0.14, 0.025, 0.1], ["#b88872", "#e2d4b5", "#a9bfc4"][i], [
        0.25,
        0.32 + i * 0.025,
        0.18,
      ]);
  } else if (studio.id === "physics") {
    cylinder(group, 0.28, 0.3, 0.06, "#859b98", [0.13, 0.035, -0.06], 24);
    const ring = mesh(
      group,
      new T.TorusGeometry(0.22, 0.009, 6, 32),
      "#d7cfab",
      [0.13, 0.072, -0.06],
    );
    ring.rotation.x = Math.PI / 2;
    const rocket = new T.Group();
    group.userData.rocket = rocket;
    rocket.position.set(0.13, 0.08, -0.06);
    group.add(rocket);
    cylinder(rocket, 0.07, 0.07, 0.34, "#eee8d8", [0, 0.22, 0]);
    cylinder(rocket, 0, 0.07, 0.16, "#d8836e", [0, 0.47, 0]);
    for (let i = 0; i < 3; i++) {
      const fin = box(rocket, [0.035, 0.17, 0.16], "#d8836e", [
        Math.sin(i * 2.094) * 0.055,
        0.1,
        Math.cos(i * 2.094) * 0.055,
      ]);
      fin.rotation.y = i * 2.094;
    }
    ball(rocket, 0.031, "#609ca7", [0, 0.3, 0.065]);
    const flame = cylinder(rocket, 0.038, 0, 0.23, "#f5c77b", [0, -0.065, 0]);
    flame.visible = false;
    animated.push((t) => {
      const h = launchHeight(t);
      rocket.position.y = 0.08 + h;
      rocket.visible = t % 14 < 10;
      flame.visible = h > 0;
      flame.scale.y = 0.8 + Math.sin(t * 33) * 0.22;
    });
    box(group, [0.12, 0.6, 0.1], "#7e9998", [-0.19, 0.31, -0.26]);
    bar(group, [-0.2, 0.57, -0.26], [0.12, 0.57, -0.26], 0.023, "#d3b486");
  } else if (studio.id === "chemistry") {
    table(group, 0.05, -0.1, 0.73);
    group.userData.vials = [];
    for (let i = 0; i < 3; i++) {
      const x = -0.18 + i * 0.23,
        color = ["#b699e0", "#e6ba78", "#a2d6ba"][i];
      group.userData.vials.push(
        ball(group, 0.078, color, [x, 0.38, -0.1], {
          roughness: 0.22,
          transparent: true,
          opacity: 0.82,
        }),
      );
      cylinder(group, 0.027, 0.026, 0.11, "#e2e7e1", [x, 0.47, -0.1]);
      const bubble = ball(group, 0.022, color, [x, 0.58, -0.1]);
      animated.push((t) => {
        const f = (t * 0.45 + i * 0.33) % 1;
        bubble.position.y = 0.53 + f * 0.35;
        bubble.scale.setScalar(1 - f * 0.8);
      });
    }
    group.userData.kineticBoard = textBoard(
      group,
      ["REACTION TOY", "A → B", "k = ?"],
      {
        width: 0.33,
        height: 0.26,
        at: [0.22, 0.63, -0.3],
      },
    );
  } else if (studio.id === "biology") {
    group.userData.growthPlants = [];
    const dna = new T.Group();
    dna.position.set(0.1, 0.1, -0.11);
    group.add(dna);
    for (let i = 0; i < 10; i++) {
      const a = i * 0.63,
        x = Math.cos(a) * 0.115,
        z = Math.sin(a) * 0.115,
        y = i * 0.066;
      ball(dna, 0.036, "#e2b17f", [x, y, z]);
      ball(dna, 0.036, "#a5d5bf", [-x, y, -z]);
      bar(dna, [x, y, z], [-x, y, -z], 0.012, "#c4d7c1");
    }
    animated.push((t) => {
      dna.rotation.y = t * 0.18;
    });
    for (const [x, z] of [
      [-0.3, -0.18],
      [0.36, 0.2],
      [0.31, -0.31],
    ]) {
      cylinder(group, 0.085, 0.055, 0.1, "#c09473", [x, 0.05, z]);
      group.userData.growthPlants.push(tree(group, x, z, 0.6, "#87ac71"));
    }
  } else if (studio.id === "ai-safety") {
    const bot = robot(group, 0.12, -0.12);
    for (const x of [-0.13, 0.38])
      box(group, [0.025, 0.68, 0.025], "#86b9c2", [x, 0.34, -0.22]);
    box(group, [0.535, 0.025, 0.025], "#86b9c2", [0.125, 0.68, -0.22]);
    const scan = box(
      group,
      [0.47, 0.01, 0.27],
      new T.MeshBasicMaterial({
        color: "#a5e1d9",
        transparent: true,
        opacity: 0.34,
        side: T.DoubleSide,
      }),
      [0.12, 0.4, -0.08],
    );
    animated.push((t) => {
      bot.head.rotation.y = Math.sin(t * 0.7) * 0.35;
      bot.arm.rotation.z = 0.4 + Math.sin(t * 1.4) * 0.3;
      scan.position.y = 0.1 + (Math.sin(t * 0.9) + 1) * 0.25;
    });
    table(group, -0.28, 0.1, 0.2);
    textBoard(group, ["TEST 01", "BOUNDARY"], {
      width: 0.18,
      height: 0.13,
      at: [-0.28, 0.4, 0.04],
    });
  } else {
    for (let i = 0; i < 3; i++) {
      box(group, [0.16, 0.46 + i * 0.07, 0.17], "#576a78", [
        -0.22 + i * 0.23,
        0.25 + i * 0.035,
        -0.2,
      ]);
      for (let j = 0; j < 4; j++)
        box(group, [0.09, 0.016, 0.005], j % 2 ? "#add4b8" : "#e3c292", [
          -0.22 + i * 0.23,
          0.16 + j * 0.075,
          -0.11,
        ]);
    }
    const network = new T.Group();
    network.position.set(0.1, 0.75, -0.17);
    group.add(network);
    for (let i = 0; i < 6; i++) {
      const a = (i * Math.PI) / 3;
      ball(network, 0.028, "#aebbe9", [
        Math.cos(a) * 0.22,
        Math.sin(a) * 0.22,
        0,
      ]);
      bar(
        network,
        [0, 0, 0],
        [Math.cos(a) * 0.22, Math.sin(a) * 0.22, 0],
        0.006,
        "#97bdcf",
      );
    }
    ball(network, 0.055, "#e3bc8d", [0, 0, 0]);
    animated.push((t) => {
      network.rotation.y = Math.sin(t * 0.4) * 0.5;
    });
  }
  if (meeting) {
    const coffee = new T.Group();
    coffee.position.set(0, 0, 0.75);
    group.add(coffee);
    cylinder(coffee, 0.28, 0.28, 0.055, "#bfa180", [0, 0.23, 0], 32);
    cylinder(coffee, 0.025, 0.1, 0.23, "#7e8c87", [0, 0.1, 0]);
    for (const x of [-0.14, 0.14]) {
      cylinder(coffee, 0.035, 0.03, 0.058, "#eee1cb", [x, 0.285, 0]);
      cylinder(coffee, 0.028, 0.028, 0.004, "#57423b", [x, 0.316, 0]);
    }
    box(coffee, [0.14, 0.007, 0.18], "#eee7d6", [0.015, 0.262, 0.02]);
    const visitor = createResearcher("#a4bcbc");
    visitor.position.set(0.3, 0, 0.95);
    visitor.rotation.y = Math.PI;
    group.add(visitor);
    group.userData.visitor = visitor;
    for (const x of [-0.39, 0.39]) {
      cylinder(group, 0.13, 0.13, 0.04, "#b9a285", [x, 0.13, 0.91]);
      cylinder(group, 0.035, 0.065, 0.12, "#83968c", [x, 0.06, 0.91]);
    }
  }
  group.userData.animate = (time) => animated.forEach((fn) => fn(time));
  group.userData.researcher = researcher;
  return group;
}
function islandPatch(parent, radius, height, color, seed) {
  const vertices = [],
    indices = [],
    rings = 8,
    sides = 48;
  for (let r = 0; r <= rings; r++)
    for (let i = 0; i <= sides; i++) {
      const a = (i / sides) * Math.PI * 2,
        shape =
          1 + 0.1 * Math.sin(a * 3 + seed) + 0.065 * Math.cos(a * 5 - seed),
        d = (r / rings) * radius * shape,
        x = Math.cos(a) * d,
        z = Math.sin(a) * d * 0.83;
      vertices.push(
        x,
        Math.sqrt(Math.max(0, PLANET_RADIUS ** 2 - x * x - z * z)) -
          PLANET_RADIUS +
          height,
        z,
      );
      if (r < rings && i < sides) {
        const v = r * (sides + 1) + i;
        indices.push(
          v,
          v + 1,
          v + sides + 1,
          v + 1,
          v + sides + 2,
          v + sides + 1,
        );
      }
    }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute("position", new T.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return mesh(parent, geometry, color);
}
export function buildPlanet(studios) {
  const group = new T.Group(),
    anchors = new Map(),
    labs = [];
  mesh(
    group,
    new T.IcosahedronGeometry(PLANET_RADIUS, 12),
    material("#356c6d", {
      roughness: 0.69,
      metalness: 0.12,
      flatShading: true,
    }),
  );
  mesh(
    group,
    new T.SphereGeometry(PLANET_RADIUS * 1.018, 64, 48),
    new T.MeshBasicMaterial({
      color: "#88bcc0",
      transparent: true,
      opacity: 0.065,
      side: T.BackSide,
      depthWrite: false,
    }),
  );
  studios.forEach((studio, index) => {
    const normal = new T.Vector3(...studio.position).normalize(),
      anchor = new T.Group();
    anchor.position.copy(normal.clone().multiplyScalar(PLANET_RADIUS));
    anchor.quaternion.setFromUnitVectors(Y, normal);
    group.add(anchor);
    islandPatch(anchor, 1.01, 0.025, "#789b8c", index);
    islandPatch(anchor, 0.96, 0.06, "#c9bf96", index);
    islandPatch(
      anchor,
      0.88,
      0.092,
      ["#a9b591", "#b9b58e", "#a8b89d", "#94b28e", "#a6b6a0", "#a9b7a1"][
        index % 6
      ],
      index,
    );
    const lab = createStudio(studio);
    lab.position.y = 0.1;
    lab.scale.setScalar(1.08);
    anchor.add(lab);
    labs.push(lab);
    for (let i = 0; i < 5; i++) {
      const a = 1.15 + i * 0.67;
      tree(
        anchor,
        Math.cos(a) * 0.64,
        Math.sin(a) * 0.55,
        0.65 + (i % 3) * 0.15,
        ["#4c7960", "#709473", "#779c7b"][i % 3],
      );
    }
    for (let i = 0; i < 4; i++) {
      const a = i * 1.7,
        rock = mesh(anchor, new T.DodecahedronGeometry(0.075, 0), "#a6b2a5", [
          Math.cos(a) * 0.7,
          -0.01,
          Math.sin(a) * 0.52,
        ]);
      rock.scale.y = 0.6;
    }
    const pin = new T.Object3D();
    pin.position.set(0, 0.92, 0);
    anchor.add(pin);
    anchor.traverse((child) => {
      child.userData.studioId = studio.id;
    });
    anchors.set(studio.id, { anchor, pin, normal, lab });
  });
  // Scenic terrain makes the reverse hemisphere worth exploring as well.
  [
    [-0.5, 0.5, -0.75],
    [0.65, -0.1, -0.8],
    [-0.2, -0.75, -0.6],
    [0.55, 0.76, -0.25],
  ].forEach((position, index) => {
    const normal = new T.Vector3(...position).normalize(),
      land = new T.Group();
    land.position.copy(normal.clone().multiplyScalar(PLANET_RADIUS));
    land.quaternion.setFromUnitVectors(Y, normal);
    group.add(land);
    islandPatch(land, 0.95, 0.03, "#b6b591", index + 7);
    islandPatch(land, 0.87, 0.07, "#839f80", index + 7);
    for (let i = 0; i < 3; i++) {
      const mountain = mesh(
        land,
        new T.ConeGeometry(0.26, 0.48 + i * 0.1, 5),
        "#809487",
        [-0.23 + i * 0.2, 0.2, -0.12 + (i % 2) * 0.15],
      );
      mountain.rotation.y = i;
    }
    for (let i = 0; i < 7; i++) {
      const a = i * 0.8;
      tree(land, Math.cos(a) * 0.59, Math.sin(a) * 0.5, 0.7, "#587d62");
    }
  });
  // Small marks are ocean illustration, never research connections or data flow.
  for (let i = 0; i < 30; i++) {
    const a = i * 2.39996,
      y = 1 - ((i + 0.5) / 30) * 2,
      radial = Math.sqrt(1 - y * y),
      normal = new T.Vector3(Math.cos(a) * radial, y, Math.sin(a) * radial);
    const wave = new T.Group();
    wave.position.copy(normal.clone().multiplyScalar(PLANET_RADIUS + 0.012));
    wave.quaternion.setFromUnitVectors(Y, normal);
    group.add(wave);
    const geometry = new T.BufferGeometry().setFromPoints([
      new T.Vector3(-0.06, 0, 0),
      new T.Vector3(0, 0, 0.018),
      new T.Vector3(0.06, 0, 0),
    ]);
    wave.add(
      new T.Line(
        geometry,
        new T.LineBasicMaterial({
          color: "#a1c1b6",
          transparent: true,
          opacity: 0.23,
        }),
      ),
    );
  }
  const clouds = new T.Group();
  group.add(clouds);
  [
    [-0.88, 0.52, 0.2],
    [0.75, 0.55, -0.35],
    [0.1, 0.92, 0.36],
    [-0.38, -0.74, 0.57],
    [0.92, -0.28, -0.12],
    [0.1, -0.3, -0.95],
    [-0.65, 0.12, -0.72],
  ].forEach((position) => {
    const normal = new T.Vector3(...position).normalize(),
      cloud = new T.Group();
    cloud.position.copy(normal.multiplyScalar(3.65));
    cloud.quaternion.setFromUnitVectors(Y, normal);
    clouds.add(cloud);
    for (let i = 0; i < 4; i++) {
      const puff = ball(cloud, 0.12 + Math.sin(i * 2) * 0.025, "#e1e8dc", [
        (i - 1.5) * 0.14,
        0,
        0,
      ]);
      puff.scale.set(1.15, 0.5, 0.8);
      puff.castShadow = false;
    }
  });
  return {
    group,
    anchors,
    animate(time) {
      labs.forEach((lab) => lab.userData.animate(time));
      clouds.rotation.y = time * 0.007;
    },
  };
}
export function disposeScene(scene) {
  const geometries = new Set(),
    materials = new Set(),
    textures = new Set();
  scene.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    for (const m of Array.isArray(object.material)
      ? object.material
      : object.material
        ? [object.material]
        : [])
      materials.add(m);
  });
  materials.forEach((m) => {
    for (const value of Object.values(m))
      if (value?.isTexture) textures.add(value);
    m.dispose();
  });
  textures.forEach((t) => t.dispose());
  geometries.forEach((g) => g.dispose());
}
