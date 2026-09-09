import * as T from "three";
import { conversationSubject } from "../lib/studioCamera.js";
import { createResearcher, createStudio } from "./researchPlanetScene.js";
import { createNoteMotion } from "../lib/carriedNote.js";
import { createCoffeeMotion } from "../lib/coffeeMotion.js";
import { projectile } from "../lib/projectileToy.js";
import { resultCardView } from "../lib/resultCards.js";
import { launcherFlightBounds } from "../lib/launcherFraming.js";
import { createLandingTargetScene } from "./landingTargetScene.js";
import { presentedLandingTarget } from "../lib/landingChallenge.js";
import { cloneResearchRobot } from "../lib/researchRobot.js";
import { createStudioInterior } from "./studioInterior.js";
import { createRobotPresence } from "../lib/robotPresence.js";
import { createOfferingArm, offeredNotePose } from "../lib/paperOffer.js";
import { createPaperGrip } from "../lib/paperGrip.js";
import { createPaperSheet, createWorkingNoteTexture } from "../lib/paperSheet.js";
import { STUDIO_LAYOUT } from "../lib/studioLayout.js";
import { createTimberTextures, timberMaterial, timberUV } from "../lib/studioTimber.js";

export function createMeetingScene(studio, robotTemplate = null) {
  const group = new T.Group(),
    items = new Map();
  const mat = (color) => new T.MeshStandardMaterial({ color, roughness: 0.85 });
  const add = (parent, geometry, color, position) => {
    const mesh = new T.Mesh(
      geometry,
      typeof color === "string" ? mat(color) : color,
    );
    mesh.position.set(...position);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  };
  const box = (parent, size, color, at) =>
    add(parent, new T.BoxGeometry(...size), color, at);
  const cylinder = (parent, radius, height, color, at) =>
    add(parent, new T.CylinderGeometry(radius, radius, height, 32), color, at);
  const interactive = (id, object, anchor) => {
    object.traverse((child) => {
      child.userData.action = id;
    });
    items.set(id, { object, anchor: new T.Vector3(...anchor) });
  };
  const placeFurniture = (object, part, y = 0) => {
    object.position.set(part.x,y,part.z);
    object.rotation.y = part.turn;
    object.name = `Furniture ${part.id}`;
  };
  const overviewGarden = new T.Group();
  group.add(overviewGarden);
  cylinder(overviewGarden, 2.35, 0.2, "#769987", [0, -0.16, 0]);
  cylinder(overviewGarden, 2.3, 0.035, "#d8c5a2", [0, -0.045, 0]);
  cylinder(overviewGarden, 2.19, 0.02, "#90ab96", [0, -0.018, 0]);
  const timber = createTimberTextures();
  const interior = createStudioInterior(timber);
  group.add(interior);
  // Warm timber paths, a rug, and a bounded garden make this a room rather than a void.
  for (let i = 0; i < 7; i++)
    box(group, [0.09, 0.012, 1.3], "#c4ba98", [-0.32 + i * 0.11, 0.002, 1.48]);
  const rug = cylinder(group, 0.99, 0.012, "#718e81", [0, 0.004, 0.33]);
  rug.name = "Studio rug";
  rug.scale.z = 0.8;
  for (let i = 0; i < 14; i++) {
    const angle = Math.PI * (0.1 + i / 16);
    const x = Math.cos(angle) * 2.08,
      z = -Math.sin(angle) * 2.08;
    cylinder(overviewGarden, 0.065, 0.35, "#cab797", [x, 0.17, z]);
    const bush = add(
      overviewGarden,
      new T.IcosahedronGeometry(0.24, 1),
      i % 2 ? "#658d70" : "#b0be89",
      [x, 0.39, z],
    );
    bush.scale.y = 0.75;
  }
  const table = new T.Group();
  placeFurniture(table, STUDIO_LAYOUT.table);
  group.add(table);
  add(table,
    timberUV(new T.CylinderGeometry(STUDIO_LAYOUT.table.radius, STUDIO_LAYOUT.table.radius, .065, 32), "table"),
    timberMaterial("#d4af7c", timber, .76), [0, .43, 0]);
  cylinder(table, 0.065, 0.41, "#677d72", [0, 0.2, 0]);
  cylinder(table, 0.24, 0.03, "#556e66", [0, 0.015, 0]);
  interactive("return-note", table, [0, 0.5, 0.35]);
  const mug = (x, z, color) => {
    const cup = new T.Group();
    cup.name = "Coffee cup";
    cup.position.set(x, 0.495, z);
    group.add(cup);
    cylinder(cup, 0.045, 0.075, color, [0, 0, 0]);
    cylinder(cup, 0.037, 0.006, "#573d32", [0, 0.039, 0]);
    add(cup, new T.TorusGeometry(0.026, 0.009, 8, 16), color, [0.05, 0, 0]);
    // WebGL defaults opaque casters to back faces. At this small scale that
    // leaks lit samples into the cup's contact edge; use its outer silhouette.
    cup.traverse((part) => {
      if (part.isMesh) part.material.shadowSide = T.FrontSide;
    });
    return cup;
  };
  const cup = mug(-0.23, 0.29, "#ecdcc2");
  mug(0.23, 0.47, "#a8c9cf");
  interactive("coffee", cup, [-0.22, 0.67, 0.3]);
  const paper = new T.Group();
  group.add(paper);
  paper.add(createPaperSheet(createWorkingNoteTexture(studio)));
  paper.position.set(0, 0.473, 0.37);
  interactive("paper", paper, [0.02, 0.57, 0.52]);
  const grip = createPaperGrip();
  paper.add(grip);
  grip.visible = false;
  const resultPaper = new T.Group();
  group.add(resultPaper);
  const resultCanvas = document.createElement("canvas");
  resultCanvas.width = 512;
  resultCanvas.height = 640;
  const resultContext = resultCanvas.getContext("2d"),
    resultTexture = new T.CanvasTexture(resultCanvas);
  resultTexture.colorSpace = T.SRGBColorSpace;
  resultPaper.add(createPaperSheet(resultTexture));
  const resultGrip = grip.clone(true);
  resultPaper.add(resultGrip);
  resultPaper.traverse((child) => {
    child.userData.action = "held-result";
  });
  resultPaper.visible = false;
  let lastResultCard = null;
  const writeResult = (card) => {
    if (card === lastResultCard) return;
    lastResultCard = card;
    if (!card) return;
    const view = resultCardView(card),
      ctx = resultContext;
    ctx.fillStyle = "#eee8d1";
    ctx.fillRect(0, 0, 512, 640);
    ctx.fillStyle = "#2c5949";
    ctx.fillRect(0, 0, 512, 74);
    ctx.fillStyle = "#f7edcc";
    ctx.font = "23px sans-serif";
    ctx.fillText(`BROWSER TOY / CARD ${card.id}`, 25, 47);
    ctx.fillStyle = "#264d3b";
    ctx.font = "30px Georgia";
    ctx.fillText(view.label, 25, 125, 462);
    ctx.font = "64px Georgia";
    ctx.fillText(view.own, 25, 215, 462);
    ctx.font = "23px sans-serif";
    ctx.fillText(view.location, 25, 275, 462);
    ctx.fillText(view.parameter, 25, 323, 462);
    ctx.fillText(`Baseline: ${view.baseline}`, 25, 393, 462);
    ctx.fillText(`Difference: ${view.delta}`, 25, 440, 462);
    ctx.fillStyle = "#5c765d";
    ctx.font = "21px sans-serif";
    ctx.fillText("Stored calculation", 25, 552);
    ctx.fillText("Not research evidence", 25, 585);
    resultTexture.needsUpdate = true;
  };
  const researcher = cloneResearchRobot(robotTemplate) || createResearcher(studio.color);
  const presentResearcher = createRobotPresence(researcher);
  const presentArm = createOfferingArm(researcher);
  researcher.scale.setScalar(robotTemplate ? 1.75 : 1.6);
  placeFurniture(researcher, STUDIO_LAYOUT.researcher, .05);
  group.add(researcher);
  // Root origins differ between the Blender robot and the fallback character.
  // Align the actual resting sole to the rug, not an assumed model-space zero.
  const rugTop = new T.Box3().setFromObject(rug).max.y;
  const sole = new T.Box3().setFromObject(researcher).min.y;
  if (Number.isFinite(rugTop) && Number.isFinite(sole))
    researcher.position.y += rugTop - sole;
  interactive("greet", researcher, [-0.78, 0.93, 0.15]);
  const visitor = createResearcher("#85b5c3");
  visitor.scale.setScalar(1.45);
  visitor.position.set(0.83, 0, 0.85);
  visitor.rotation.y = -2.3;
  group.add(visitor);
  for (const part of [STUDIO_LAYOUT.researcherChair, STUDIO_LAYOUT.visitorChair]) {
    const chair = new T.Group();
    placeFurniture(chair, part);
    group.add(chair);
    box(chair, [part.width, 0.1, 0.35], "#cba17f", [0, 0.16, 0]);
    box(chair, [part.width, 0.34, 0.075], "#cba17f", [0, 0.34, -0.19]);
    for (const dx of [-0.15, 0.15])
      for (const dz of [-0.12, 0.12])
        box(chair, [0.04, 0.16, 0.04], "#526e64", [dx, 0.06, dz]);
    if (part.id === "visitor-chair")
      interactive("seat", chair, [part.x, .55, part.z]);
  }
  const boardGroup = new T.Group();
  placeFurniture(boardGroup, STUDIO_LAYOUT.board);
  group.add(boardGroup);
  box(boardGroup, [STUDIO_LAYOUT.board.width, 0.82, STUDIO_LAYOUT.board.depth], "#c6ab83", [0, 0.82, 0]);
  for (const x of [-0.4, 0.4])
    box(boardGroup, [0.045, 0.57, 0.05], "#b79772", [x, 0.28, 0]);
  const canvas = document.createElement("canvas");
  canvas.width = 768;
  canvas.height = 512;
  const context = canvas.getContext("2d"),
    texture = new T.CanvasTexture(canvas);
  texture.colorSpace = T.SRGBColorSpace;
  add(
    boardGroup,
    new T.PlaneGeometry(1.04, 0.74),
    new T.MeshBasicMaterial({ map: texture }),
    [0, 0.82, 0.036],
  );
  const pinDisplay = new T.Group();
  pinDisplay.name = "Results side panel";
  boardGroup.add(pinDisplay);
  box(pinDisplay, [STUDIO_LAYOUT.board.pinWidth, 0.84, 0.045], "#bca079", [STUDIO_LAYOUT.board.pinX, 0.82, 0]);
  const pinCanvas = document.createElement("canvas");
  pinCanvas.width = 512;
  pinCanvas.height = 768;
  const pinContext = pinCanvas.getContext("2d"),
    pinTexture = new T.CanvasTexture(pinCanvas);
  pinTexture.colorSpace = T.SRGBColorSpace;
  add(
    pinDisplay,
    new T.PlaneGeometry(0.54, 0.79),
    new T.MeshBasicMaterial({ map: pinTexture }),
    [STUDIO_LAYOUT.board.pinX, 0.82, 0.027],
  );
  interactive("board", boardGroup, [-1.12, 1.35, -0.85]);
  const writeBoard = (board, cards = []) => {
    const pinned = (board.pinnedResultIds || [])
      .map((id) => cards.find((card) => card.id === id))
      .filter(Boolean);
    if (pinned.length) {
      const ctx = pinContext;
      ctx.fillStyle = "#516d54";
      ctx.fillRect(0, 0, 512, 768);
      ctx.fillStyle = "#f3e7c5";
      ctx.font = "23px sans-serif";
      ctx.fillText(`${pinned.length} PINNED / LATEST CARDS`, 24, 40, 464);
      pinned
        .slice(-2)
        .reverse()
        .forEach((card, i) => {
          const view = resultCardView(card),
            y = 70 + i * 338;
          ctx.fillStyle = "#f0e8cf";
          ctx.fillRect(22, y, 468, 312);
          ctx.fillStyle = "#ac7651";
          ctx.beginPath();
          ctx.arc(254, y + 14, 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#315441";
          ctx.font = "22px sans-serif";
          ctx.fillText(`TOY / CARD ${card.id}`, 43, y + 49);
          ctx.font = "27px Georgia";
          ctx.fillText(view.label, 43, y + 96, 420);
          ctx.font = "47px Georgia";
          ctx.fillText(view.own, 43, y + 155, 420);
          ctx.font = "22px sans-serif";
          ctx.fillText(view.location, 43, y + 207, 420);
          ctx.fillText(`Baseline ${view.baseline}`, 43, y + 248, 420);
          ctx.fillText(`Δ ${view.delta}`, 43, y + 284, 420);
        });
      pinTexture.needsUpdate = true;
    } else {
      pinContext.fillStyle = "#516d54";
      pinContext.fillRect(0, 0, 512, 768);
      pinContext.fillStyle = "#f3e7c5";
      pinContext.font = "28px Georgia";
      pinContext.fillText("Results board", 35, 80);
      pinContext.font = "23px sans-serif";
      pinContext.fillText("Pin a calculation here", 35, 150);
      pinContext.fillText("to discuss it together.", 35, 187);
      pinTexture.needsUpdate = true;
    }
    context.fillStyle = "#25483e";
    context.fillRect(0, 0, 768, 512);
    context.fillStyle = "#e9ddbc";
    context.font = "23px sans-serif";
    context.fillText("OUR SHARED QUESTION", 36, 49);
    context.font = "26px Georgia";
    const words = (board.question || "What are you curious about?")
      .slice(0, 140)
      .split(" ");
    let line = "",
      y = 95;
    for (const word of words) {
      if (context.measureText(`${line} ${word}`).width > 690) {
        context.fillText(line, 36, y);
        y += 33;
        line = word;
      } else line += `${line ? " " : ""}${word}`;
    }
    context.fillText(line, 36, y);
    context.font = "18px sans-serif";
    context.fillStyle = "#c1d1b7";
    context.fillText(
      `Assume: ${board.assumption || "What might not be true?"}`,
      36,
      187,
      695,
    );
    context.fillText(
      `Try: ${board.test || "What would change our mind?"}`,
      36,
      219,
      695,
    );
    context.strokeStyle = "#ead5a3";
    context.lineWidth = 3;
    context.lineCap = "round";
    context.lineJoin = "round";
    for (const stroke of board.strokes || []) {
      context.beginPath();
      stroke.forEach(([x, z], i) =>
        context[i ? "lineTo" : "moveTo"](x * 768, 235 + z * 245),
      );
      context.stroke();
    }
    if (!(board.strokes || []).length) {
      context.font = "19px sans-serif";
      context.fillStyle = "#b6cbb0";
      context.fillText("A thought → an assumption → a small test", 36, 320);
    }
    texture.needsUpdate = true;
  };
  const lab = createStudio(studio);
  placeFurniture(lab, STUDIO_LAYOUT.lab, .02);
  lab.scale.setScalar(1.02);
  // Keep the entire ideal arc over the open right-hand strip of the studio,
  // away from the board and coffee table. The model still owns local SI axes.
  if (studio.id === "physics") lab.rotation.y = Math.PI / 2;
  lab.userData.researcher.visible = false;
  group.add(lab);
  interactive("experiment", lab, [0.94, 1.15, -0.85]);
  // Same ideal path as the bench, uniformly scaled to 0.04 studio units/metre.
  // The meeting owns this rocket; the globe's periodic decorative launch is not sampled.
  let lastAim = null,
    aimPath = null,
    launcherPivot = null,
    landingMarker = null;
  let lastKinetic = "";
  const reactant = new T.Color("#dfb578"),
    product = new T.Color("#75aaa0");
  const pathPoints = (angle) =>
    Array.from({ length: 61 }, (_, i) => {
      const p = projectile(angle, i / 60);
      return new T.Vector3(0.13 - p.x * 0.04, 0.08 + p.y * 0.04, -0.06);
    });
  if (studio.id === "physics") {
    launcherPivot = new T.Group();
    launcherPivot.position.set(0.13, 0.08, 0.055);
    lab.add(launcherPivot);
    box(launcherPivot, [0.055, 0.36, 0.045], "#668c7b", [0, 0.17, 0]);
    add(
      launcherPivot,
      new T.TorusGeometry(0.07, 0.016, 8, 24),
      "#edbd79",
      [0, 0.35, 0.02],
    );
    const dial = add(
      lab,
      new T.TorusGeometry(0.28, 0.009, 6, 48, Math.PI * 0.3),
      "#c4c7a1",
      [0.13, 0.08, 0.06],
    );
    dial.rotation.z = Math.PI * 0.59;
    const baseline = new T.Line(
      new T.BufferGeometry().setFromPoints(pathPoints(45)),
      new T.LineDashedMaterial({
        color: "#c0c7af",
        dashSize: 0.035,
        gapSize: 0.025,
        transparent: true,
        opacity: 0.7,
      }),
    );
    baseline.computeLineDistances();
    baseline.raycast = () => {};
    lab.add(baseline);
    aimPath = new T.Line(
      new T.BufferGeometry().setFromPoints(pathPoints(45)),
      new T.LineBasicMaterial({
        color: "#efc283",
        transparent: true,
        opacity: 0.8,
      }),
    );
    aimPath.raycast = () => {};
    lab.add(aimPath);
    lab.traverse((object) => {
      object.userData.action = "launcher";
    });
    landingMarker = createLandingTargetScene();
    lab.add(landingMarker.group);
  }
  const bookcase = new T.Group();
  placeFurniture(bookcase, STUDIO_LAYOUT.bookcase);
  group.add(bookcase);
  for (const y of [0.12, 0.4])
    box(bookcase, [STUDIO_LAYOUT.bookcase.width, 0.045, STUDIO_LAYOUT.bookcase.depth], "#b19778", [0, y, 0]);
  for (let i = 0; i < 8; i++)
    box(
      bookcase,
      [0.07, 0.19 + (i % 3) * 0.025, 0.15],
      ["#99ac9c", "#d2b694", "#8ba4b5"][i % 3],
      [-0.31 + i * 0.09, 0.25, 0],
    );
  group.updateMatrixWorld(true);
  const conversation = conversationSubject(researcher);
  const equipmentBounds = new T.Box3().setFromObject(lab).expandByScalar(0.16);
  if (studio.id === "physics") {
    const body = lab.userData.rocket.clone(true);
    body.position.set(0, 0, 0);
    body.rotation.set(0, 0, 0);
    body.updateMatrixWorld(true);
    equipmentBounds.union(
      launcherFlightBounds(new T.Box3().setFromObject(body), lab.matrixWorld),
    );
    equipmentBounds.union(landingMarker.bounds.clone().applyMatrix4(lab.matrixWorld));
    // Clone shares geometry/materials; it owns no extra GPU resources to dispose.
  }
  const noteMotion = createNoteMotion(paper);
  const coffeeMotion = createCoffeeMotion(cup, researcher);
  const resultMotion = createNoteMotion(resultPaper);
  const resultRest = {
    position: new T.Vector3(0.94, 0.68, -0.85),
    quaternion: new T.Quaternion(),
    scale: 1,
  };
  const handPosition = new T.Vector3(),
    handQuaternion = new T.Quaternion();
  const paperTilt = new T.Quaternion().setFromEuler(new T.Euler(-0.2, 0, 0.1));
  return {
    group,
    items,
    visitor,
    writeBoard,
    equipmentBounds,
    conversation,
    paper,
    presentResearcher,
    presentArm,
    presentCoffee(args) {
      const pose = coffeeMotion(args);
      items.get("coffee").anchor.copy(cup.position).add(new T.Vector3(.01, .175, .01));
      return pose;
    },
    presentLanding(challenge, pendingTargetId) {
      if (!landingMarker) return;
      const target = presentedLandingTarget(challenge, pendingTargetId);
      const shot = challenge.shots.at(-1);
      landingMarker.present(target, shot?.targetId === target?.id && shot?.status === "landed" && shot.hit);
    },
    presentNote(args) {
      visitor.userData.arm.rotation.x =
        args.held || args.resultCard ? -1.35 : 0;
      visitor.updateWorldMatrix(true, true);
      handPosition.set(0, -0.13, 0.055);
      visitor.userData.arm.localToWorld(handPosition);
      visitor.getWorldQuaternion(handQuaternion).multiply(paperTilt);
      grip.visible = args.held && args.firstPerson;
      resultPaper.visible = Boolean(args.resultCard);
      resultGrip.visible = Boolean(args.resultCard && args.firstPerson);
      writeResult(args.resultCard);
      resultMotion({
        ...args,
        held: Boolean(args.resultCard),
        revision: args.resultRevision,
        restPose: resultRest,
        overviewHand: { position: handPosition, quaternion: handQuaternion },
      });
      const pose = noteMotion({
        ...args,
        restPose: args.offered ? offeredNotePose(researcher) : undefined,
        overviewHand: { position: handPosition, quaternion: handQuaternion },
      });
      if (args.offered) items.get("paper").anchor.copy(paper.position).y += .08;
      else items.get("paper").anchor.set(.02, .57, .52);
      return pose;
    },
    animate(time, gesture, age, parameter, firstPerson, flight, kinetics) {
      overviewGarden.visible = !firstPerson;
      interior.visible = firstPerson;
      visitor.visible = !firstPerson;
      researcher.rotation.y =
        STUDIO_LAYOUT.researcher.turn +
        (gesture.type === "board" ? -0.65 * Math.max(0, 1 - age / 2) : 0);
      if (lab.userData.rocket) {
        const angle = Number.isFinite(parameter) ? parameter : 45;
        if (lastAim !== angle) {
          aimPath.geometry.setFromPoints(pathPoints(angle));
          launcherPivot.rotation.z = Math.PI / 2 - (angle * Math.PI) / 180;
          lastAim = angle;
        }
        const rocket = lab.userData.rocket;
        rocket.visible = true;
        rocket.position.set(
          0.13 - (flight?.x || 0) * 0.04,
          0.08 + (flight?.y || 0) * 0.04,
          -0.06,
        );
        rocket.rotation.z =
          Math.PI / 2 -
          (flight
            ? flight.landed
              ? 0
              : flight.heading
            : (angle * Math.PI) / 180);
      } else {
        lab.userData.animate(
          kinetics
            ? kinetics.time
            : gesture.type === "experiment"
              ? 6 + Math.min(age, 4)
              : time * 0.6,
        );
        if (kinetics) {
          const identity = `${kinetics.id}:${kinetics.index}:${kinetics.parameter}`;
          if (identity !== lastKinetic) {
            lastKinetic = identity;
            if (lab.userData.vials) {
              // Symbolic fraction-to-color mapping, not the physical color of a chemical.
              lab.userData.vials[0].material.color
                .copy(reactant)
                .lerp(product, 1 - kinetics.baseline);
              lab.userData.vials[1].material.color
                .copy(reactant)
                .lerp(product, 1 - kinetics.own);
              lab.userData.kineticBoard.userData.write([
                "REACTION TOY",
                `t = ${kinetics.time.toFixed(1)} s`,
                `A: ${(kinetics.own * 100).toFixed(1)}%`,
              ]);
            }
            if (lab.userData.growthPlants) {
              // Plant sizes symbolize normalized population; no botanical simulation.
              lab.userData.growthPlants[0].scale.setScalar(
                0.18 + 0.52 * Math.sqrt(kinetics.baseline),
              );
              lab.userData.growthPlants[1].scale.setScalar(
                0.18 + 0.52 * Math.sqrt(kinetics.own),
              );
            }
          }
        }
      }
    },
  };
}
