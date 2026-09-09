import { useEffect, useRef, useState } from "react";
import * as T from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { createMeetingScene } from "./meetingScene.js";
import { disposeScene } from "./researchPlanetScene.js";
import {
  ROOM_ACTIONS,
  stepVisitor,
  advanceGestureClock,
  initialGestureClock,
} from "../lib/meetingInteraction.js";
import Icon from "./WorldIcon.jsx";
import { kineticsFrame } from "../lib/kineticsBench.js";
import { inspectionPose, inspectionBlend, conversationPose } from "../lib/studioCamera.js";
import { canReturnNote } from "../lib/carriedNote.js";
import "../carried-note.css";
import { canPinResult, heldResultCard } from "../lib/resultCards.js";
import {
  draggedLaunchAngle,
  keyedLaunchAngle,
} from "../lib/launcherInteraction.js";
import RoomLauncher from "./RoomLauncher.jsx";
import { createStudioLighting } from "../lib/studioLighting.js";
import { VISITOR_SPAWN } from "../lib/studioLayout.js";
import { createCoffeeSeat } from "../lib/studioSeat.js";
import { studioPrompt } from "../lib/studioPrompt.js";
import RoomControls from "./RoomControls.jsx";
import WalkPad from "./WalkPad.jsx";
import { createWalkInput } from "../lib/walkInput.js";
const ROOM_LABEL =
  "Research room game. WASD to walk, mouse or drag to look, E to interact, G to sit or stand, R to read a held item, Q to put away. Escape releases the mouse.";
const LAUNCHER_LABEL =
  "Launcher aiming. Drag up or down, or use Up and Down arrows to aim. Space or Enter launches. Escape returns to the room.";

export default function InteractiveStudio({
  studio,
  value,
  paused,
  reducedMotion,
  dispatch,
  apiRef,
  flightRef,
  calculation,
  robotTemplate,
  onViewChange,
  actions,
  dialogueActive = false,
}) {
  const hostRef = useRef(null),
    labels = useRef(new Map()),
    stateRef = useRef({});
  const [mode, setMode] = useState("overview"),
    [locked, setLocked] = useState(false);
  const [target, setTarget] = useState(null),
    [error, setError] = useState(""),
    [failed, setFailed] = useState(false);
  const [nearTable, setNearTable] = useState(false);
  const [nearBoard, setNearBoard] = useState(false);
  const [flying, setFlying] = useState(false);
  const [seated, setSeated] = useState(false);
  useEffect(() => { onViewChange?.(mode); }, [mode, onViewChange]);
  useEffect(() => {
    if (dialogueActive) {
      apiRef.current?.release();
      setError((message) => message.startsWith("Mouse capture") ? "" : message);
    }
  }, [dialogueActive, apiRef]);
  const carriedResult = heldResultCard(value);
  const prompt = studioPrompt(mode, target, value.noteHeld, carriedResult, seated);
  const compactHud = mode === "first-person" && !value.launcherActive && !failed;
  const instructions = mode === "first-person"
    ? value.noteOffered
      ? "Aim at researcher · E take · Q leave note · T talk"
      : seated
        ? "Seated · Mouse / drag look · T talk · G stand · WASD stands & walks"
        : "WASD move · Mouse / drag look · E interact · G sit · T talk · Esc release"
    : "Click an object to interact · Drag to orbit · Enter first-person to walk";
  stateRef.current = { value, paused, reducedMotion, dispatch, calculation, dialogueActive };
  useEffect(() => {
    const host = hostRef.current;
    let renderer;
    try {
      renderer = new T.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "low-power",
      });
    } catch {
      setFailed(true);
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
    renderer.outputColorSpace = T.SRGBColorSpace;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFShadowMap;
    const canvas = renderer.domElement;
    canvas.tabIndex = 0;
    canvas.setAttribute("aria-label", ROOM_LABEL);
    host.prepend(canvas);
    const scene = new T.Scene(),
      camera = new T.PerspectiveCamera(44, 1, 0.05, 35);
    let environment = null, environmentScene = null, pmrem = null;
    try {
      environmentScene = new RoomEnvironment();
      pmrem = new T.PMREMGenerator(renderer);
      environment = pmrem.fromScene(environmentScene, 0.04);
      scene.environment = environment.texture;
      scene.environmentIntensity = 0.55;
    } catch {
      // Reflection lighting is optional; basic scene lights retain a usable room.
    } finally {
      environmentScene?.dispose();
      pmrem?.dispose();
    }
    const lighting = createStudioLighting(scene);
    const room = createMeetingScene(studio, robotTemplate);
    scene.add(room.group);
    let portrait = null;
    let controls,
      firstPerson = false,
      width = 1,
      height = 1,
      dragging = null,
      lastUnlock = -Infinity;
    let motionClock = initialGestureClock(stateRef.current.value.gesture.id),
      previous = null,
      lastBoard = null;
    let yaw = 0.75,
      pitch = -0.07,
      player = { ...VISITOR_SPAWN },
      aimed = null,
      lastNearTable = null,
      lastNearBoard = null,
      lastCards = null;
    let aiming = false,
      lastFlying = null;
    let inspecting = false,
      previousView = null,
      handoff = null;
    const seat = createCoffeeSeat();
    const walkInput = createWalkInput();
    const held = new Set(),
      ray = new T.Raycaster(),
      point = new T.Vector2(),
      projected = new T.Vector3();
    const clearLook = () => {
      const pointerId = dragging?.pointerId;
      dragging = null;
      if (
        pointerId !== undefined &&
        canvas.hasPointerCapture(pointerId)
      )
        canvas.releasePointerCapture(pointerId);
    };
    const clear = () => {
      held.clear();
      walkInput.clear();
      clearLook();
    };
    const applyFirstPerson = () => {
      const at = seat.position(player);
      camera.position.set(at.x, seat.height, at.z);
      camera.rotation.set(pitch, yaw, 0, "YXZ");
      if (camera.fov !== seat.fov) {
        camera.fov = seat.fov;
        camera.updateProjectionMatrix();
      }
    };
    const stand = () => {
      const look = seat.stand();
      if (!look) return;
      yaw = look.yaw;
      pitch = look.pitch;
      clear();
      setSeated(false);
    };
    const acquireOrbit = (target, inspection = false) => {
      controls?.dispose();
      controls = new OrbitControls(camera, canvas);
      controls.target.copy(target);
      controls.enablePan = false;
      controls.enabled = !aiming;
      controls.minDistance = inspection ? 0.8 : 3;
      controls.maxDistance = inspection
        ? Math.max(12, camera.position.distanceTo(target) * 1.8)
        : 8;
      controls.minPolarAngle = 0.15;
      controls.maxPolarAngle = Math.PI / 2.05;
      controls.update();
    };
    const beginHandoff = (pose) => {
      clear();
      controls?.dispose();
      controls = null;
      handoff = {
        from: {
          position: camera.position.clone(),
          quaternion: camera.quaternion.clone(),
          fov: camera.fov,
        },
        to: pose,
        elapsed: 0,
      };
    };
    const inspectEquipment = () => {
      // Opening the controls while walking cannot teleport the visitor.
      if (firstPerson) return;
      const pose = inspectionPose(room.equipmentBounds, width / height);
      if (!pose) return;
      if (!inspecting)
        previousView = {
          position: camera.position.clone(),
          quaternion: camera.quaternion.clone(),
          target: controls?.target.clone() || new T.Vector3(0, 0.35, 0),
          fov: camera.fov,
        };
      inspecting = true;
      setMode("inspection");
      beginHandoff(pose);
    };
    const restoreView = () => {
      if (!inspecting) return;
      inspecting = false;
      const pose = previousView;
      previousView = null;
      if (firstPerson || !pose) return;
      setMode("overview");
      beginHandoff(pose);
    };
    const overview = () => {
      if (stateRef.current.value.launcherActive)
        stateRef.current.dispatch({ type: "launcher", action: "close" });
      if (document.pointerLockElement === canvas) document.exitPointerLock();
      clear();
      stand();
      firstPerson = false;
      inspecting = false;
      previousView = null;
      handoff = null;
      setMode("overview");
      setSeated(false);
      setTarget(null);
      camera.fov = 44;
      camera.position.set(3.35, 3.0, 4.2);
      camera.up.set(0, 1, 0);
      acquireOrbit(new T.Vector3(0, 0.35, 0));
      resize();
    };
    const start = () => {
      if (stateRef.current.value.launcherActive)
        stateRef.current.dispatch({ type: "launcher", action: "close" });
      if (!firstPerson) {
        handoff = null;
        inspecting = false;
        previousView = null;
        controls?.dispose();
        controls = null;
        clear();
        firstPerson = true;
        setMode("first-person");
        camera.fov = 64;
        camera.updateProjectionMatrix();
        applyFirstPerson();
      }
      canvas.focus();
      setError("");
      if (!canvas.requestPointerLock) {
        setError(
          "Mouse capture is unavailable. Drag to look; WASD still moves.",
        );
        return;
      }
      try {
        const request = canvas.requestPointerLock();
        request?.catch(() =>
          setError(
            "Mouse capture was not allowed. Drag to look; WASD still moves.",
          ),
        );
      } catch {
        setError(
          "Mouse capture was not allowed. Drag to look; WASD still moves.",
        );
      }
    };
    const release = () => {
      clear();
      if (document.pointerLockElement === canvas) document.exitPointerLock();
    };
    const takeSeat = () => {
      if (aiming) return;
      clear();
      setError("");
      if (seat.seated) stand();
      else {
        const look = seat.sit(yaw, pitch);
        yaw = look.yaw;
        pitch = look.pitch;
        // An explicit cut avoids travelling through the table from an arbitrary
        // walking position. No orbit/handoff writer survives the mode change.
        handoff = null;
        inspecting = false;
        previousView = null;
        controls?.dispose();
        controls = null;
        firstPerson = true;
        setMode("first-person");
        setSeated(true);
      }
      applyFirstPerson();
      camera.updateMatrixWorld();
      canvas.focus();
    };
    const noteAction = (action) => {
      if (action === "return" && !canReturnNote(seat.position(player), firstPerson)) {
        setError("Move closer to the coffee table to put the note down.");
        return;
      }
      if (action === "read") release();
      setError("");
      stateRef.current.dispatch(
        action === "read" && stateRef.current.value.material === "paper"
          ? { type: "close-material" }
          : { type: "note", action },
      );
    };
    const resultAction = (action) => {
      if (action === "pin" && !canPinResult(seat.position(player), firstPerson)) {
        setError("Move closer to the shared board to pin this result.");
        return;
      }
      release();
      setError("");
      stateRef.current.dispatch(
        action === "read" && stateRef.current.value.material === "result"
          ? { type: "close-material" }
          : { type: "result-card", action },
      );
    };
    const interact = (id) => {
      if (!id) return;
      if (id === "offered-note") {
        if (stateRef.current.value.noteOffered) noteAction("take");
        return;
      }
      if (id === "seat") {
        takeSeat();
        return;
      }
      if (id === "launcher") {
        release();
        stateRef.current.dispatch({ type: "launcher", action: "open" });
        return;
      }
      if (id === "held-result") {
        resultAction("read");
        return;
      }
      if (id === "board" && heldResultCard(stateRef.current.value)) {
        resultAction("pin");
        return;
      }
      if (id === "return-note") {
        if (stateRef.current.value.noteHeld) noteAction("return");
        return;
      }
      if (!(id === "paper" && !stateRef.current.value.noteHeld)) release();
      stateRef.current.dispatch({ type: "interact", id });
    };
    const pick = (x = 0, y = 0, limit = Infinity) => {
      point.set(x, y);
      ray.setFromCamera(point, camera);
      // Three raycasts include invisible objects. Stored cards, hidden board
      // extensions and the first-person avatar must not intercept interactions.
      const hit = ray.intersectObject(room.group, true).find(({ object }) => {
        for (let node = object; node; node = node.parent)
          if (!node.visible) return false;
        return true;
      });
      if (!hit || hit.distance > limit) return null;
      const action = hit.object.userData.action;
      if (stateRef.current.value.noteOffered && ["paper", "greet"].includes(action))
        return "offered-note";
      if (action === "return-note" && !stateRef.current.value.noteHeld)
        return null;
      return action || null;
    };
    const lockChange = () => {
      const isLocked = document.pointerLockElement === canvas;
      setLocked(isLocked);
      clear();
      if (!isLocked) lastUnlock = performance.now();
    };
    const lockError = () =>
      setError(
        "Mouse capture is unavailable here. Drag to look; all actions also have buttons.",
      );
    const moveMouse = (event) => {
      if (stateRef.current.dialogueActive) return;
      if (!firstPerson || aiming) return;
      if (document.pointerLockElement === canvas) {
        yaw -= event.movementX * 0.0024;
        pitch = T.MathUtils.clamp(
          pitch - event.movementY * 0.0024,
          -1.05,
          0.95,
        );
      }
    };
    const down = (event) => {
      if (stateRef.current.dialogueActive) return;
      if (event.button !== 0) return;
      if (handoff) {
        // A drag takes ownership at the delivered pose, without snapping to the
        // uncompleted destination or replaying latent OrbitControls state.
        const distance = camera.position.distanceTo(handoff.to.target);
        const target = camera
          .getWorldDirection(new T.Vector3())
          .multiplyScalar(distance)
          .add(camera.position);
        handoff = null;
        acquireOrbit(target, inspecting);
      }
      setError("");
      canvas.focus();
      if (aiming) {
        event.preventDefault();
        event.stopImmediatePropagation();
        dragging = {
          kind: "aim",
          pointerId: event.pointerId,
          y: event.clientY,
          angle: stateRef.current.value.illustration.parameter ?? 45,
        };
        canvas.setPointerCapture(event.pointerId);
        return;
      }
      dragging = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        lastX: event.clientX,
        lastY: event.clientY,
        moved: false,
      };
      if (firstPerson && document.pointerLockElement !== canvas)
        canvas.setPointerCapture(event.pointerId);
    };
    const drag = (event) => {
      if (stateRef.current.dialogueActive) return;
      if (
        aiming &&
        dragging?.kind === "aim" &&
        dragging.pointerId === event.pointerId
      ) {
        event.preventDefault();
        stateRef.current.dispatch({
          type: "launcher",
          action: "aim",
          angle: draggedLaunchAngle(dragging.angle, dragging.y, event.clientY),
        });
        return;
      }
      if (!dragging || !firstPerson || document.pointerLockElement === canvas)
        return;
      const dx = event.clientX - dragging.lastX,
        dy = event.clientY - dragging.lastY;
      dragging.moved ||=
        Math.hypot(event.clientX - dragging.x, event.clientY - dragging.y) > 5;
      yaw -= dx * 0.004;
      pitch = T.MathUtils.clamp(pitch - dy * 0.004, -1.05, 0.95);
      dragging.lastX = event.clientX;
      dragging.lastY = event.clientY;
    };
    const up = (event) => {
      const origin = dragging;
      dragging = null;
      if (canvas.hasPointerCapture(event.pointerId))
        canvas.releasePointerCapture(event.pointerId);
      if (origin?.kind === "aim") return;
      if (
        !origin ||
        origin.moved ||
        Math.hypot(event.clientX - origin.x, event.clientY - origin.y) > 5
      )
        return;
      if (document.pointerLockElement === canvas) interact(pick(0, 0, 2.5));
      else {
        const rect = canvas.getBoundingClientRect();
        interact(
          pick(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            1 - ((event.clientY - rect.top) / rect.height) * 2,
          ),
        );
      }
    };
    const keyDown = (event) => {
      if (stateRef.current.dialogueActive) { clear(); return; }
      if (
        event.target?.isContentEditable ||
        /^(INPUT|TEXTAREA|SELECT)$/.test(event.target?.tagName)
      ) {
        release();
        return;
      }
      if (
        event.key.toLowerCase() === "r" &&
        !event.repeat &&
        stateRef.current.value.material === "paper" &&
        event.target?.closest?.(".meeting-note")
      ) {
        event.preventDefault();
        noteAction("read");
        return;
      }
      if (
        !event.repeat &&
        event.target?.closest?.(".meeting-result-reader") &&
        ["r", "q"].includes(event.key.toLowerCase())
      ) {
        event.preventDefault();
        resultAction(event.key.toLowerCase() === "r" ? "read" : "store");
        return;
      }
      if (
        document.activeElement !== canvas &&
        document.pointerLockElement !== canvas
      )
        return;
      if (aiming) {
        const angle = keyedLaunchAngle(
          stateRef.current.value.illustration.parameter ?? 45,
          event.key,
          event.shiftKey,
        );
        if (angle !== null) {
          event.preventDefault();
          stateRef.current.dispatch({ type: "launcher", action: "aim", angle });
        } else if ([" ", "Enter"].includes(event.key)) {
          event.preventDefault();
          if (!event.repeat) stateRef.current.calculation.run();
        } else if (event.key === "Escape") {
          event.preventDefault();
          stateRef.current.dispatch({ type: "launcher", action: "close" });
        }
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "q" && stateRef.current.value.noteOffered && !event.repeat) {
        event.preventDefault();
        noteAction("decline");
        return;
      }
      if (
        ["r", "q"].includes(key) &&
        heldResultCard(stateRef.current.value) &&
        !event.repeat
      ) {
        event.preventDefault();
        resultAction(key === "r" ? "read" : "store");
        return;
      }
      if (
        ["r", "q"].includes(key) &&
        stateRef.current.value.noteHeld &&
        !event.repeat
      ) {
        event.preventDefault();
        if (key === "r") noteAction("read");
        else noteAction("return");
        return;
      }
      if (!firstPerson) return;
      if (key === "g" && !event.repeat) {
        event.preventDefault();
        takeSeat();
        return;
      }
      if (
        [
          "w",
          "a",
          "s",
          "d",
          "arrowup",
          "arrowdown",
          "arrowleft",
          "arrowright",
          "shift",
          "e",
        ].includes(key)
      ) {
        event.preventDefault();
        held.add(key);
        if (!event.repeat) {
          const step = {
            w: [1, 0],
            arrowup: [1, 0],
            s: [-1, 0],
            arrowdown: [-1, 0],
            a: [0, -1],
            arrowleft: [0, -1],
            d: [0, 1],
            arrowright: [0, 1],
          }[key];
          if (step) {
            stand();
            held.add(key);
            if (event.shiftKey) held.add("shift");
            player = stepVisitor(player, yaw, ...step, 0.018, event.shiftKey);
          }
        }
        if (key === "e" && !event.repeat) interact(pick(0, 0, 2.5));
      }
      if (key === "escape") release();
      if (key === "t" && !event.repeat) {
        event.preventDefault();
        interact("greet");
      }
    };
    const keyUp = (event) => held.delete(event.key.toLowerCase());
    const resize = () => {
      width = host.clientWidth;
      height = host.clientHeight;
      if (!width || !height) return;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      portrait = conversationPose(room.conversation, camera.aspect);
      if (!firstPerson) camera.fov = width / height < 0.9 ? 60 : 44;
      camera.updateProjectionMatrix();
      if (inspecting && !firstPerson) inspectEquipment();
    };
    overview();
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    const onHidden = () => {
      if (document.hidden) release();
    };
    const onContextLost = (event) => {
      event.preventDefault();
      release();
      setFailed(true);
    };
    const onBlur = () => release();
    canvas.addEventListener("pointerdown", down, true);
    canvas.addEventListener("pointermove", drag);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", clearLook);
    canvas.addEventListener("lostpointercapture", clearLook);
    canvas.addEventListener("blur", onBlur);
    canvas.addEventListener("webglcontextlost", onContextLost);
    document.addEventListener("pointerlockchange", lockChange);
    document.addEventListener("pointerlockerror", lockError);
    document.addEventListener("mousemove", moveMouse);
    document.addEventListener("keydown", keyDown);
    document.addEventListener("keyup", keyUp);
    document.addEventListener("visibilitychange", onHidden);
    window.addEventListener("blur", release);
    const prepareWalk = () => {
      if (aiming || stateRef.current.dialogueActive) return false;
      stand();
      if (!firstPerson) {
        handoff = null;
        inspecting = false;
        previousView = null;
        controls?.dispose();
        controls = null;
        firstPerson = true;
        setMode("first-person");
      }
      return true;
    };
    apiRef.current = {
      start,
      overview,
      release,
      focus: () => canvas.focus(),
      inspectEquipment,
      restoreView,
      takeSeat,
      setAim(active) {
        if (active === aiming) return;
        release();
        setError("");
        aiming = active;
        if (active) stand();
        canvas.setAttribute("aria-label", active ? LAUNCHER_LABEL : ROOM_LABEL);
        if (active) {
          if (!firstPerson) inspectEquipment();
          canvas.focus();
        } else if (
          !firstPerson &&
          stateRef.current.value.material !== "experiment"
        )
          restoreView();
        if (controls) controls.enabled = !active;
      },
      noteAction,
      resultAction,
      interactCurrent() {
        camera.updateMatrixWorld();
        interact(pick(0, 0, 2.5));
      },
      consumeEscape() {
        if (document.pointerLockElement === canvas) {
          release();
          return true;
        }
        return performance.now() - lastUnlock < 300;
      },
      move(forward, right) {
        if (!prepareWalk()) return;
        player = stepVisitor(player, yaw, forward, right, 0.05, true);
        applyFirstPerson();
      },
      beginWalk(id, forward, right) {
        if (!prepareWalk()) return;
        canvas.focus();
        if (!walkInput.press(id, forward, right)) return;
        const axes = walkInput.axes(held);
        player = stepVisitor(player, yaw, axes.forward, axes.right, .018, held.has("shift"));
        applyFirstPerson();
      },
      endWalk: (id) => walkInput.release(id),
    };
    renderer.setAnimationLoop((now) => {
      const dt =
        previous === null ? 0 : Math.min((now - previous) / 1000, 0.05);
      previous = now;
      if (document.hidden) return;
      const { value, paused, reducedMotion } = stateRef.current;
      motionClock = advanceGestureClock(
        motionClock,
        dt,
        value.gesture.id,
        paused,
        reducedMotion,
      );
      const { elapsed, age } = motionClock;
      const flight = flightRef.current;
      const isFlying = Boolean(flight && !flight.landed);
      if (isFlying !== lastFlying) {
        lastFlying = isFlying;
        setFlying(isFlying);
      }
      const kinetics = kineticsFrame(value.illustration);
      room.presentLanding(value.launcherChallenge, stateRef.current.calculation.busy ? stateRef.current.calculation.context : null);
      if (value.board !== lastBoard || value.resultCards !== lastCards) {
        room.writeBoard(value.board, value.resultCards);
        lastBoard = value.board;
        lastCards = value.resultCards;
      }
      lighting.present(firstPerson);
      room.animate(
        elapsed,
        value.gesture,
        age,
        value.illustration?.parameter,
        firstPerson,
        flight,
        kinetics,
      );
      const talking = firstPerson && stateRef.current.dialogueActive && !aiming;
      if (talking) {
        clear();
        if (portrait) {
          camera.position.copy(portrait.position);
          camera.quaternion.copy(portrait.quaternion);
          if (camera.fov !== portrait.fov) {
            camera.fov = portrait.fov;
            camera.updateProjectionMatrix();
          }
        } else applyFirstPerson();
      } else if (handoff) {
        handoff.elapsed += paused || reducedMotion ? 0.65 : dt;
        const pose = inspectionBlend(handoff.from, handoff.to, handoff.elapsed);
        camera.position.copy(pose.position);
        camera.quaternion.copy(pose.quaternion);
        camera.fov = pose.fov;
        camera.updateProjectionMatrix();
        if (pose.done) {
          const target = handoff.to.target;
          handoff = null;
          acquireOrbit(target, inspecting);
        }
      } else if (firstPerson) {
        const { forward: f, right: r } = walkInput.axes(held);
        if (!aiming && (f || r)) {
          stand();
          player = stepVisitor(player, yaw, f, r, dt, held.has("shift"));
        }
        applyFirstPerson();
      } else if (!aiming) controls?.update();
      camera.updateMatrixWorld();
      const presence = room.presentResearcher({ camera, firstPerson, elapsed, gesture: value.gesture, age, dt, paused, reducedMotion });
      const armPose = room.presentArm({ offered: value.noteOffered, gesture: value.gesture.type, age, dt, paused, reducedMotion });
      host.dataset.researcherArm = armPose.pitch.toFixed(3);
      const coffeePose = room.presentCoffee({ gesture: value.gesture, age, offered: value.noteOffered, paused, reducedMotion });
      host.dataset.coffeePhase = coffeePose.phase;
      host.dataset.coffeeGripError = coffeePose.gripError.toFixed(6);
      host.dataset.coffeePosition = coffeePose.position.toArray().map(v => v.toFixed(3)).join(",");
      host.dataset.researcherGaze = presence ? `${presence.yaw.toFixed(3)},${presence.pitch.toFixed(3)}` : "";
      host.dataset.researcherEyes = presence ? presence.eyeScale.toFixed(3) : "";
      const notePose = room.presentNote({
        resultCard: heldResultCard(value),
        resultRevision: value.resultRevision,
        camera,
        firstPerson,
        conversation: talking && Boolean(portrait),
        held: value.noteHeld,
        offered: value.noteOffered,
        revision: value.noteRevision,
        dt,
        paused,
        reducedMotion,
      });
      const nextNearTable = canReturnNote(seat.position(player), true);
      if (nextNearTable !== lastNearTable) {
        lastNearTable = nextNearTable;
        setNearTable(nextNearTable);
        if (nextNearTable)
          setError((current) =>
            current === "Move closer to the coffee table to put the note down."
              ? ""
              : current,
          );
      }
      room.group.updateMatrixWorld(true);
      const nextNearBoard = canPinResult(seat.position(player), true);
      if (nextNearBoard !== lastNearBoard) {
        lastNearBoard = nextNearBoard;
        setNearBoard(nextNearBoard);
        if (nextNearBoard)
          setError((current) =>
            current === "Move closer to the shared board to pin this result."
              ? ""
              : current,
          );
      }
      const nextTarget = firstPerson ? pick(0, 0, 2.5) : null;
      if (nextTarget !== aimed) {
        aimed = nextTarget;
        setTarget(aimed);
      }
      for (const [id, entry] of room.items) {
        const label = labels.current.get(id);
        if (!label) continue;
        projected.copy(entry.anchor).project(camera);
        const show =
          !firstPerson &&
          !(id === "paper" && value.noteHeld) &&
          Math.abs(projected.x) < 0.9 &&
          Math.abs(projected.y) < 0.9;
        label.style.visibility = show ? "visible" : "hidden";
        label.tabIndex = show ? 0 : -1;
        label.style.transform = `translate(-50%,-50%) translate(${Math.max(65, Math.min(width - 65, ((projected.x + 1) * width) / 2))}px,${((1 - projected.y) * height) / 2}px)`;
      }
      host.dataset.visitor = `${player.x.toFixed(2)},${player.z.toFixed(2)}`;
      host.dataset.walkPointers = String(walkInput.size);
      host.dataset.posture = seat.seated ? "seated" : "standing";
      host.dataset.launcherActive = String(aiming);
      host.dataset.launchAngle = String(value.illustration.parameter ?? 45);
      host.dataset.noteHeld = String(value.noteHeld);
      host.dataset.noteOffered = String(value.noteOffered);
      host.dataset.resultHeld = String(value.heldResultId ?? "");
      host.dataset.pinnedResults = (value.board.pinnedResultIds || []).join(
        ",",
      );
      host.dataset.notePosition = room.paper.position
        .toArray()
        .map((v) => v.toFixed(3))
        .join(",");
      host.dataset.noteSettled = String(notePose.done);
      host.dataset.flightProgress = flight ? String(flight.fraction) : "";
      host.dataset.challengeTarget = value.launcherChallenge.targetId || "";
      host.dataset.challengeShot = value.launcherChallenge.shots.at(-1)?.status || "";
      host.dataset.robotAsset = robotTemplate ? studio.id : "fallback";
      host.dataset.kineticSample = kinetics ? String(kinetics.index) : "";
      host.dataset.kineticValue = kinetics ? String(kinetics.own) : "";
      host.dataset.flightPosition = flight
        ? `${flight.x.toFixed(3)},${flight.y.toFixed(3)}`
        : "";
      host.dataset.camera = camera.position
        .toArray()
        .map((v) => v.toFixed(2))
        .join(",");
      host.dataset.look = `${yaw.toFixed(3)},${pitch.toFixed(3)}`;
      host.dataset.cameraFov = camera.fov.toFixed(3);
      host.dataset.gesture = value.gesture.type;
      host.dataset.cameraOwner = talking && portrait ? "conversation-portrait" : handoff
        ? "handoff"
        : aiming
          ? "launcher-aim"
          : firstPerson
            ? seat.seated ? "seated-look" : "pointer-look"
            : inspecting
              ? "inspection-orbit"
              : "overview-orbit";
      host.dataset.handoff = String(Math.min(age / 1.2, 1));
      renderer.render(scene, camera);
    });
    return () => {
      release();
      apiRef.current = null;
      renderer.setAnimationLoop(null);
      controls?.dispose();
      observer.disconnect();
      canvas.removeEventListener("pointerdown", down, true);
      canvas.removeEventListener("pointermove", drag);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", clearLook);
      canvas.removeEventListener("lostpointercapture", clearLook);
      canvas.removeEventListener("blur", onBlur);
      canvas.removeEventListener("webglcontextlost", onContextLost);
      document.removeEventListener("pointerlockchange", lockChange);
      document.removeEventListener("pointerlockerror", lockError);
      document.removeEventListener("mousemove", moveMouse);
      document.removeEventListener("keydown", keyDown);
      document.removeEventListener("keyup", keyUp);
      document.removeEventListener("visibilitychange", onHidden);
      window.removeEventListener("blur", release);
      lighting.dispose();
      disposeScene(scene);
      environment?.dispose();
      renderer.dispose();
      canvas.remove();
    };
  }, [studio, apiRef, robotTemplate]);
  useEffect(() => {
    apiRef.current?.setAim(value.launcherActive);
    if (value.launcherActive) return;
    if (value.material === "experiment") apiRef.current?.inspectEquipment();
    else apiRef.current?.restoreView();
  }, [value.material, value.launcherActive]);
  const cameraTools = (
      <div className="meeting-camera-tools">
        <button onClick={() => apiRef.current?.start()}>
          <Icon name="play" size={15} />
          {locked
            ? "Playing · Esc to release"
            : mode === "first-person"
              ? "Capture mouse"
              : "Enter first-person"}
        </button>
        <button
          onClick={() => apiRef.current?.overview()}
          aria-label="Room overview"
        >
          <Icon name="compass" size={17} />
        </button>
        {mode === "first-person" && !value.launcherActive && !failed && (
          <button onClick={() => apiRef.current?.takeSeat()} aria-keyshortcuts="G">
            <Icon name="coffee" size={16} />
            {seated ? "Stand up" : "Take a seat"}
          </button>
        )}
        {studio.id === "physics" && !value.launcherActive && (
          <button
            onClick={() => dispatch({ type: "launcher", action: "open" })}
            aria-label="Use launcher"
          >
            <Icon name="physics" size={16} />
            Use launcher
          </button>
        )}
        {value.material === "experiment" && mode !== "first-person" && (
          <button
            onClick={() => apiRef.current?.inspectEquipment()}
            aria-label="Inspect equipment"
          >
            <Icon name={studio.id} size={16} />
            Equipment view
          </button>
        )}
      </div>
  );
  return (
    <>
    <div
      className={`meeting-scene-wrap ${compactHud ? "is-play-view" : ""} ${value.launcherActive ? "is-aiming" : ""}`}
      data-view={mode}
      data-note-held={value.noteHeld}
      data-result-held={Boolean(carriedResult)}
      inert={dialogueActive ? "" : undefined}
    >
      {compactHud ? <RoomControls apiRef={apiRef} actions={actions} instructions={`${instructions} · Hold an arrow button to walk`}>
        {cameraTools}
      </RoomControls> : cameraTools}
      <div
        ref={hostRef}
        className="meeting-game-canvas"
        data-pointer-locked={locked}
      >
        {mode === "first-person" && (
          <div className={`meeting-reticle ${prompt ? "has-target" : ""}`} aria-hidden="true"><i /></div>
        )}
        {!failed &&
          ROOM_ACTIONS.map((action) => (
            <button
              key={action.id}
              className="meeting-hotspot"
              ref={(el) =>
                el
                  ? labels.current.set(action.id, el)
                  : labels.current.delete(action.id)
              }
              onClick={() =>
                action.id === "board" && carriedResult
                  ? apiRef.current?.resultAction("pin")
                  : action.id === "experiment" && studio.id === "physics"
                    ? dispatch({ type: "launcher", action: "open" })
                    : dispatch({ type: "interact", id: action.id })
              }
              aria-label={action.id === "paper" && value.noteOffered ? "Take offered note" : action.label}
            >
              <Icon name={action.icon} size={15} />
              <span>{action.id === "paper" && value.noteOffered ? "Take the note" : action.hint}</span>
            </button>
          ))}
      </div>
      {value.launcherActive && (
        <RoomLauncher
          value={value}
          dispatch={dispatch}
          calculation={calculation}
          flying={flying}
          paused={paused}
        />
      )}
      {carriedResult && (
        <div
          className={`meeting-note-pocket ${value.material ? "is-compact" : ""}`}
          aria-label="Carried result card"
        >
          <span>
            <Icon name="paper" size={16} />
            Result card {carriedResult.id}
            <small>In your hand</small>
          </span>
          <div>
            <button
              onClick={() =>
                apiRef.current && !failed
                  ? apiRef.current.resultAction("read")
                  : dispatch(
                      value.material === "result"
                        ? { type: "close-material" }
                        : { type: "result-card", action: "read" },
                    )
              }
            >
              {value.material === "result" ? "Lower result" : "Read result"}
              <kbd>R</kbd>
            </button>
            <button
              className="meeting-note-return"
              aria-label="Pin carried result to board"
              disabled={!failed && mode === "first-person" && !nearBoard}
              onClick={() =>
                apiRef.current && !failed
                  ? apiRef.current.resultAction("pin")
                  : dispatch({ type: "result-card", action: "pin" })
              }
            >
              Pin to board<kbd>E</kbd>
            </button>
          </div>
          {mode === "first-person" && !nearBoard && !failed && (
            <small>
              Bring this to the shared board to pin it. Q keeps it in your
              collection.
            </small>
          )}
        </div>
      )}
      {value.noteHeld && (
        <div
          className={`meeting-note-pocket ${value.material ? "is-compact" : ""}`}
          aria-label="Carried working note"
        >
          <span>
            <Icon name="paper" size={16} /> Working note{" "}
            <small>In your hand</small>
          </span>
          <div>
            <button
              onClick={() => {
                if (value.material === "paper")
                  dispatch({ type: "close-material" });
                else if (apiRef.current && !failed)
                  apiRef.current.noteAction("read");
                else dispatch({ type: "note", action: "read" });
              }}
            >
              {value.material === "paper" ? "Lower note" : "Read note"}{" "}
              <kbd>R</kbd>
            </button>
            <button
              disabled={!failed && mode === "first-person" && !nearTable}
              className="meeting-note-return"
              aria-label="Put note on table"
              onClick={() =>
                apiRef.current && !failed
                  ? apiRef.current.noteAction("return")
                  : dispatch({ type: "note", action: "return" })
              }
            >
              Put down <kbd>Q</kbd>
            </button>
          </div>
          {mode === "first-person" && !nearTable && !failed && (
            <small>Bring it to the coffee table to put it down.</small>
          )}
        </div>
      )}
      <div className="meeting-control-hint">
        <span role="status">
          {instructions}
        </span>
        {mode === "first-person" && (
          <div className="meeting-hud-actions">
            <div className="meeting-context-action">
              {prompt ? <button aria-label={prompt} onClick={() => apiRef.current?.interactCurrent()}>
                <kbd aria-hidden="true">E</kbd> {prompt}
              </button> : <span className="meeting-context-idle">Look at an object to interact</span>}
            </div>
            <WalkPad apiRef={apiRef} />
          </div>
        )}
      </div>
      {(error || failed) && (
        <p className="meeting-game-error" role="status">
          {failed
            ? "3D is unavailable. Use the action bar below; your discussion and materials still work."
            : error}
        </p>
      )}
    </div>
    {!compactHud && actions}
    </>
  );
}
